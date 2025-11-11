#!/usr/bin/env python3
"""
OPTIMIZED GPU SLAM FOR GOOGLE COLAB
====================================
Optimizations:
1. Mixed precision (FP16) for 2-3x faster depth inference
2. Proper batched depth processing with frame pairing
3. Increased batch size for better GPU utilization
4. GPU memory monitoring and optimization
5. Reduced CPU-GPU data transfers
"""

import cv2
import torch
import torch.cuda.amp as amp
import math
import numpy as np
import time
import os
import pickle
import json
from collections import deque
from typing import List, Tuple, Optional

# =============== Configuration ===============
try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
except Exception:
    DEPTH_SCALE = 5.0

print("🔧 Initializing optimized MiDaS...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print("✅ GPU optimizations enabled")

    # Set number of threads for CPU operations
    torch.set_num_threads(4)
    print(f"✅ CPU threads set to: {torch.get_num_threads()}")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
midas.to(device).eval()

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ Memory: {total_mem:.2f} GB")
else:
    print("⚠️ CPU MODE")

# =============== OPTIMIZED GPU DEPTH PROCESSOR ===============
class OptimizedGPUDepthProcessor:
    """GPU depth processor with mixed precision and proper batching"""

    def __init__(self, model, transform_fn, device, batch_size=8):
        self.model = model
        self.transform = transform_fn
        self.device = device
        self.batch_size = batch_size

        # Frame buffer stores (frame, gray, frame_id) tuples
        self.frame_buffer = []
        self.frame_id_counter = 0

        # Use automatic mixed precision for faster inference
        self.use_amp = torch.cuda.is_available()
        self.scaler = amp.GradScaler() if self.use_amp else None

        # Statistics
        self.peak_gpu_memory = 0.0
        self.total_frames_processed = 0
        self.total_depth_time = 0.0

    def process_batch(self, frame_data_list: List[Tuple]) -> List[Tuple]:
        """
        Process batch of frames and return (frame_id, gray, frame_bgr, depth_map)

        Args:
            frame_data_list: List of (frame_bgr, gray, frame_id) tuples

        Returns:
            List of (frame_id, gray, frame_bgr, depth_map) tuples
        """
        if not frame_data_list:
            return []

        start_time = time.time()

        # Prepare batch
        inputs = []
        original_sizes = []

        for frame_bgr, gray, frame_id in frame_data_list:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            original_sizes.append(rgb.shape[:2])
            inp = self.transform(rgb).to(self.device)
            inputs.append(inp)

        # Stack into batch
        batch_inp = torch.stack(inputs, 0)

        # Process with mixed precision
        with torch.no_grad():
            if self.use_amp:
                with amp.autocast():
                    preds = self.model(batch_inp)
            else:
                preds = self.model(batch_inp)

        # Convert predictions to numpy
        results = []
        for i, (frame_bgr, gray, frame_id) in enumerate(frame_data_list):
            pred = preds[i]
            pred_interpolated = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=original_sizes[i],
                mode="bicubic",
                align_corners=False
            ).squeeze()

            depth_map = pred_interpolated.cpu().numpy()
            results.append((frame_id, gray, frame_bgr, depth_map))

        # Update statistics
        elapsed = time.time() - start_time
        self.total_frames_processed += len(frame_data_list)
        self.total_depth_time += elapsed

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            self.peak_gpu_memory = max(self.peak_gpu_memory, allocated)

        return results

    def add_frame(self, frame_bgr: np.ndarray, gray: np.ndarray) -> List[Tuple]:
        """
        Add frame to buffer and process batch if full

        Returns:
            List of processed (frame_id, gray, frame_bgr, depth_map) tuples
        """
        self.frame_buffer.append((frame_bgr, gray, self.frame_id_counter))
        self.frame_id_counter += 1

        if len(self.frame_buffer) >= self.batch_size:
            batch_data = self.frame_buffer
            self.frame_buffer = []
            return self.process_batch(batch_data)

        return []

    def flush(self) -> List[Tuple]:
        """Process remaining frames in buffer"""
        if self.frame_buffer:
            batch_data = self.frame_buffer
            self.frame_buffer = []
            return self.process_batch(batch_data)
        return []

    def get_stats(self) -> dict:
        """Get processing statistics"""
        avg_fps = self.total_frames_processed / self.total_depth_time if self.total_depth_time > 0 else 0
        return {
            'total_frames': self.total_frames_processed,
            'total_time': self.total_depth_time,
            'avg_depth_fps': avg_fps,
            'peak_gpu_memory': self.peak_gpu_memory
        }

# =============== KeyFrame Class ===============
class KeyFrame:
    def __init__(self, frame_id, pose, features_3d, descriptors,
                 keypoints_2d, fx, fy, cx, cy, depth_scale, max_depth):
        self.id = frame_id
        self.pose = pose.copy()
        self.features_3d = features_3d
        self.descriptors = descriptors
        self.keypoints_2d = keypoints_2d
        self.timestamp = time.time()
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.depth_scale = depth_scale
        self.max_depth = max_depth

    def to_dict(self):
        return {
            'id': self.id,
            'pose': self.pose.tolist(),
            'features_3d': self.features_3d,
            'descriptors': self.descriptors.tolist() if self.descriptors is not None else None,
            'keypoints_2d': self.keypoints_2d,
            'timestamp': self.timestamp,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy
        }

    @staticmethod
    def from_dict(data):
        kf = KeyFrame(
            frame_id=data['id'],
            pose=np.array(data['pose']),
            features_3d=data['features_3d'],
            descriptors=np.array(data['descriptors']) if data['descriptors'] is not None else None,
            keypoints_2d=data['keypoints_2d'],
            fx=data['fx'],
            fy=data['fy'],
            cx=data['cx'],
            cy=data['cy'],
            depth_scale=DEPTH_SCALE,
            max_depth=3.0
        )
        kf.timestamp = data['timestamp']
        return kf

# =============== Storage Manager ===============
class KeyframeStorageManager:
    def __init__(self, storage_dir="keyframes_storage"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        self.index_file = os.path.join(self.storage_dir, "keyframes_index.json")
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {'keyframes': {}}

    def _save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def save_keyframe(self, keyframe):
        try:
            filename = os.path.join(self.storage_dir, f"keyframe_{keyframe.id:06d}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(keyframe.to_dict(), f)

            self.index['keyframes'][str(keyframe.id)] = {
                'filename': filename,
                'pose': keyframe.pose.tolist(),
                'timestamp': keyframe.timestamp,
                'num_features': len(keyframe.features_3d) if keyframe.features_3d is not None else 0
            }

            self._save_index()
            return True
        except Exception as e:
            print(f"❌ Error saving keyframe: {e}")
            return False

# =============== OPTIMIZED SLAM ===============
class OptimizedSLAM:
    """Optimized SLAM with proper batched depth processing"""

    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                 rotation_sensitivity=8.0, forward_step=0.025,
                 distance_threshold=0.5, depth_processor=None):

        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

        self.depth_processor = depth_processor

        # Pose tracking
        self.pose = np.zeros(3, dtype=float)
        self.pose_at_last_keyframe = np.zeros(3, dtype=float)
        self.keyframe_trajectory = []
        self.path_distance = 0.0

        # Motion parameters
        self.FIXED_FORWARD_DISTANCE = forward_step
        self.ROTATION_SENSITIVITY = rotation_sensitivity
        self.motion_mode = "REST"

        # SIFT detector
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.04)

        # Keyframe management
        self.storage_manager = KeyframeStorageManager()
        self.keyframes_info = {}
        self.current_keyframe = None
        self.keyframe_counter = 0
        self.first_frame_initialized = False

        # Tracking
        self.prev_gray = None
        self.prev_keypoint_positions = []
        self.total_distance_traveled = 0.0
        self.total_distance_at_last_keyframe = 0.0
        self.distance_threshold = distance_threshold

        # Statistics
        self.frame_count = 0

    def feature_extraction(self, gray):
        """Extract SIFT features"""
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)

        # Mask lower 2/3
        h, w = gray.shape
        mask = np.zeros_like(enhanced, dtype=np.uint8)
        mask[int(h * 0.33):, :] = 255

        kp, desc = self.sift.detectAndCompute(enhanced, mask=mask)

        if kp is None or desc is None or len(kp) == 0:
            return [], None

        # Sort by response
        pairs = list(zip(kp, desc))
        pairs.sort(key=lambda x: -x[0].response)
        pairs = pairs[:min(len(pairs), 300)]

        kp = [p[0] for p in pairs]
        desc = np.array([p[1] for p in pairs])

        return kp, desc

    def _midas_to_metric_depth(self, midas_value):
        if midas_value < 1e-3:
            return None
        return DEPTH_SCALE / (midas_value + 1e-6)

    def _backproject_to_world(self, u, v, metric_depth, pose):
        if metric_depth is None or metric_depth <= 0:
            return None

        X_cam = (u - self.cx) * metric_depth / self.fx
        Y_cam = (v - self.cy) * metric_depth / self.fy
        Z_cam = metric_depth

        yaw = pose[2]
        c, s = math.cos(yaw), math.sin(yaw)

        x_world = pose[0] + (c * Z_cam + s * X_cam)
        y_world = pose[1] + (s * Z_cam - c * X_cam)

        return (x_world, y_world, Z_cam)

    def _detect_motion_state(self, gray):
        """Detect if camera is rotating or moving forward"""
        if self.prev_gray is None or len(self.prev_keypoint_positions) == 0:
            return False, False

        prev_pts = np.array(self.prev_keypoint_positions, dtype=np.float32).reshape(-1, 1, 2)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3
        )

        if status is None or np.sum(status) < 5:
            return False, False

        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        displacements_h = good_next[:, 0, 0] - good_prev[:, 0, 0]
        displacements_v = good_next[:, 0, 1] - good_prev[:, 0, 1]

        median_h = np.median(displacements_h)
        median_v = np.median(displacements_v)

        has_rotation = abs(median_h) > 1.5
        has_forward_motion = median_v > 0.5

        return has_forward_motion, has_rotation

    def update(self, gray, frame_bgr, depth_map):
        """Update SLAM state with frame and depth"""
        self.frame_count += 1
        h, w = gray.shape

        # Motion detection
        if self.prev_gray is not None:
            has_forward, has_rotation = self._detect_motion_state(gray)

            if has_rotation:
                self.motion_mode = "ROTATING"
                # Update rotation
                u_look = int(self.cx)
                v_look = int(h * 0.9)

                prev_pt = np.array([[[u_look, v_look]]], dtype=np.float32)
                next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, prev_pt, None, winSize=(21, 21)
                )

                if status is not None and status[0][0] == 1:
                    u_next, _ = next_pt[0][0]
                    delta_u = u_next - u_look
                    yaw_delta = (delta_u / self.fx) * self.ROTATION_SENSITIVITY
                    self.pose[2] += yaw_delta

            elif has_forward:
                self.motion_mode = "FORWARD"
                # Move forward
                yaw = self.pose[2]
                self.pose[0] += self.FIXED_FORWARD_DISTANCE * math.cos(yaw)
                self.pose[1] += self.FIXED_FORWARD_DISTANCE * math.sin(yaw)
                self.path_distance += self.FIXED_FORWARD_DISTANCE
                self.total_distance_traveled += self.FIXED_FORWARD_DISTANCE
            else:
                self.motion_mode = "REST"

        # Feature extraction
        kp, desc = self.feature_extraction(gray)

        # Build 3D features
        features_3d = []
        keypoints_2d = []

        for kp_pt in kp:
            u, v = int(kp_pt.pt[0]), int(kp_pt.pt[1])
            if 0 <= u < w and 0 <= v < h and depth_map is not None:
                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)

                if metric_depth is not None and metric_depth <= 3.0:
                    world_pos = self._backproject_to_world(u, v, metric_depth, self.pose)
                    if world_pos is not None:
                        features_3d.append(world_pos)
                        keypoints_2d.append((u, v))

        # Keyframe decision
        if not self.first_frame_initialized:
            self._create_keyframe(features_3d, desc, keypoints_2d)
            self.first_frame_initialized = True
            print("🔷 First keyframe")
        else:
            dist_since_kf = self.total_distance_traveled - self.total_distance_at_last_keyframe
            if dist_since_kf >= self.distance_threshold:
                self._create_keyframe(features_3d, desc, keypoints_2d)

        # Update tracking
        self.prev_gray = gray
        self.prev_keypoint_positions = [(kp_pt.pt[0], kp_pt.pt[1]) for kp_pt in kp]

    def _create_keyframe(self, features_3d, desc, keypoints_2d):
        """Create new keyframe"""
        if self.current_keyframe is not None:
            self.storage_manager.save_keyframe(self.current_keyframe)
            self.keyframes_info[str(self.current_keyframe.id)] = {
                'pose': self.current_keyframe.pose.tolist(),
                'num_features': len(self.current_keyframe.features_3d) if self.current_keyframe.features_3d else 0,
                'timestamp': self.current_keyframe.timestamp
            }

        keyframe = KeyFrame(
            frame_id=self.keyframe_counter,
            pose=self.pose,
            features_3d=features_3d,
            descriptors=desc,
            keypoints_2d=keypoints_2d,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            depth_scale=DEPTH_SCALE,
            max_depth=3.0
        )

        self.current_keyframe = keyframe
        self.pose_at_last_keyframe = self.pose.copy()
        self.total_distance_at_last_keyframe = self.total_distance_traveled
        self.keyframe_counter += 1
        self.keyframe_trajectory.append(self.pose.copy())

        dist_traveled = self.total_distance_traveled - (self.total_distance_at_last_keyframe - self.distance_threshold if self.keyframe_counter > 1 else 0)
        print(f"✅ KF#{keyframe.id} | Dist traveled: {dist_traveled:.3f}m | Total: {self.path_distance:.2f}m")

    def draw_top_view(self, scale=100, size=900):
        """Draw top-down view"""
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        center_x, center_y = size // 2, int(size * 0.7)

        def project(x, y):
            dx = x - self.pose[0]
            dy = y - self.pose[1]
            sx = int(center_x + dx * scale)
            sy = int(center_y - dy * scale)
            return sx, sy

        # Draw trajectory
        if len(self.keyframe_trajectory) > 1:
            points = []
            for (x, y, _) in self.keyframe_trajectory:
                px, py = project(x, y)
                if 0 <= px < size and 0 <= py < size:
                    points.append((px, py))

            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i+1], (0, 165, 255), 3)

        # Draw keyframes
        for kf_id_str, kf_info in self.keyframes_info.items():
            kf_pose = np.array(kf_info['pose'])
            x, y, yaw = kf_pose
            px, py = project(x, y)

            if 0 <= px < size and 0 <= py < size:
                cv2.circle(canvas, (px, py), 7, (255, 0, 0), -1)

        # Draw current position
        cv2.circle(canvas, (center_x, center_y), 8, (0, 255, 0), -1)

        # Info
        cv2.putText(canvas, f"Distance: {self.path_distance:.2f}m", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Keyframes: {len(self.keyframes_info)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return canvas

# =============== MAIN PROCESSING FUNCTION ===============
def run_optimized_slam(video_path: str,
                      target_w: int = 1280,
                      target_h: int = 720,
                      batch_size: int = 8,
                      rotation_sensitivity: float = 8.0,
                      forward_step: float = 0.025,
                      distance_threshold: float = 0.5):
    """
    Run optimized SLAM with batched depth processing

    Args:
        video_path: Path to video
        target_w, target_h: Resolution
        batch_size: Depth processing batch size (higher = better GPU utilization)
        rotation_sensitivity: Rotation scaling factor
        forward_step: Forward motion step per frame
        distance_threshold: Distance between keyframes
    """
    print("\n" + "="*80)
    print("🚀 OPTIMIZED SLAM")
    print("="*80)
    print(f"Resolution: {target_w}x{target_h}")
    print(f"Batch size: {batch_size}")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"Forward step: {forward_step}m/frame")
    print(f"Rotation sensitivity: {rotation_sensitivity}")
    print("="*80 + "\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Camera intrinsics
    cx, cy = target_w / 2.0, target_h / 2.0
    fx, fy = target_w * 0.9, target_h * 0.9

    # Create depth processor
    depth_processor = OptimizedGPUDepthProcessor(
        model=midas,
        transform_fn=transform,
        device=device,
        batch_size=batch_size
    )

    # Create SLAM
    slam = OptimizedSLAM(
        fx=fx, fy=fy, cx=cx, cy=cy,
        rotation_sensitivity=rotation_sensitivity,
        forward_step=forward_step,
        distance_threshold=distance_threshold,
        depth_processor=depth_processor
    )

    # Processing loop
    fps = 0.0
    prev_time = time.time()
    frame_counter = 0

    # Queue to hold frames waiting for depth
    pending_frames = deque()

    print("🎬 Processing video...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Flush remaining frames
            remaining = depth_processor.flush()
            for (frame_id, gray, frame_bgr, depth_map) in remaining:
                slam.update(gray, frame_bgr, depth_map)
            print("\n✅ Reached end of video!")
            break

        frame_counter += 1
        frame_resized = cv2.resize(frame, (target_w, target_h))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Add to depth processor
        processed_batch = depth_processor.add_frame(frame_resized, gray)

        # Process any completed depth maps
        for (frame_id, gray_processed, frame_bgr, depth_map) in processed_batch:
            slam.update(gray_processed, frame_bgr, depth_map)

        # Update FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else 1.0 / dt

        # Print progress every 8 frames
        if frame_counter % 8 == 0:
            progress_pct = (frame_counter / total_frames) * 100
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"Frame: {frame_counter}/{total_frames} ({progress_pct:.1f}%) | "
                  f"FPS: {fps:.1f} | GPU: {gpu_mem:.2f}GB | "
                  f"Dist: {slam.path_distance:.2f}m | KFs: {len(slam.keyframes_info)}")

    cap.release()

    # Final statistics
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)

    depth_stats = depth_processor.get_stats()
    print(f"Total frames processed: {frame_counter}")
    print(f"Total keyframes: {len(slam.keyframes_info)}")
    print(f"Total distance: {slam.path_distance:.2f}m")
    print(f"Avg depth FPS: {depth_stats['avg_depth_fps']:.1f}")
    print(f"Peak GPU memory: {depth_stats['peak_gpu_memory']:.2f}GB")

    if torch.cuda.is_available():
        print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # Save final map
    final_map = slam.draw_top_view(scale=100, size=900)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"slam_optimized_{timestamp}.png"
    cv2.imwrite(output_path, final_map)
    print(f"✅ Map saved: {output_path}")
    print("="*80)

    return slam, depth_processor

# =============== ENTRY POINT ===============
if __name__ == "__main__":
    # Example usage
    slam, processor = run_optimized_slam(
        video_path="your_video.mp4",
        target_w=1280,
        target_h=720,
        batch_size=16,  # Increase for better GPU utilization
        rotation_sensitivity=8.0,
        forward_step=0.025,
        distance_threshold=0.5
    )
