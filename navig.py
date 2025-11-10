#!/usr/bin/env python3
"""
ENHANCED ROVER NAVIGATION WITH MiDaS DEPTH INTEGRATION
=========================================================

This module integrates MiDaS monocular depth estimation with feature-based SLAM
for robust pose estimation. Key improvements:

1. Backproject 2D keypoints to 3D using depth + camera intrinsics
2. Validate feature matches geometrically using 3D consistency
3. Improve solvePnP robustness by filtering outliers via depth
4. Reduce false positives in scenes with depth variation

Author: SLAM Navigation Pipeline
Date: 2025-11-10
"""

import cv2
import torch
import math
import numpy as np
import time
import os
import pickle
import json
from typing import Tuple, List, Optional

# =============== DEPTH CONFIGURATION ===============

try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
except Exception:
    DEPTH_SCALE = 5.0

print("🔧 Initializing MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

if torch.cuda.is_available():
    print(f"✅ CUDA - GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CPU MODE")


# =============== DEPTH UTILITIES ===============

class DepthEstimator:
    """Efficient MiDaS depth estimation wrapper"""

    def __init__(self, model=midas, transform_fn=transform, device_=device):
        self.model = model
        self.transform = transform_fn
        self.device = device_

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB frame

        Args:
            frame: RGB frame (H, W, 3)

        Returns:
            Depth map (H, W) - inverse depth (higher = closer)
        """
        # Resize to match MiDaS input
        h_orig, w_orig = frame.shape[:2]

        # Apply transform
        input_batch = self.transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h_orig, w_orig),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def normalize_depth(self, depth_map: np.ndarray, max_depth: float = 3.0) -> np.ndarray:
        """Normalize and clip depth map"""
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)

        # Convert to metric: invert and scale
        depth_metric = max_depth / (depth_norm + 1e-6)
        return np.clip(depth_metric, 0, max_depth)


class Point3DBackprojector:
    """Backproject 2D image points to 3D using depth and camera intrinsics"""

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)

    def backproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Backproject single 2D pixel to 3D point in camera frame

        Formula: P_cam = depth * K^-1 * [u, v, 1]^T

        Args:
            u, v: Image coordinates (pixels)
            depth: Depth value at (u, v)

        Returns:
            3D point in camera frame (x, y, z)
        """
        pixel_homogeneous = np.array([u, v, 1.0], dtype=np.float32)
        point_3d = depth * (self.K_inv @ pixel_homogeneous)
        return point_3d

    def backproject_keypoints(self, keypoints: List, depth_map: np.ndarray) -> List[np.ndarray]:
        """
        Backproject multiple keypoints to 3D

        Args:
            keypoints: List of cv2.KeyPoint objects
            depth_map: Depth map (H, W)

        Returns:
            List of 3D points (N, 3) in camera frame
        """
        points_3d = []
        for kp in keypoints:
            u, v = int(kp.pt[0]), int(kp.pt[1])

            # Boundary check
            if 0 <= u < depth_map.shape[1] and 0 <= v < depth_map.shape[0]:
                depth = depth_map[v, u]
                if depth > 0:
                    p3d = self.backproject(kp.pt[0], kp.pt[1], depth)
                    points_3d.append(p3d)
                else:
                    points_3d.append(None)
            else:
                points_3d.append(None)

        return points_3d


class Depth3DMatchValidator:
    """Validate feature matches using 3D geometric consistency"""

    def __init__(self, reprojection_threshold: float = 0.1, 
                 depth_consistency_threshold: float = 0.15):
        self.reproj_thresh = reprojection_threshold
        self.depth_consistency_thresh = depth_consistency_threshold

    def validate_match_geometric_3d(self, 
                                    match: cv2.DMatch,
                                    kp1: cv2.KeyPoint,
                                    kp2: cv2.KeyPoint,
                                    points_3d_1: List[np.ndarray],
                                    points_3d_2: List[np.ndarray],
                                    R: np.ndarray,
                                    t: np.ndarray) -> bool:
        """
        Validate a match using 3D geometric constraints

        Steps:
        1. Get 3D points in both frames
        2. Transform point from frame 1 to frame 2
        3. Check reprojection error and depth consistency

        Args:
            match: cv2.DMatch object
            kp1, kp2: KeyPoints in frame 1 and 2
            points_3d_1, points_3d_2: Backprojected 3D points
            R, t: Rotation and translation between frames

        Returns:
            True if match is geometrically valid
        """
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        # Check if both points were successfully backprojected
        if (idx1 >= len(points_3d_1) or idx2 >= len(points_3d_2) or
            points_3d_1[idx1] is None or points_3d_2[idx2] is None):
            return False

        p3d_1 = points_3d_1[idx1]
        p3d_2 = points_3d_2[idx2]

        # Depth consistency check: depth should be positive and not jump too much
        depth1 = p3d_1[2]
        depth2 = p3d_2[2]

        if depth1 <= 0 or depth2 <= 0:
            return False

        # Allow depth variation up to threshold
        depth_ratio = max(depth1, depth2) / (min(depth1, depth2) + 1e-6)
        if depth_ratio > (1.0 + self.depth_consistency_thresh):
            return False

        # Transform p3d_1 to frame 2 and check consistency
        try:
            p3d_1_in_frame2 = R @ p3d_1 + t
        except:
            return False

        # Distance check: should be similar
        euclidean_dist = np.linalg.norm(p3d_1_in_frame2 - p3d_2)
        if euclidean_dist > self.reproj_thresh:
            return False

        return True

    def filter_matches_3d(self,
                          matches: List[cv2.DMatch],
                          kp1: List[cv2.KeyPoint],
                          kp2: List[cv2.KeyPoint],
                          points_3d_1: List[np.ndarray],
                          points_3d_2: List[np.ndarray],
                          R: Optional[np.ndarray] = None,
                          t: Optional[np.ndarray] = None) -> List[cv2.DMatch]:
        """
        Filter matches based on 3D geometric constraints

        Args:
            matches: List of cv2.DMatch objects
            kp1, kp2: KeyPoints in both frames
            points_3d_1, points_3d_2: Backprojected 3D points
            R, t: Optional pose (if None, only depth consistency checked)

        Returns:
            Filtered list of geometrically valid matches
        """
        valid_matches = []

        for match in matches:
            if R is not None and t is not None:
                if self.validate_match_geometric_3d(match, kp1[match.queryIdx], 
                                                    kp2[match.trainIdx],
                                                    points_3d_1, points_3d_2, R, t):
                    valid_matches.append(match)
            else:
                # Depth-only validation
                idx1, idx2 = match.queryIdx, match.trainIdx
                if (idx1 < len(points_3d_1) and idx2 < len(points_3d_2) and
                    points_3d_1[idx1] is not None and points_3d_2[idx2] is not None):

                    depth_ratio = max(points_3d_1[idx1][2], points_3d_2[idx2][2]) / \
                                  (min(points_3d_1[idx1][2], points_3d_2[idx2][2]) + 1e-6)
                    if depth_ratio <= (1.0 + self.depth_consistency_thresh):
                        valid_matches.append(match)

        return valid_matches


# =============== KeyFrame Class (ENHANCED) ===============

class KeyFrame:
    def __init__(self, frame_id: int, pose: np.ndarray, features_3d: List,
                 descriptors: np.ndarray, keypoints_2d: List, 
                 fx: float, fy: float, cx: float, cy: float,
                 depth_scale: float = DEPTH_SCALE, max_depth: float = 3.0,
                 depth_map: Optional[np.ndarray] = None):
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
        self.depth_map = depth_map  # Store depth map for validation
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    @staticmethod
    def from_dict(data):
        descriptors = None
        if data.get('descriptors') is not None:
            descriptors = np.array(data['descriptors'], dtype=np.float32)

        kf = KeyFrame(
            frame_id=data['id'],
            pose=np.array(data['pose']),
            features_3d=data.get('features_3d', []),
            descriptors=descriptors,
            keypoints_2d=data.get('keypoints_2d', []),
            fx=data.get('fx', 600.0),
            fy=data.get('fy', 600.0),
            cx=data.get('cx', 320.0),
            cy=data.get('cy', 240.0),
            depth_scale=DEPTH_SCALE,
            max_depth=3.0,
            depth_map=None
        )
        kf.timestamp = data.get('timestamp', time.time())
        return kf


# =============== Storage Manager ===============

class KeyframeStorageManager:
    def __init__(self, storage_dir: str = "keyframes_storage"):
        self.storage_dir = storage_dir
        self.index_file = os.path.join(self.storage_dir, "keyframes_index.json")
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {'keyframes': {}}
        return {'keyframes': {}}

    def get_all_keyframes(self):
        return sorted([int(kf_id) for kf_id in self.index.get('keyframes', {}).keys()])

    def load_keyframe(self, kf_id: int):
        try:
            if str(kf_id) not in self.index.get('keyframes', {}):
                return None
            filename = self.index['keyframes'][str(kf_id)]['filename']
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return KeyFrame.from_dict(data)
        except Exception:
            return None


# =============== Enhanced Rover Navigation ===============

class RoverNavigation:
    def __init__(self, fx: float = 600.0, fy: float = 600.0, 
                 cx: float = 320.0, cy: float = 240.0,
                 forward_step: float = 0.3, max_jump_distance: float = 2.0,
                 depth_scale: float = DEPTH_SCALE):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        self.storage_manager = KeyframeStorageManager()
        self.all_keyframes = {}
        self.keyframe_trajectory = []

        self.current_rover_pose = None
        self.previous_rover_pose = None
        self.current_matched_kf = None
        self.total_distance = 0.0
        self.goal_keyframe = None

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create(
            nfeatures=300,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )

        # NEW: Depth utilities
        self.depth_estimator = DepthEstimator()
        self.backprojector = Point3DBackprojector(fx, fy, cx, cy)
        self.match_validator = Depth3DMatchValidator()

        self.forward_step_distance = forward_step
        self.forward_steps_count = 0
        self.max_jump_distance = max_jump_distance
        self.depth_scale = depth_scale

        self.load_map()

    def load_map(self) -> bool:
        all_kf_ids = self.storage_manager.get_all_keyframes()
        if not all_kf_ids:
            print("❌ No keyframes found!")
            return False

        print(f"📚 Loading {len(all_kf_ids)} keyframes...")
        for kf_id in all_kf_ids:
            kf = self.storage_manager.load_keyframe(kf_id)
            if kf:
                self.all_keyframes[kf_id] = kf
                self.keyframe_trajectory.append(kf.pose.copy())

        print(f"✅ Map loaded: {len(self.all_keyframes)} keyframes")
        return True

    def feature_extraction(self, gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Extract SIFT features from grayscale frame"""
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)

        # Create mask to focus on lower 2/3 of image
        img_height, img_width = gray.shape
        mask = np.zeros_like(enhanced, dtype=np.uint8)
        h_start = int(img_height * 0.33)
        mask[h_start:, :] = 255

        kp, desc = self.sift.detectAndCompute(enhanced, mask=mask)

        if kp is None or desc is None or len(kp) == 0:
            return [], None

        pairs = list(zip(kp, desc))
        pairs.sort(key=lambda x: -x[0].response)
        top = pairs[:min(len(pairs), 300)]

        kp = [p[0] for p in top]
        desc = np.array([p[1] for p in top])

        return kp, desc

    def estimate_depth_current_frame(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for current frame using MiDaS"""
        print("  🌊 Estimating depth with MiDaS...")
        depth_map = self.depth_estimator.estimate_depth(frame)
        depth_map_metric = self.depth_estimator.normalize_depth(depth_map)
        return depth_map_metric

    def match_frame_to_candidates_with_depth(self, 
                                            gray: np.ndarray,
                                            frame_color: np.ndarray,
                                            current_kf_id: int) -> Tuple[Optional[int], float, int]:
        """
        Match current frame to candidate keyframes using depth-validated matching

        ALGORITHM:
        1. Extract features from current frame
        2. Estimate depth for current frame
        3. Backproject current frame keypoints to 3D
        4. For each candidate keyframe:
           a. Match features with keyframe
           b. Filter matches using 3D depth consistency
           c. Run solvePnP with filtered inliers
           d. Score matches
        5. Return best match

        Args:
            gray: Grayscale current frame
            frame_color: Color current frame
            current_kf_id: Current matched keyframe ID

        Returns:
            (best_kf_id, confidence, inlier_count)
        """
        # Extract current frame features
        kp_curr, desc_curr = self.feature_extraction(gray)
        if desc_curr is None or len(desc_curr) == 0:
            return None, 0, 0

        # Estimate depth for current frame
        depth_curr = self.estimate_depth_current_frame(frame_color)

        # Backproject current frame keypoints to 3D
        points_3d_curr = self.backprojector.backproject_keypoints(kp_curr, depth_curr)

        # Get candidate keyframes
        all_ids = sorted(self.all_keyframes.keys())
        try:
            current_idx = all_ids.index(current_kf_id)
        except ValueError:
            return None, 0, 0

        candidates = []
        for i in range(1, 4):
            if current_idx + i < len(all_ids):
                candidates.append(all_ids[current_idx + i])

        if not candidates:
            return None, 0, 0

        # Matcher - NORM_L2 for SIFT descriptors
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches_info = []

        for kf_id in candidates:
            kf = self.all_keyframes.get(kf_id)
            if kf is None or kf.descriptors is None:
                continue

            try:
                kf_desc = np.array(kf.descriptors, dtype=np.float32)
                desc_curr_float32 = np.array(desc_curr, dtype=np.float32)
                matches = bf.knnMatch(desc_curr_float32, kf_desc, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 5:
                continue

            # DEPTH-BASED VALIDATION: Filter matches using 3D consistency
            print(f"  🔍 Validating matches for KF#{kf_id} with depth...")
            valid_matches = self.match_validator.filter_matches_3d(
                good_matches, kp_curr, kf.keypoints_2d if kf.keypoints_2d else [],
                points_3d_curr, kf.features_3d if kf.features_3d else []
            )

            if len(valid_matches) < 5:
                print(f"     ⚠️ Too few valid matches after depth filtering: {len(valid_matches)}")
                continue

            # Prepare 3D-2D correspondence
            object_points = []
            image_points = []

            for match in valid_matches:
                kf_feat_idx = match.trainIdx
                curr_feat_idx = match.queryIdx

                if (curr_feat_idx < len(points_3d_curr) and 
                    kf_feat_idx < len(kf.features_3d)):

                    if points_3d_curr[curr_feat_idx] is None:
                        continue

                    # Use backprojected 3D point from current frame
                    p3d_curr = points_3d_curr[curr_feat_idx]

                    # Project to keyframe coordinate system
                    # (simplified: just use 3D distance)
                    object_points.append(p3d_curr)
                    image_points.append(kp_curr[curr_feat_idx].pt)

            if len(object_points) < 5:
                continue

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            # solvePnP with improved robustness
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_points, image_points, self.K, None,
                    iterationsCount=200,  # Increased iterations
                    reprojectionError=6.0,  # Reduced threshold (tighter)
                    confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            except:
                continue

            if success and inliers is not None:
                inlier_count = len(inliers)
                confidence = len(valid_matches) / (len(kf.descriptors) + 1e-6)

                print(f"     ✅ KF#{kf_id}: {inlier_count} inliers, conf={confidence:.2f}")

                matches_info.append({
                    'kf_id': kf_id,
                    'inliers': inlier_count,
                    'confidence': confidence,
                    'good_matches': len(valid_matches)
                })

        if not matches_info:
            return None, 0, 0

        # Sort by inliers (primary), confidence (secondary)
        matches_info.sort(key=lambda x: (x['inliers'], x['confidence']), reverse=True)
        best = matches_info[0]

        return best['kf_id'], best['confidence'], best['inliers']

    def get_candidate_keyframes(self, current_kf_id: int) -> List[int]:
        """Get 3 forward keyframes only"""
        all_ids = sorted(self.all_keyframes.keys())
        try:
            current_idx = all_ids.index(current_kf_id)
        except ValueError:
            return []

        candidates = []
        for i in range(1, 4):
            if current_idx + i < len(all_ids):
                candidates.append(all_ids[current_idx + i])

        return candidates

    def validate_jump_distance(self, keyframe_pose: np.ndarray) -> bool:
        """Check if jump distance is within safety limits"""
        if self.current_rover_pose is None:
            return True

        dx = keyframe_pose[0] - self.current_rover_pose[0]
        dy = keyframe_pose[1] - self.current_rover_pose[1]
        jump_distance = math.sqrt(dx**2 + dy**2)

        return jump_distance <= self.max_jump_distance

    def move_forward_incrementally(self, steps: int = 1):
        """Move rover forward incrementally"""
        if self.current_rover_pose is None:
            return

        self.previous_rover_pose = self.current_rover_pose.copy()
        current_yaw = self.current_rover_pose[2]

        for _ in range(steps):
            self.current_rover_pose[0] += self.forward_step_distance * math.cos(current_yaw)
            self.current_rover_pose[1] += self.forward_step_distance * math.sin(current_yaw)
            self.forward_steps_count += 1
            self.total_distance += self.forward_step_distance

    def process_navigation_video(self, video_path: str, start_kf_id: int, 
                                goal_kf_id: int, process_every_n: int = 10):
        """
        Process navigation video with depth-enhanced matching

        Args:
            video_path: Path to video file
            start_kf_id: Starting keyframe ID
            goal_kf_id: Goal keyframe ID
            process_every_n: Process every Nth frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        if start_kf_id not in self.all_keyframes:
            print(f"❌ Starting keyframe {start_kf_id} not found!")
            return

        if goal_kf_id not in self.all_keyframes:
            print(f"❌ Goal keyframe {goal_kf_id} not found!")
            return

        self.current_matched_kf = start_kf_id
        self.current_rover_pose = self.all_keyframes[start_kf_id].pose.copy()
        self.previous_rover_pose = self.current_rover_pose.copy()
        self.goal_keyframe = goal_kf_id
        self.total_distance = 0.0
        self.forward_steps_count = 0

        print(f"\n🚀 ENHANCED ROVER NAVIGATION WITH DEPTH")
        print(f"📹 Video: {video_path} | FPS: {fps:.1f}")
        print(f"⏭️ Processing every {process_every_n}th frame")
        print(f"📍 Start: KF#{start_kf_id} → Goal: KF#{goal_kf_id}")
        print(f"🌊 DEPTH-ENHANCED MATCHING ENABLED")
        print("=" * 80)

        cv2.namedWindow("NAVIGATION MAP", cv2.WINDOW_NORMAL)
        cv2.namedWindow("CURRENT FRAME", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("NAVIGATION MAP", 900, 900)
        cv2.resizeWindow("CURRENT FRAME", 640, 480)

        frame_count = 0
        process_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Video processing complete!")
                break

            frame_count += 1
            process_counter += 1

            if process_counter < process_every_n:
                continue

            process_counter = 0
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            print(f"\n[Frame {frame_count:5d}] Processing with depth estimation...")

            # Match with depth validation
            matched_kf_id, confidence, inliers = self.match_frame_to_candidates_with_depth(
                gray, frame, self.current_matched_kf
            )

            if matched_kf_id is None or inliers < 5:
                print(f"  ❌ No keyframe detected - Moving forward")
                self.move_forward_incrementally(steps=1)
                print(f"  📍 Rover Pos: ({self.current_rover_pose[0]:.3f}, {self.current_rover_pose[1]:.3f})")
            else:
                kf_pose = self.all_keyframes[matched_kf_id].pose
                if self.validate_jump_distance(kf_pose):
                    print(f"  ✅ MATCHED: KF#{matched_kf_id} | Inliers: {inliers}")
                    self.previous_rover_pose = self.current_rover_pose.copy()
                    self.current_rover_pose = kf_pose.copy()
                    self.current_matched_kf = matched_kf_id

                    dx = self.current_rover_pose[0] - self.previous_rover_pose[0]
                    dy = self.current_rover_pose[1] - self.previous_rover_pose[1]
                    jump_distance = math.sqrt(dx**2 + dy**2)
                    self.total_distance += jump_distance

                    print(f"  📍 Rover at: ({self.current_rover_pose[0]:.3f}, {self.current_rover_pose[1]:.3f})")
                    print(f"  ↔️ Distance: {jump_distance:.3f}m | Total: {self.total_distance:.3f}m")
                else:
                    print(f"  ⚠️ Jump too large - rejecting")
                    self.move_forward_incrementally(steps=1)

            # Draw map
            map_canvas = self.draw_map()
            cv2.imshow("NAVIGATION MAP", map_canvas)
            cv2.imshow("CURRENT FRAME", frame)

            time.sleep(0.2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n⏹️ Navigation stopped by user")
                break

        cap.release()

        # Summary
        print(f"\n{'='*80}")
        print("🎯 NAVIGATION SUMMARY (ENHANCED WITH DEPTH)")
        print(f"{'='*80}")
        print(f"Total Distance: {self.total_distance:.3f}m")
        print(f"Forward Steps: {self.forward_steps_count} x {self.forward_step_distance}m")
        print(f"Final Position: ({self.current_rover_pose[0]:.3f}, {self.current_rover_pose[1]:.3f})")
        print(f"Goal Reached: {'✅ YES' if self.current_matched_kf == goal_kf_id else '❌ NO'}")
        print(f"{'='*80}")

        cv2.destroyAllWindows()

    def draw_map(self, scale: int = 100, size: int = 900) -> np.ndarray:
        """Draw navigation map (existing implementation)"""
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        center_x, center_y = size // 2, size // 2

        if self.keyframe_trajectory:
            all_x = [p[0] for p in self.keyframe_trajectory]
            all_y = [p[1] for p in self.keyframe_trajectory]
            map_center_x = (max(all_x) + min(all_x)) / 2
            map_center_y = (max(all_y) + min(all_y)) / 2
        else:
            map_center_x, map_center_y = 0, 0

        def project(x, y):
            dx = x - map_center_x
            dy = y - map_center_y
            sx = int(center_x + dx * scale)
            sy = int(center_y - dy * scale)
            return sx, sy

        # Draw path
        if len(self.keyframe_trajectory) > 1:
            path_points = []
            for pose in self.keyframe_trajectory:
                px, py = project(pose[0], pose[1])
                if 0 <= px < size and 0 <= py < size:
                    path_points.append((px, py))

            for i in range(len(path_points) - 1):
                cv2.line(canvas, path_points[i], path_points[i+1], (0, 165, 255), 3)

        # Draw keyframes
        for kf_id, kf in self.all_keyframes.items():
            kf_x, kf_y, kf_yaw = kf.pose
            kf_sx, kf_sy = project(kf_x, kf_y)

            if 0 <= kf_sx < size and 0 <= kf_sy < size:
                if kf_id == self.goal_keyframe:
                    color = (0, 0, 255)
                    cv2.circle(canvas, (kf_sx, kf_sy), 14, color, -1)
                    cv2.putText(canvas, "GOAL", (kf_sx + 15, kf_sy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif kf_id == self.current_matched_kf:
                    color = (0, 255, 255)
                    cv2.circle(canvas, (kf_sx, kf_sy), 10, color, -1)
                else:
                    color = (255, 0, 0)
                    cv2.circle(canvas, (kf_sx, kf_sy), 6, color, -1)

        # Draw rover
        if self.current_rover_pose is not None:
            rov_x, rov_y = self.current_rover_pose[0], self.current_rover_pose[1]
            rov_sx, rov_sy = project(rov_x, rov_y)
            if 0 <= rov_sx < size and 0 <= rov_sy < size:
                cv2.circle(canvas, (rov_sx, rov_sy), 12, (0, 255, 0), -1)
                cv2.putText(canvas, "ROVER", (rov_sx + 20, rov_sy - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Stats
        cv2.putText(canvas, "🌊 DEPTH-ENHANCED NAVIGATION", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Distance: {self.total_distance:.3f}m", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return canvas


# =============== Main ===============

def main():
    print("\n" + "="*80)
    print("🚀 ENHANCED ROVER NAVIGATION WITH MiDaS DEPTH")
    print("="*80)

    nav = RoverNavigation(fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                         forward_step=0.3, max_jump_distance=2.0)

    if not nav.all_keyframes:
        print("❌ No keyframes available!")
        return

    all_kf_ids = sorted(nav.all_keyframes.keys())
    print(f"\n📍 Available Keyframes: {all_kf_ids}")

    while True:
        start_kf_input = input("\n🎯 Enter STARTING keyframe number: ").strip()
        try:
            start_kf = int(start_kf_input)
            if start_kf in nav.all_keyframes:
                break
            else:
                print(f"❌ Keyframe {start_kf} not found!")
        except ValueError:
            print("❌ Please enter a valid number!")

    while True:
        goal_kf_input = input("🎯 Enter GOAL keyframe number: ").strip()
        try:
            goal_kf = int(goal_kf_input)
            if goal_kf in nav.all_keyframes:
                break
            else:
                print(f"❌ Keyframe {goal_kf} not found!")
        except ValueError:
            print("❌ Please enter a valid number!")

    video_path = input("\n📹 Enter video file path: ").strip()
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    nav.process_navigation_video(video_path, start_kf, goal_kf, process_every_n=10)


if __name__ == "__main__":
    main()
