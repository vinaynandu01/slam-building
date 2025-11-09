# ===============================================================================
# HYBRID SLAM SYSTEM - BUILD MAP or LOCALIZE
# ===============================================================================
# Combines:
# - Map Building from tuned.py (excellent keyframe selection)
# - Localization from local.py (excellent feature matching)
# ===============================================================================

import cv2
import torch
import math
import numpy as np
import time
import os
import pickle
import requests
from collections import defaultdict, deque
import json

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

# =============== Stream Capture ===============

class RoverStreamCapture:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.resp = None
        self.bytes_buf = b""
        self.is_opened = False
        self._connect()

    def _connect(self):
        try:
            self.resp = requests.get(self.stream_url, stream=True, timeout=5)
            self.is_opened = (self.resp.status_code == 200)
        except Exception:
            self.is_opened = False

    def isOpened(self):
        return self.is_opened

    def read(self):
        if not self.is_opened:
            return False, None

        try:
            for chunk in self.resp.iter_content(chunk_size=1024):
                self.bytes_buf += chunk
                a = self.bytes_buf.find(b'\xff\xd8')
                b = self.bytes_buf.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = self.bytes_buf[a:b+2]
                    self.bytes_buf = self.bytes_buf[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        return True, img
            return False, None
        except Exception:
            self.is_opened = False
            return False, None

    def release(self):
        if self.resp is not None:
            self.resp.close()
        self.is_opened = False

# =============== Keyframe Class ===============

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
        self.connected_keyframes = set()
        self.map_points = []

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
        descriptors = None
        if data['descriptors'] is not None:
            descriptors = np.array(data['descriptors'], dtype=np.uint8)
        kf = KeyFrame(
            frame_id=data['id'],
            pose=np.array(data['pose']),
            features_3d=data['features_3d'],
            descriptors=descriptors,
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

# =============== Keyframe Storage Manager ===============

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
                'num_features': len(keyframe.features_3d)
            }
            self._save_index()
            return True
        except Exception as e:
            print(f"❌ Error saving keyframe: {e}")
            return False

    def load_keyframe(self, kf_id):
        try:
            if str(kf_id) not in self.index['keyframes']:
                return None
            filename = self.index['keyframes'][str(kf_id)]['filename']
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return KeyFrame.from_dict(data)
        except Exception as e:
            return None

    def get_all_keyframes(self):
        """Get all keyframe IDs"""
        return sorted([int(kf_id) for kf_id in self.index['keyframes'].keys()])

# =============== HYBRID SLAM SYSTEM ===============

class HybridSLAM:
    """
    Combines best of tuned.py (map building) + local.py (localization)
    """
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                 lookahead_height_ratio=0.9,
                 min_features_per_frame=30, max_features_per_frame=150,
                 rotation_sensitivity=3.0, feature_lifetime_frames=50,
                 max_depth_threshold=3.0, spatial_grid_size=0.10,
                 tracked_feature_threshold=20,
                 distance_threshold=0.3,
                 rotation_threshold=30.0,
                 feature_coverage_threshold=0.6,
                 keyframe_cooldown=30,
                 local_map_size=10):

        # Camera params
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

        # Pose tracking
        self.pose = np.zeros(3, dtype=float)
        self.pose_at_last_keyframe = np.zeros(3, dtype=float)
        self.trajectory = []
        self.path_distance = 0.0
        self.meter_markers = []

        # Motion
        self.motion_mode = "REST"
        self.FIXED_FORWARD_DISTANCE = 0.004
        self.ROTATION_SENSITIVITY = rotation_sensitivity
        self.rest_frames_count = 0
        self.rest_frames_threshold = 10

        # Features
        self.min_features_per_frame = min_features_per_frame
        self.max_features_per_frame = max_features_per_frame
        self.current_feature_target = max_features_per_frame
        self.spatial_grid_size = spatial_grid_size
        self.spatial_grid = {}
        self.max_map_features = 15000
        self.feature_lifetime_frames = feature_lifetime_frames
        self.visible_map_points = []
        self.max_depth_threshold = max_depth_threshold

        # Keyframe thresholds from tuned.py
        self.tracked_feature_threshold = tracked_feature_threshold
        self.distance_threshold = distance_threshold
        self.rotation_threshold = math.radians(rotation_threshold)
        self.feature_coverage_threshold = feature_coverage_threshold
        self.keyframe_cooldown = keyframe_cooldown
        self.frames_since_last_keyframe = 0

        # Feature tracking
        self.successfully_tracked_count = 0

        # Loop closure
        self.loop_closure_history = []
        self.loop_closure_threshold = 0.90
        self.loop_closure_cooldown = 0
        self.loop_closure_cooldown_frames = 100

        # Keyframe management
        self.storage_manager = KeyframeStorageManager()
        self.keyframes_info = {}
        self.current_keyframe = None
        self.prev_keyframe = None
        self.keyframe_counter = 0
        self.first_frame_initialized = False
        self.local_map_size = local_map_size

        # Feature extraction and tracking
        self.sift = cv2.SIFT_create()
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=6, table_number=12, key_size=20),
                                          dict(checks=50))
        self.prev_gray = None
        self.current_features = []
        self.prev_keypoint_positions = []
        self.frame_count = 0

    def feature_extraction(self, gray, depth_map):
        """Extract features from image"""
        try:
            kp, des = self.sift.detectAndCompute(gray, None)
            if des is None:
                return [], None
            des = np.array(des, dtype=np.uint8)
            return kp, des
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return [], None

    def _midas_to_metric_depth(self, midas_value):
        """Convert MiDaS depth to metric depth"""
        if midas_value < 1e-3:
            return None
        return DEPTH_SCALE / (midas_value + 1e-6)

    def _backproject_to_world(self, u, v, metric_depth, pose):
        """Backproject 2D point to 3D world coordinates"""
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

    def _extract_depth_pattern(self, depth_map, keypoints):
        """Extract depth pattern for loop closure"""
        if depth_map is None or len(keypoints) < 10:
            return None
        depths = []
        for kp in keypoints[:20]:
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)
                if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                    depths.append(metric_depth)
        if len(depths) < 5:
            return None
        depths = np.array(depths)
        pattern = depths / (np.max(depths) + 1e-6)
        return pattern

    def _detect_motion_state(self, gray):
        """Detect forward motion and rotation"""
        if self.prev_gray is None or len(self.prev_keypoint_positions) == 0:
            return False, False
        prev_pts = np.array(self.prev_keypoint_positions, dtype=np.float32).reshape(-1, 1, 2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if status is None or np.sum(status) < 5:
            return False, False
        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]
        displacements_h = good_next[:, 0, 0] - good_prev[:, 0, 0]
        displacements_v = good_next[:, 0, 1] - good_prev[:, 0, 1]
        median_h = np.median(displacements_h)
        median_v = np.median(displacements_v)
        has_rotation = abs(median_h) > 0.5
        has_forward_motion = median_v > 0.05
        return has_forward_motion, has_rotation

    def _should_create_keyframe(self, coverage):
        """Tuned keyframe creation from tuned.py"""
        self.frames_since_last_keyframe += 1
        if self.frames_since_last_keyframe < self.keyframe_cooldown:
            return False
        reasons = []
        if self.successfully_tracked_count < self.tracked_feature_threshold:
            reasons.append(f"Tracked {self.successfully_tracked_count} < {self.tracked_feature_threshold}")
        dist_since_kf = math.sqrt(
            (self.pose[0] - self.pose_at_last_keyframe[0])**2 +
            (self.pose[1] - self.pose_at_last_keyframe[1])**2
        )
        if dist_since_kf > self.distance_threshold:
            reasons.append(f"Distance {dist_since_kf:.3f}m > {self.distance_threshold:.3f}m")
        rot_since_kf = abs(self.pose[2] - self.pose_at_last_keyframe[2])
        if rot_since_kf > math.pi:
            rot_since_kf = 2*math.pi - rot_since_kf
        if rot_since_kf > self.rotation_threshold:
            reasons.append(f"Rotation {math.degrees(rot_since_kf):.1f}° > {math.degrees(self.rotation_threshold):.1f}°")
        if coverage < self.feature_coverage_threshold:
            reasons.append(f"Coverage {coverage:.1%} < {self.feature_coverage_threshold:.1%}")
        return len(reasons) > 0

    def _spatial_grid_key(self, x, y):
        return (int(x / self.spatial_grid_size), int(y / self.spatial_grid_size))

    def _add_to_spatial_grid(self, x, y, z, descriptor, quality, frame):
        key = self._spatial_grid_key(x, y)
        if key in self.spatial_grid:
            existing = self.spatial_grid[key]
            if quality > existing['quality']:
                self.spatial_grid[key] = {
                    'x': x, 'y': y, 'z': z,
                    'descriptor': descriptor,
                    'quality': quality,
                    'frame': frame
                }
        else:
            if len(self.spatial_grid) < self.max_map_features:
                self.spatial_grid[key] = {
                    'x': x, 'y': y, 'z': z,
                    'descriptor': descriptor,
                    'quality': quality,
                    'frame': frame
                }

    def build_map(self, source):
        """
        Build map from IP rover or video file - uses TUNED map building logic
        """
        print("\n" + "="*60)
        print("🗺️  BUILDING MAP...")
        print("="*60)

        # Open video source
        if source.startswith("http://") or source.startswith("https://"):
            print(f"🌐 Connecting to rover: {source}")
            cap = RoverStreamCapture(source)
        else:
            print(f"📹 Loading video: {source}")
            cap = cv2.VideoCapture(source)

        self.frame_count = 0
        self.first_frame_initialized = False

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                frame = cv2.resize(frame, (640, 480))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Estimate depth using MiDaS
                input_batch = transform(frame).to(device)
                with torch.no_grad():
                    depth_map = midas(input_batch)
                    depth_map = torch.nn.functional.interpolate(
                        depth_map.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth_map = depth_map.cpu().numpy()

                # Extract features
                kp, des = self.feature_extraction(gray, depth_map)

                if not self.first_frame_initialized:
                    if len(kp) < self.min_features_per_frame:
                        print(f"⏳ Frame {self.frame_count}: Need more features ({len(kp)})")
                        self.prev_gray = gray.copy()
                        continue

                    print(f"✅ First frame initialized with {len(kp)} features")
                    features_3d = []
                    keypoints_2d = []

                    for i, point in enumerate(kp):
                        u, v = int(point.pt[0]), int(point.pt[1])
                        midas_depth = depth_map[v, u]
                        metric_depth = self._midas_to_metric_depth(midas_depth)
                        if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                            world_pt = self._backproject_to_world(u, v, metric_depth, self.pose)
                            if world_pt is not None:
                                features_3d.append(world_pt)
                                keypoints_2d.append([u, v])
                                self._add_to_spatial_grid(world_pt[0], world_pt[1], world_pt[2],
                                                         des[i].tobytes() if des is not None else None,
                                                         1.0, self.frame_count)

                    if len(features_3d) > 0:
                        kf = KeyFrame(
                            frame_id=self.keyframe_counter,
                            pose=self.pose.copy(),
                            features_3d=features_3d,
                            descriptors=des,
                            keypoints_2d=keypoints_2d,
                            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                            depth_scale=DEPTH_SCALE, max_depth=self.max_depth_threshold
                        )
                        self.storage_manager.save_keyframe(kf)
                        self.keyframes_info[self.keyframe_counter] = kf
                        self.current_keyframe = kf
                        self.pose_at_last_keyframe = self.pose.copy()
                        self.frames_since_last_keyframe = 0
                        self.keyframe_counter += 1
                        print(f"🔑 Keyframe #{self.keyframe_counter-1} saved at {self.frame_count}")

                    self.first_frame_initialized = True
                    self.prev_gray = gray.copy()
                    self.current_features = kp
                    self.prev_keypoint_positions = [pt.pt for pt in kp]
                    self.trajectory.append(self.pose.copy())
                    continue

                # Track features in subsequent frames
                if des is not None and len(kp) > 0:
                    if self.current_features and len(self.current_features) > 0:
                        prev_des = None
                        if self.prev_keyframe is not None and self.prev_keyframe.descriptors is not None:
                            prev_des = self.prev_keyframe.descriptors

                        if prev_des is not None:
                            try:
                                des = np.array(des, dtype=np.uint8)
                                prev_des = np.array(prev_des, dtype=np.uint8)
                                matches = self.flann.knnMatch(prev_des, des, k=2)

                                good_matches = []
                                for match_pair in matches:
                                    if len(match_pair) == 2:
                                        m, n = match_pair
                                        if m.distance < 0.7 * n.distance:
                                            good_matches.append(m)

                                self.successfully_tracked_count = len(good_matches)

                                # Update pose with tracked features
                                for match in good_matches:
                                    prev_idx = match.trainIdx
                                    curr_idx = match.queryIdx
                                    if curr_idx < len(kp):
                                        u, v = int(kp[curr_idx].pt[0]), int(kp[curr_idx].pt[1])
                                        midas_depth = depth_map[v, u]
                                        metric_depth = self._midas_to_metric_depth(midas_depth)
                                        if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                                            world_pt = self._backproject_to_world(u, v, metric_depth, self.pose)
                                            if world_pt is not None:
                                                self._add_to_spatial_grid(world_pt[0], world_pt[1], world_pt[2],
                                                                         des[curr_idx].tobytes(),
                                                                         1.0, self.frame_count)
                            except Exception as e:
                                print(f"⚠️  Matching error: {e}")

                # Detect motion
                has_motion, has_rotation = self._detect_motion_state(gray)

                if has_motion:
                    self.pose[1] += self.FIXED_FORWARD_DISTANCE
                    self.path_distance += self.FIXED_FORWARD_DISTANCE

                if has_rotation:
                    rotation_amount = 0.05 * self.ROTATION_SENSITIVITY
                    self.pose[2] += rotation_amount

                self.trajectory.append(self.pose.copy())

                # Keyframe decision
                coverage = min(self.successfully_tracked_count / self.max_features_per_frame, 1.0) if self.successfully_tracked_count > 0 else 0.0

                if self._should_create_keyframe(coverage):
                    features_3d = []
                    keypoints_2d = []

                    for i, point in enumerate(kp):
                        u, v = int(point.pt[0]), int(point.pt[1])
                        midas_depth = depth_map[v, u]
                        metric_depth = self._midas_to_metric_depth(midas_depth)
                        if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                            world_pt = self._backproject_to_world(u, v, metric_depth, self.pose)
                            if world_pt is not None:
                                features_3d.append(world_pt)
                                keypoints_2d.append([u, v])

                    if len(features_3d) > 0:
                        kf = KeyFrame(
                            frame_id=self.keyframe_counter,
                            pose=self.pose.copy(),
                            features_3d=features_3d,
                            descriptors=des,
                            keypoints_2d=keypoints_2d,
                            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
                            depth_scale=DEPTH_SCALE, max_depth=self.max_depth_threshold
                        )
                        self.storage_manager.save_keyframe(kf)
                        self.keyframes_info[self.keyframe_counter] = kf
                        self.current_keyframe = kf
                        self.pose_at_last_keyframe = self.pose.copy()
                        self.frames_since_last_keyframe = 0
                        self.keyframe_counter += 1
                        print(f"🔑 Keyframe #{self.keyframe_counter-1} saved at frame {self.frame_count}")

                self.prev_keyframe = self.current_keyframe
                self.prev_gray = gray.copy()
                self.current_features = kp
                self.prev_keypoint_positions = [pt.pt for pt in kp]

                if self.frame_count % 30 == 0:
                    print(f"📊 Frame {self.frame_count} | Keyframes: {self.keyframe_counter} | Tracked: {self.successfully_tracked_count}")

        except KeyboardInterrupt:
            print("\n⏹️  Stopped by user")
        finally:
            cap.release()
            self.visualize_map(show_keyframe_titles=False)

    def visualize_map(self, show_keyframe_titles=False):
        """
        Visualize the map - WITHOUT keyframe titles if show_keyframe_titles=False
        This matches the user's requirement: no kf0, kf1, etc labels on the map
        """
        print("\n🎨 Visualizing map...")

        all_kf_ids = self.storage_manager.get_all_keyframes()
        if not all_kf_ids:
            print("❌ No keyframes to visualize")
            return

        keyframes = []
        for kf_id in all_kf_ids:
            kf = self.storage_manager.load_keyframe(kf_id)
            if kf is not None:
                keyframes.append(kf)

        if not keyframes:
            print("❌ Could not load keyframes")
            return

        # Calculate bounds
        all_x = [kf.pose[0] for kf in keyframes]
        all_y = [kf.pose[1] for kf in keyframes]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        range_x = max_x - min_x + 0.5
        range_y = max_y - min_y + 0.5

        # Create canvas
        size = 1000
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        scale = size / max(range_x, range_y)

        def project(x, y):
            px = int((x - min_x) * scale)
            py = int((y - min_y) * scale)
            return px, py

        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory)-1):
                p1 = project(self.trajectory[i][0], self.trajectory[i][1])
                p2 = project(self.trajectory[i+1][0], self.trajectory[i+1][1])
                if (0 <= p1[0] < size and 0 <= p1[1] < size and
                    0 <= p2[0] < size and 0 <= p2[1] < size):
                    cv2.line(canvas, p1, p2, (200, 200, 200), 1)

        # Draw features from spatial grid
        for key, point_data in self.spatial_grid.items():
            px, py = project(point_data['x'], point_data['y'])
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)

        # Draw keyframes WITHOUT titles (no kf0, kf1 labels)
        for kf in keyframes:
            x, y = kf.pose[0], kf.pose[1]
            sx, sy = project(x, y)
            if 0 <= sx < size and 0 <= sy < size:
                cv2.circle(canvas, (sx, sy), 8, (0, 0, 255), -1)
                arrow_len = 30
                yaw = kf.pose[2]
                ex = int(sx + arrow_len * math.cos(yaw))
                ey = int(sy - arrow_len * math.sin(yaw))
                cv2.arrowedLine(canvas, (sx, sy), (ex, ey), (0, 0, 255), 2, tipLength=0.3)

        # Add text info (WITHOUT keyframe titles/numbers)
        cv2.putText(canvas, f"Total Keyframes: {len(keyframes)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, f"Total Features: {len(self.spatial_grid)}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(canvas, f"Path Distance: {self.path_distance:.2f}m", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("Top-down Map View", canvas)
        cv2.imwrite("map_topdown.png", canvas)
        print("✅ Map visualization saved as 'map_topdown.png'")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def localize_image(self, image_path):
        """
        Localize a single image - uses LOCALIZATION logic from local.py
        """
        print(f"\n📷 Loading image: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot load image: {image_path}")
            return None, None, 0, 0

        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp_test, desc_test = self.feature_extraction(gray, None)

        if desc_test is None or len(desc_test) == 0:
            print("❌ No features extracted from test image")
            return None, None, 0, 0

        print(f"✅ Extracted {len(kp_test)} features from test image")

        all_kf_ids = self.storage_manager.get_all_keyframes()
        if not all_kf_ids:
            print("❌ No keyframes in storage. Build a map first!")
            return None, None, 0, 0
        print(f"🔍 Searching in {len(all_kf_ids)} keyframes...")
        best_match_kf_id = None
        best_inliers = 0
        best_confidence = 0.0
        best_pnp_pose = None

        for kf_id in all_kf_ids:
            keyframe = self.storage_manager.load_keyframe(kf_id)
            if keyframe is None or keyframe.descriptors is None:
                continue

            try:
                kf_descriptors = np.array(keyframe.descriptors, dtype=np.uint8)
                desc_test_uint8 = np.array(desc_test, dtype=np.uint8)

                matches = self.flann.knnMatch(kf_descriptors, desc_test_uint8, k=2)

                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                if len(good_matches) < 4:
                    continue

                # Pose estimation with PnP
                src_pts = np.array([keyframe.features_3d[m.queryIdx] for m in good_matches],
                                 dtype=np.float32)
                dst_pts = np.array([kp_test[m.trainIdx].pt for m in good_matches],
                                 dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    src_pts, dst_pts, self.K,
                    np.array([0, 0, 0, 0]), useExtrinsicGuess=False
                )

                if success:
                    confidence = len(good_matches) / max(len(kp_test), len(keyframe.keypoints_2d))
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match_kf_id = kf_id
                        best_inliers = len(good_matches)
                        best_pnp_pose = (rvec, tvec)

            except Exception as e:
                continue

        if best_match_kf_id is not None:
            print(f"✅ MATCHED with Keyframe #{best_match_kf_id}")
            print(f"   Inliers: {best_inliers}, Confidence: {best_confidence:.2%}")
            return best_match_kf_id, best_pnp_pose, best_inliers, best_confidence
        else:
            print("❌ No match found")
            return None, None, 0, 0

    def display_localization_on_map(self, image_path):
        """
        Load map and display the localized position of the test image
        """
        print("\n🗺️  Loading already-built map...")

        matched_kf_id, pnp_pose, inliers, confidence = self.localize_image(image_path)

        if matched_kf_id is None:
            print("❌ Could not localize image")
            return

        # Recreate map canvas
        all_kf_ids = self.storage_manager.get_all_keyframes()
        keyframes = []
        for kf_id in all_kf_ids:
            kf = self.storage_manager.load_keyframe(kf_id)
            if kf is not None:
                keyframes.append(kf)

        if not keyframes:
            print("❌ No keyframes to visualize")
            return

        all_x = [kf.pose[0] for kf in keyframes]
        all_y = [kf.pose[1] for kf in keyframes]

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        range_x = max_x - min_x + 0.5
        range_y = max_y - min_y + 0.5

        size = 1000
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        scale = size / max(range_x, range_y)

        def project(x, y):
            px = int((x - min_x) * scale)
            py = int((y - min_y) * scale)
            return px, py

        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory)-1):
                p1 = project(self.trajectory[i][0], self.trajectory[i][1])
                p2 = project(self.trajectory[i+1][0], self.trajectory[i+1][1])
                if (0 <= p1[0] < size and 0 <= p1[1] < size and
                    0 <= p2[0] < size and 0 <= p2[1] < size):
                    cv2.line(canvas, p1, p2, (200, 200, 200), 1)

        # Draw features
        for key, point_data in self.spatial_grid.items():
            px, py = project(point_data['x'], point_data['y'])
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1)

        # Draw keyframes
        for kf in keyframes:
            x, y = kf.pose[0], kf.pose[1]
            sx, sy = project(x, y)
            if 0 <= sx < size and 0 <= sy < size:
                cv2.circle(canvas, (sx, sy), 8, (0, 0, 255), -1)
                arrow_len = 30
                yaw = kf.pose[2]
                ex = int(sx + arrow_len * math.cos(yaw))
                ey = int(sy - arrow_len * math.sin(yaw))
                cv2.arrowedLine(canvas, (sx, sy), (ex, ey), (0, 0, 255), 2, tipLength=0.3)

        # Show matched keyframe position with RED crosshair
        if matched_kf_id is not None:
            kf = self.storage_manager.load_keyframe(matched_kf_id)
            if kf is not None:
                loc_x, loc_y, loc_yaw = kf.pose
                loc_sx, loc_sy = project(loc_x, loc_y)
                if 0 <= loc_sx < size and 0 <= loc_sy < size:
                    cv2.circle(canvas, (loc_sx, loc_sy), 30, (0, 0, 255), 3)
                    cv2.line(canvas, (loc_sx-40, loc_sy), (loc_sx+40, loc_sy), (0, 0, 255), 2)
                    cv2.line(canvas, (loc_sx, loc_sy-40), (loc_sx, loc_sy+40), (0, 0, 255), 2)
                    arrow_len = 50
                    arrow_ex = int(loc_sx + arrow_len * math.cos(loc_yaw))
                    arrow_ey = int(loc_sy - arrow_len * math.sin(loc_yaw))
                    cv2.arrowedLine(canvas, (loc_sx, loc_sy), (arrow_ex, arrow_ey),
                                  (0, 0, 255), 2, tipLength=0.3)
                    cv2.putText(canvas, f"🎯 TEST IMAGE MATCHED",
                               (loc_sx - 120, loc_sy - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Info text
        y = 30
        cv2.putText(canvas, "ALREADY-BUILT MAP WITH LOCALIZED POSITION", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 40
        if matched_kf_id is not None:
            cv2.putText(canvas, f"✅ Test Image Matched with KF#{matched_kf_id}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 35
            cv2.putText(canvas, f"Confidence: {confidence:.2%} | Inliers: {inliers}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y += 35
            cv2.putText(canvas, "Position: RED CROSSHAIR on map", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(canvas, "❌ Test Image NOT matched", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Localization Result", canvas)
        cv2.imwrite("localization_result.png", canvas)
        print("✅ Localization result saved as 'localization_result.png'")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main menu - Choose between BUILD or LOCALIZE"""
    print("\n" + "="*60)
    print("🚀 HYBRID SLAM SYSTEM - BUILD MAP or LOCALIZE")
    print("="*60)
    print("\n1️⃣  BUILD MAP (from IP rover or video file)")
    print("2️⃣  LOCALIZE IMAGE (find position in already-built map)")
    print("\n" + "="*60)

    choice = input("Enter choice (1 or 2): ").strip()

    slam = HybridSLAM()

    if choice == "1":
        print("\n📌 MAP BUILDING MODE")
        print("\nChoose source:")
        print("1. IP Rover (live stream)")
        print("2. Video file (local)")
        source_choice = input("Enter choice (1 or 2): ").strip()

        if source_choice == "1":
            rover_url = input("Enter rover IP address/URL (e.g., http://192.168.1.100:8080/stream): ").strip()
            slam.build_map(rover_url)
        elif source_choice == "2":
            video_path = input("Enter video file path: ").strip()
            if not os.path.exists(video_path):
                print(f"❌ File not found: {video_path}")
                return
            slam.build_map(video_path)
        else:
            print("❌ Invalid choice")

    elif choice == "2":
        print("\n📍 LOCALIZATION MODE")
        image_path = input("Enter test image path: ").strip()
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return
        slam.display_localization_on_map(image_path)

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
