# ===============================================================================
# TUNED SLAM: BALANCED KEYFRAME SELECTION (Not too sparse, not too dense)
# ===============================================================================
# Feature Detector: SIFT (Scale-Invariant Feature Transform)
# Solution: Increase thresholds to reduce keyframe frequency
# - Distance threshold: 0.1m → 0.3m (create KF every 30cm, not 10cm)
# - Rotation threshold: 15° → 30° (create KF every 30°, not 15°)
# - Coverage threshold: 30% → 60% (need 60% feature coverage to trigger)
# - Tracked threshold: 10 → 20 (old features need to drop to 20)
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
from concurrent.futures import ThreadPoolExecutor
import threading

# =============== Google Colab Detection & Display Setup ===============
try:
    import google.colab
    IN_COLAB = True
    from google.colab.patches import cv2_imshow
    from IPython import display
    from PIL import Image
    print("🌐 Running in Google Colab - Using IPython display")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - Using cv2.imshow")

# =============== Configuration ===============
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

# =============== Lightweight Keyframe Class ===============
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
            print(f"❌ Error loading keyframe: {e}")
            return None

# =============== TUNED SLAM - REDUCED KEYFRAME FREQUENCY ===============
class TunedFeatureTrackingSLAM:
    """SLAM with tuned thresholds to reduce keyframe frequency"""
    
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                 lookahead_height_ratio=0.9,
                 min_features_per_frame=30, max_features_per_frame=150,
                 rotation_sensitivity=3.0, feature_lifetime_frames=50,
                 max_depth_threshold=3.0, spatial_grid_size=0.10,
                 tracked_feature_threshold=20,    # INCREASED from 10
                 distance_threshold=0.3,           # INCREASED from 0.1m
                 rotation_threshold=30.0,          # INCREASED from 15°
                 feature_coverage_threshold=0.6,   # INCREASED from 0.3
                 keyframe_cooldown=30,             # NEW: Minimum frames between KFs
                 local_map_size=10):
        
        # Camera params
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        
        # Pose tracking
        self.pose = np.zeros(3, dtype=float)
        self.pose_at_last_keyframe = np.zeros(3, dtype=float)
        self.keyframe_trajectory = []  # Stores ONLY keyframe poses for trajectory line
        self.path_distance = 0.0
        self.meter_markers = []
        
        # Motion
        self.motion_mode = "REST"
        self.FIXED_FORWARD_DISTANCE = 0.008
        self.BASE_ROTATION_SENSITIVITY = rotation_sensitivity  # Base sensitivity
        self.ROTATION_SENSITIVITY = rotation_sensitivity  # Will be adjusted dynamically
        self.BASE_FPS = 15.0  # Reference FPS for rotation sensitivity calibration
        self.rest_frames_count = 0
        self.rest_frames_threshold = 10
        
        # Features (minimal RAM usage - only for current processing)
        self.min_features_per_frame = min_features_per_frame
        self.max_features_per_frame = max_features_per_frame
        self.current_feature_target = max_features_per_frame
        self.spatial_grid_size = spatial_grid_size
        self.feature_lifetime_frames = feature_lifetime_frames
        self.visible_map_points = []
        self.max_depth_threshold = max_depth_threshold

        # Multi-threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # ============ TUNED KEYFRAME THRESHOLDS ============
        self.tracked_feature_threshold = tracked_feature_threshold      # ✅ 20
        self.distance_threshold = distance_threshold                    # ✅ 0.3m
        self.rotation_threshold = math.radians(rotation_threshold)      # ✅ 30°
        self.feature_coverage_threshold = feature_coverage_threshold    # ✅ 60%
        self.keyframe_cooldown = keyframe_cooldown                      # ✅ 30 frames min between KFs
        self.frames_since_last_keyframe = 0                             # Track cooldown
        
        # Feature tracking
        self.successfully_tracked_count = 0
        
        # Loop closure - every 5th keyframe
        self.loop_closure_enabled = True
        self.loop_closure_check_interval = 5  # Check every 5th keyframe
        self.loop_closure_min_inliers = 5  # Lowered for better SIFT matching
        self.loop_closure_matches = []
        self.last_loop_closure_kf = -1
        
        # Keyframe management
        self.storage_manager = KeyframeStorageManager()
        self.keyframes_info = {}
        self.current_keyframe = None
        self.prev_keyframe = None
        self.keyframe_counter = 0
        self.first_frame_initialized = False
        self.local_map_size = local_map_size
        
        # Feature detection - SIFT
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.sift = cv2.SIFT_create(
            nfeatures=300,
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        self.total_distance_traveled = 0.0
        self.total_distance_at_last_keyframe = 0.0
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Depth processing
        self.lookahead_height_ratio = lookahead_height_ratio
        self.lookahead_pixel = None
        self.lookahead_depth_curr = None
        self.lookahead_world_pos = None
        
        # Previous frame
        self.prev_gray = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_keypoint_positions = []
        
        # MiDaS frame skipping
        self.frame_count = 0
        self.depth_frame_counter = 0
        self.depth_compute_every_n_frames = 2
        
        # Statistics
        self.current_features = []
        self.visible_features = []
        self.feature_counts = {'total': 0, 'valid_depth': 0, 'displayed': 0, 'stored': 0}
        self.rotation_magnitude = 0.0
        self.translation_magnitude = 0.0
        
        # Keyframe matching
        self.matched_keyframe_id = None
        self.keyframe_match_confidence = 0.0
        self.matched_kf_position = None
        
        # Keyframe trigger reasons
        self.last_keyframe_trigger_reason = ""
        self.keyframe_cooldown_reason = ""
    
    # ======== FUNCTION 1: FEATURE EXTRACTION ========
    def feature_extraction(self, gray, depth_map):
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)

        img_height, img_width = gray.shape
        mask = np.zeros_like(enhanced, dtype=np.uint8)
        h_start = int(img_height * 0.33)
        mask[h_start:, :] = 255

        kp_sift, desc_sift = self.sift.detectAndCompute(enhanced, mask=mask)

        if kp_sift is None or desc_sift is None or len(kp_sift) == 0:
            return [], None

        kp_desc_pairs = list(zip(kp_sift, desc_sift))
        kp_desc_pairs.sort(key=lambda x: -x[0].response)

        n_features = min(len(kp_desc_pairs), self.max_features_per_frame)
        kp_desc_pairs = kp_desc_pairs[:n_features]

        kp_sift = [kp for kp, _ in kp_desc_pairs]
        desc_sift = np.array([desc for _, desc in kp_desc_pairs])

        return kp_sift, desc_sift
    
    # ======== FEATURE TRACKING + COVERAGE CHECK ========
    def track_old_features(self, current_gray):
        if self.prev_gray is None or len(self.prev_keypoint_positions) == 0:
            self.successfully_tracked_count = 0
            return 0, 0.0
        
        prev_pts = np.array(self.prev_keypoint_positions, dtype=np.float32).reshape(-1, 1, 2)
        
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, current_gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if status is None:
            self.successfully_tracked_count = 0
            return 0, 0.0
        
        successfully_tracked = np.sum(status == 1)
        self.successfully_tracked_count = successfully_tracked
        
        if successfully_tracked > 0:
            tracked_pts = next_pts[status.flatten() == 1]
            xs = tracked_pts[:, 0, 0]
            ys = tracked_pts[:, 0, 1]
            
            x_min, x_max = np.min(xs), np.max(xs)
            y_min, y_max = np.min(ys), np.max(ys)
            
            tracked_area = (x_max - x_min) * (y_max - y_min)
            image_area = current_gray.shape[0] * current_gray.shape[1]
            coverage = tracked_area / image_area if image_area > 0 else 0.0
            
            return successfully_tracked, coverage
        else:
            return 0, 0.0
    
    # ======== FUNCTION 2: MAP BUILDING ========
    def map_building(self, kp_sift, desc_sift, gray, depth_map, pose):
        img_height, img_width = gray.shape

        valid_depth_count = 0
        new_features = []
        all_features = []
        features_3d = []
        keypoints_2d = []

        for i, kp in enumerate(kp_sift):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            
            if not (0 <= u < img_width and 0 <= v < img_height):
                continue
            
            if depth_map is not None:
                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)
            else:
                metric_depth = None
            
            if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                valid_depth_count += 1
                
                world_pos = self._backproject_to_world(u, v, metric_depth, pose)
                
                if world_pos is not None:
                    x_world, y_world, z_cam = world_pos
                    descriptor = desc_sift[i]
                    quality = kp.response
                    
                    self._add_to_spatial_grid(x_world, y_world, metric_depth,
                                            descriptor, quality, self.frame_count)
                    
                    self.visible_map_points.append({
                        'x': x_world, 'y': y_world, 'z': metric_depth,
                        'u': u, 'v': v, 'frame': self.frame_count
                    })
                    
                    new_features.append((u, v))
                    all_features.append((u, v))
                    features_3d.append((x_world, y_world, z_cam))
                    keypoints_2d.append((u, v))
        
        self.feature_counts['total'] = len(kp_sift)
        self.feature_counts['valid_depth'] = valid_depth_count
        self.feature_counts['stored'] = len(self.keyframes_info)

        self._update_visible_features()
        self.feature_counts['displayed'] = len(self.visible_map_points)

        self.current_features = all_features
        self.visible_features = [(int(pt['u']), int(pt['v'])) for pt in self.visible_map_points]

        self._adjust_feature_target()
        self.prev_keypoints = kp_sift
        self.prev_descriptors = desc_sift

        self.prev_keypoint_positions = [(kp.pt[0], kp.pt[1]) for kp in kp_sift]
        
        return new_features, features_3d, keypoints_2d
    
    # ======== FUNCTION 3: LOCALIZATION ========
    def localization_loop(self, gray, kp_sift, desc_sift):
        if self.successfully_tracked_count < self.tracked_feature_threshold:
            self._match_with_keyframes(desc_sift)
        else:
            self.matched_keyframe_id = None
            self.keyframe_match_confidence = 0.0
            self.matched_kf_position = None
    
    def _match_with_keyframes(self, desc_sift):
        if desc_sift is None or len(self.keyframes_info) == 0:
            return

        best_match_kf_id = None
        best_match_count = 0
        best_confidence = 0.0
        best_position = None

        for kf_id_str, kf_info in self.keyframes_info.items():
            kf_id = int(kf_id_str)
            keyframe = self.storage_manager.load_keyframe(kf_id)
            if keyframe is None or keyframe.descriptors is None:
                continue

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(desc_sift, keyframe.descriptors, k=2)

            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    # Use same ratio as loop closure (0.8 for SIFT)
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)
            
            match_confidence = len(good_matches) / (len(keyframe.descriptors) + 1e-6)
            
            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_match_kf_id = kf_id
                best_confidence = match_confidence
                best_position = keyframe.pose.copy()
        
        self.matched_keyframe_id = best_match_kf_id
        self.keyframe_match_confidence = best_confidence
        self.matched_kf_position = best_position
        
        if best_match_kf_id is not None:
            print(f"🎯 Matched with KF#{best_match_kf_id} (conf: {best_confidence*100:.1f}%)")
    
    # ======== FUNCTION 4: LOOP CLOSURE WITH POSE CORRECTION ========
    def loop_closure_and_bundle_adjustment(self):
        """Check loop closure after EVERY keyframe"""
        if not self.loop_closure_enabled:
            return

        if len(self.keyframes_info) < 3:
            return

        print(f"\n🔍 Loop closure check for KF#{self.keyframe_counter-1}...")
        
        matched_kf_id, confidence, inliers = self._detect_loop_closure()

        if matched_kf_id is not None and inliers >= self.loop_closure_min_inliers:
            print(f"🔄 LOOP CLOSURE! Matched KF#{matched_kf_id} (inliers={inliers}, conf={confidence*100:.1f}%)")
            self._correct_pose_on_loop_closure(matched_kf_id)
            self.loop_closure_matches.append({
                'current_kf': self.keyframe_counter - 1,
                'matched_kf': matched_kf_id,
                'inliers': inliers
            })
        else:
            print(f"   No loop closure")
    
    def _detect_loop_closure(self):
        """Match current keyframe against ALL previous keyframes - NO spatial filter"""
        if self.current_keyframe is None or self.current_keyframe.descriptors is None:
            return None, 0, 0

        # SIFT descriptors are float32, not uint8!
        current_desc = np.array(self.current_keyframe.descriptors, dtype=np.float32)
        current_kf_id = self.current_keyframe.id

        # Get ALL previous keyframes (only skip very recent - gap of 10)
        all_kf_ids = sorted([int(k) for k in self.keyframes_info.keys()])
        candidate_kf_ids = [kf_id for kf_id in all_kf_ids if kf_id < current_kf_id - 10]

        if len(candidate_kf_ids) == 0:
            return None, 0, 0

        best_match_kf_id = None
        best_inliers = 0
        best_confidence = 0.0

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # Check ALL candidates (no spatial filtering!)
        for kf_id in candidate_kf_ids:
            kf = self.storage_manager.load_keyframe(kf_id)
            if kf is None or kf.descriptors is None:
                continue

            try:
                # SIFT descriptors must be float32
                kf_descriptors = np.array(kf.descriptors, dtype=np.float32)
                matches = bf.knnMatch(current_desc, kf_descriptors, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test - more lenient for SIFT (0.8 instead of 0.75)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.8 * n.distance:  # More lenient ratio
                        good_matches.append(m)

            # Lower threshold for SIFT matching
            if len(good_matches) < 8:
                continue

            # Build 3D-2D correspondences
            object_points = []
            image_points = []

            for match in good_matches:
                kf_feat_idx = match.trainIdx
                curr_feat_idx = match.queryIdx

                if kf_feat_idx < len(kf.features_3d) and curr_feat_idx < len(self.current_keyframe.keypoints_2d):
                    x_world, y_world, z_world = kf.features_3d[kf_feat_idx]
                    u, v = self.current_keyframe.keypoints_2d[curr_feat_idx]
                    object_points.append([x_world, y_world, z_world])
                    image_points.append([u, v])

            if len(object_points) < 8:
                continue

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            # PnP RANSAC - optimized for SIFT
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, None,
                iterationsCount=200,          # More iterations for robustness
                reprojectionError=12.0,       # More lenient (was 8.0)
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE  # Better for SIFT features
            )

            if success and inliers is not None:
                inlier_count = len(inliers)
                confidence = len(good_matches) / (len(kf.descriptors) + 1e-6)
                
                print(f"   KF#{kf_id}: {inlier_count} inliers, conf={confidence*100:.1f}%")
                
                if inlier_count > best_inliers:
                    best_inliers = inlier_count
                    best_match_kf_id = kf_id
                    best_confidence = confidence

        return best_match_kf_id, best_confidence, best_inliers
    
    def _correct_pose_on_loop_closure(self, matched_kf_id):
        """Correct current pose to align with matched keyframe"""
        matched_kf = self.storage_manager.load_keyframe(matched_kf_id)
        if matched_kf is None:
            return

        # Calculate pose correction
        # Simple approach: Update current pose to be close to matched keyframe
        matched_pose = matched_kf.pose

        # Calculate drift between current pose and where we should be
        dx = self.pose[0] - matched_pose[0]
        dy = self.pose[1] - matched_pose[1]

        drift_distance = math.sqrt(dx**2 + dy**2)

        print(f"   Drift detected: {drift_distance:.3f}m")
        print(f"   Correcting pose: ({self.pose[0]:.3f}, {self.pose[1]:.3f}) → ({matched_pose[0]:.3f}, {matched_pose[1]:.3f})")

        # Apply correction - snap to matched keyframe location
        self.pose[0] = matched_pose[0]
        self.pose[1] = matched_pose[1]
        self.pose[2] = matched_pose[2]

        # Update last keyframe reference
        self.pose_at_last_keyframe = self.pose.copy()

        # Update trajectory to show loop closure
        if len(self.keyframe_trajectory) > 0:
            self.keyframe_trajectory[-1] = self.pose.copy()
    
    # ======== Helper Methods ========
    def _spatial_grid_key(self, x, y):
        return (int(x / self.spatial_grid_size), int(y / self.spatial_grid_size))
    
    def _add_to_spatial_grid(self, x, y, z, descriptor, quality, frame):
        # Features are stored in keyframes on disk, not in RAM
        # This method kept for backward compatibility but does minimal work
        pass
    
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
    
    def _adjust_feature_target(self):
        if len(self.current_features) < self.min_features_per_frame:
            self.current_feature_target = min(
                self.current_feature_target + 10,
                self.max_features_per_frame
            )
        elif len(self.current_features) > self.max_features_per_frame:
            self.current_feature_target = max(
                self.current_feature_target - 10,
                self.min_features_per_frame
            )
    
    def _update_visible_features(self):
        self.visible_map_points = [
            pt for pt in self.visible_map_points
            if self.frame_count - pt['frame'] < self.feature_lifetime_frames
        ]
    
    def _detect_motion_state(self, gray):
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
    
    # ======== MULTI-CRITERIA KEYFRAME DECISION ========
    def _should_create_keyframe(self, coverage):
        """Create keyframe every 0.5m traveled - STRICT COOLDOWN"""
        distance_traveled = self.total_distance_traveled - self.total_distance_at_last_keyframe
        
        if distance_traveled >= 0.2:
            self.last_keyframe_trigger_reason = f"Distance {distance_traveled:.3f}m >= 0.2m"
            return True
        else:
            return False
    
    def _create_keyframe(self, gray, depth_map, kp_sift, desc_sift, features_3d, keypoints_2d):
        if self.current_keyframe is not None:
            self.storage_manager.save_keyframe(self.current_keyframe)
            self.keyframes_info[str(self.current_keyframe.id)] = {
                'pose': self.current_keyframe.pose.tolist(),
                'num_features': len(self.current_keyframe.features_3d),
                'timestamp': self.current_keyframe.timestamp
            }

        keyframe = KeyFrame(
            frame_id=self.keyframe_counter,
            pose=self.pose,
            features_3d=features_3d,
            descriptors=desc_sift,
            keypoints_2d=keypoints_2d,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            depth_scale=DEPTH_SCALE,
            max_depth=self.max_depth_threshold
        )
        
        self.prev_keyframe = self.current_keyframe
        self.current_keyframe = keyframe
        self.pose_at_last_keyframe = self.pose.copy()
        self.total_distance_at_last_keyframe = self.total_distance_traveled
        self.keyframe_counter += 1
        self.frames_since_last_keyframe = 0  # Reset cooldown

        # Store ONLY keyframe pose in trajectory
        self.keyframe_trajectory.append(self.pose.copy())

        print(f"✅ KF#{keyframe.id} created | Reason: {self.last_keyframe_trigger_reason}")

        return keyframe
    
    def _initialize_first_frame(self, gray, depth_map, kp_sift, desc_sift, features_3d, keypoints_2d):
        if not self.first_frame_initialized:
            print("\n🔷 INITIALIZING FIRST FRAME AS REFERENCE")
            self._create_keyframe(gray, depth_map, kp_sift, desc_sift, features_3d, keypoints_2d)
            self.first_frame_initialized = True
            print("✅ First keyframe created\n")
    
    def depth_map(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).to(device)
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        return pred.cpu().numpy()

    def adjust_rotation_sensitivity(self, current_fps):
        """Dynamically adjust rotation sensitivity based on FPS

        If FPS increases, rotation sensitivity should DECREASE proportionally
        to maintain consistent rotation behavior (higher FPS = less time between frames)
        """
        if current_fps > 0:
            # Inverse relationship: higher FPS needs LOWER sensitivity
            self.ROTATION_SENSITIVITY = self.BASE_ROTATION_SENSITIVITY * (self.BASE_FPS / current_fps)

    def update(self, gray, frame_bgr):
        img_height, img_width = gray.shape
        self.frame_count += 1
        
        # DEPTH COMPUTATION
        if self.depth_frame_counter % self.depth_compute_every_n_frames == 0:
            depth_map = self.depth_map(frame_bgr)
            self.prev_depth = depth_map
        else:
            depth_map = self.prev_depth
        
        self.depth_frame_counter += 1
        
        # Lookahead point
        u_look = int(self.cx)
        v_look = int(img_height * self.lookahead_height_ratio)
        self.lookahead_pixel = (u_look, v_look)
        
        if depth_map is not None:
            midas_depth = depth_map[v_look, u_look]
            self.lookahead_depth_curr = self._midas_to_metric_depth(midas_depth)
        else:
            self.lookahead_depth_curr = None
        
        # MOTION DETECTION
        if self.prev_gray is not None:
            has_forward, has_rotation = self._detect_motion_state(gray)
            
            if has_rotation:
                self.motion_mode = "ROTATING"
                self.rest_frames_count = 0
            elif has_forward:
                self.motion_mode = "FORWARD"
                self.rest_frames_count = 0
            else:
                self.rest_frames_count += 1
                if self.rest_frames_count >= self.rest_frames_threshold:
                    self.motion_mode = "REST"
            
            # POSE UPDATE
            if self.motion_mode == "ROTATING":
                prev_pt = np.array([[[u_look, v_look]]], dtype=np.float32)
                next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, prev_pt, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                
                if status is not None and status[0][0] == 1:
                    u_next, _ = next_pt[0][0]
                    delta_u = u_next - u_look
                    yaw_raw = delta_u / max(1e-6, self.fx)
                    yaw_delta = yaw_raw * self.ROTATION_SENSITIVITY
                    
                    self.pose[2] += yaw_delta
                    self.rotation_magnitude = abs(math.degrees(yaw_delta))
                    self.translation_magnitude = 0.0
            
            elif self.motion_mode == "FORWARD":
                trans_step = self.FIXED_FORWARD_DISTANCE
                yaw = self.pose[2]
                
                self.pose[0] += trans_step * math.cos(yaw)
                self.pose[1] += trans_step * math.sin(yaw)
                
                old_dist = self.path_distance
                self.path_distance += trans_step
                self.total_distance_traveled += trans_step
                
                if int(self.path_distance) > int(old_dist):
                    self.meter_markers.append({
                        'x': self.pose[0],
                        'y': self.pose[1],
                        'distance': int(self.path_distance)
                    })
                
                self.translation_magnitude = trans_step
                self.rotation_magnitude = 0.0
            
            else:
                self.translation_magnitude = 0.0
                self.rotation_magnitude = 0.0

        # FEATURE TRACKING
        if self.prev_gray is not None:
            self.track_old_features(gray)

        # PARALLEL PROCESSING
        kp_sift, desc_sift = self.feature_extraction(gray, depth_map)
        new_features, features_3d, keypoints_2d = self.map_building(
            kp_sift, desc_sift, gray, depth_map, self.pose
        )
        _, coverage = self.track_old_features(gray) if self.prev_gray is not None else (0, 0.0)

        self.localization_loop(gray, kp_sift, desc_sift)

        # KEYFRAME MANAGEMENT
        if not self.first_frame_initialized:
            self._initialize_first_frame(gray, depth_map, kp_sift, desc_sift, features_3d, keypoints_2d)
        elif self._should_create_keyframe(coverage):
            self._create_keyframe(gray, depth_map, kp_sift, desc_sift, features_3d, keypoints_2d)
            # Check for loop closure after creating keyframe
            self.loop_closure_and_bundle_adjustment()
        
        self.prev_gray = gray
    
    def draw_camera_view(self, frame_bgr, depth_map):
        if depth_map is not None:
            depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            vis = cv2.addWeighted(frame_bgr, 0.4, depth_colored, 0.6, 0)
        else:
            vis = frame_bgr.copy()
        
        for pt in self.visible_map_points:
            u, v = int(pt['u']), int(pt['v'])
            age = self.frame_count - pt['frame']
            alpha = 1.0 - (age / self.feature_lifetime_frames)
            brightness = int(255 * alpha)
            cv2.circle(vis, (u, v), 3, (brightness, brightness, 255), -1)
        
        if self.lookahead_pixel is not None:
            u_look, v_look = self.lookahead_pixel
            cv2.circle(vis, (u_look, v_look), 10, (0, 255, 255), -1)
            
            if self.lookahead_depth_curr is not None:
                cv2.putText(vis, f"{self.lookahead_depth_curr:.2f}m",
                           (u_look + 20, v_look - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        y = 25
        cv2.putText(vis, f"Detected: {self.feature_counts['total']}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(vis, f"Tracked: {self.successfully_tracked_count}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        
        dist_since_kf = math.sqrt(
            (self.pose[0] - self.pose_at_last_keyframe[0])**2 + 
            (self.pose[1] - self.pose_at_last_keyframe[1])**2
        )
        rot_since_kf = abs(self.pose[2] - self.pose_at_last_keyframe[2])
        if rot_since_kf > math.pi:
            rot_since_kf = 2*math.pi - rot_since_kf
        
        cv2.putText(vis, f"Dist: {dist_since_kf:.3f}m | Rot: {math.degrees(rot_since_kf):.1f}°", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        y += 25

        cv2.putText(vis, f"Keyframes: {len(self.keyframes_info) + (1 if self.current_keyframe else 0)}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y += 25
        if self.matched_keyframe_id is not None:
            cv2.putText(vis, f"🎯 Matched KF#{self.matched_keyframe_id}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y += 30
        state_colors = {"ROTATING": (0, 165, 255), "FORWARD": (0, 255, 0), "REST": (128, 128, 128)}
        state_color = state_colors.get(self.motion_mode, (255, 255, 255))
        
        if self.motion_mode == "ROTATING":
            state_text = f"ROTATING ({self.rotation_magnitude:.2f}deg)"
        elif self.motion_mode == "FORWARD":
            state_text = f"FORWARD ({self.translation_magnitude:.4f}m)"
        else:
            state_text = "REST"
        
        cv2.putText(vis, state_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        y += 30
        if self.keyframe_cooldown_reason:
            cv2.putText(vis, f"⏱️ {self.keyframe_cooldown_reason}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 0), 1)
        elif self.last_keyframe_trigger_reason:
            text = self.last_keyframe_trigger_reason[:80]
            cv2.putText(vis, f"✅ KF Trigger: {text}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis
    
    def draw_top_view(self, scale=100, size=900, fps=None):
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        
        center_x, center_y = size // 2, size // 2
        cam_x, cam_y, cam_yaw = self.pose
        cam_screen_x = center_x
        cam_screen_y = int(size * 0.70)
        
        def project(x, y):
            dx = x - cam_x
            dy = y - cam_y
            sx = int(cam_screen_x + dx * scale)
            sy = int(cam_screen_y - dy * scale)
            return sx, sy
        
        # NO FEATURE POINTS DRAWN - Only keyframes shown as blue lines below

        # ORANGE trajectory: Connect ONLY keyframes (sparse waypoints)
        if len(self.keyframe_trajectory) > 1:
            kf_points = []
            for (x, y, _) in self.keyframe_trajectory:
                px, py = project(x, y)
                if 0 <= px < size and 0 <= py < size:
                    kf_points.append((px, py))

            if len(kf_points) > 1:
                for i in range(len(kf_points) - 1):
                    cv2.line(canvas, kf_points[i], kf_points[i+1], (0, 165, 255), 3, cv2.LINE_AA)
        
        # Meter markers
        for marker in self.meter_markers:
            mx, my = project(marker['x'], marker['y'])
            if 0 <= mx < size and 0 <= my < size:
                cv2.circle(canvas, (mx, my), 6, (0, 255, 0), -1)
                cv2.putText(canvas, f"{marker['distance']}m", (mx + 12, my + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # STORED KEYFRAMES with BLUE perpendicular lines
        KEYFRAME_COLOR = (255, 0, 0)
        LINE_LENGTH = 30
        
        for kf_id_str, kf_info in self.keyframes_info.items():
            kf_id = int(kf_id_str)
            kf_pose = np.array(kf_info['pose'])
            kf_x, kf_y, kf_yaw = kf_pose
            kf_sx, kf_sy = project(kf_x, kf_y)
            
            if 0 <= kf_sx < size and 0 <= kf_sy < size:
                cv2.circle(canvas, (kf_sx, kf_sy), 7, KEYFRAME_COLOR, -1)
                cv2.circle(canvas, (kf_sx, kf_sy), 12, KEYFRAME_COLOR, 2)
                
                perp_angle = kf_yaw + math.pi / 2
                line_ex = int(kf_sx + LINE_LENGTH * math.cos(perp_angle))
                line_ey = int(kf_sy - LINE_LENGTH * math.sin(perp_angle))
                
                cv2.line(canvas, (kf_sx, kf_sy), (line_ex, line_ey),
                        KEYFRAME_COLOR, 3, cv2.LINE_AA)
                
                cv2.putText(canvas, f"KF{kf_id}", (kf_sx + 20, kf_sy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYFRAME_COLOR, 1)
        
        # CURRENT keyframe
        if self.current_keyframe is not None:
            cur_x, cur_y, cur_yaw = self.current_keyframe.pose
            cur_sx, cur_sy = project(cur_x, cur_y)
            
            if 0 <= cur_sx < size and 0 <= cur_sy < size:
                cv2.circle(canvas, (cur_sx, cur_sy), 7, (255, 255, 0), -1)
                cv2.circle(canvas, (cur_sx, cur_sy), 12, (255, 255, 0), 2)
                
                perp_angle = cur_yaw + math.pi / 2
                line_ex = int(cur_sx + LINE_LENGTH * math.cos(perp_angle))
                line_ey = int(cur_sy - LINE_LENGTH * math.sin(perp_angle))
                
                cv2.line(canvas, (cur_sx, cur_sy), (line_ex, line_ey),
                        (255, 255, 0), 3, cv2.LINE_AA)
                
                cv2.putText(canvas, f"KF{self.current_keyframe.id}*", 
                           (cur_sx + 20, cur_sy - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Current camera
        arrow_len = 40
        arrow_ex = int(cam_screen_x + arrow_len * math.cos(cam_yaw))
        arrow_ey = int(cam_screen_y - arrow_len * math.sin(cam_yaw))
        
        cv2.circle(canvas, (cam_screen_x, cam_screen_y), 8, (0, 255, 0), -1)
        
        arrow_colors = {"ROTATING": (0, 165, 255), "FORWARD": (0, 255, 0), "REST": (128, 128, 128)}
        arrow_color = arrow_colors.get(self.motion_mode, (255, 255, 255))
        
        cv2.arrowedLine(canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                       arrow_color, 3, tipLength=0.3)
        
        perp_angle = cam_yaw + math.pi / 2
        line_ex_curr = int(cam_screen_x + LINE_LENGTH * math.cos(perp_angle))
        line_ey_curr = int(cam_screen_y - LINE_LENGTH * math.sin(perp_angle))
        cv2.line(canvas, (cam_screen_x, cam_screen_y), (line_ex_curr, line_ey_curr),
                (0, 255, 0), 2, cv2.LINE_AA)
        
        # Info
        y = 30
        if fps is not None:
            cv2.putText(canvas, f"FPS: {fps:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            y += 30
        
        cv2.putText(canvas, f"Distance: {self.path_distance:.2f}m", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y += 30
        cv2.putText(canvas, f"Loop Closures: {len(self.loop_closure_matches)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y += 30
        total_kfs = len(self.keyframes_info) + (1 if self.current_keyframe else 0)
        cv2.putText(canvas, f"Keyframes: {total_kfs}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        y += 30
        cv2.putText(canvas, f"Cooldown: {self.frames_since_last_keyframe}/{self.keyframe_cooldown}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 0), 2)
        
        # Legend
        y = size - 220
        cv2.putText(canvas, "LEGEND:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y += 25
        cv2.circle(canvas, (30, y-5), 3, (0, 165, 255), -1)
        cv2.putText(canvas, "ORANGE = Path", (50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        
        y += 25
        cv2.circle(canvas, (30, y-5), 5, (255, 0, 0), -1)
        cv2.putText(canvas, "BLUE = Stored Keyframes", (50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        y += 25
        cv2.circle(canvas, (30, y-5), 5, (255, 255, 0), -1)
        cv2.putText(canvas, "CYAN = Current KF (memory)*", (50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        y += 25
        cv2.circle(canvas, (30, y-5), 4, (0, 255, 0), -1)
        cv2.putText(canvas, "GREEN = Current Camera", (50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return canvas
    
    def run_slam(self, source=0, target_w=640, target_h=480,
                 use_rover_stream=False, rover_url=None, max_frames=None):
        if use_rover_stream and rover_url:
            cap = RoverStreamCapture(rover_url)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("❌ Cannot open video source")
            return

        print("\n" + "="*110)
        print("🎯 TUNED SLAM - BALANCED KEYFRAME SELECTION (SIFT)")
        print("="*110)
        print(f"✅ Environment: {'Google Colab (GPU)' if IN_COLAB else 'Local'}")
        print(f"✅ Keyframe triggers (ANY met + cooldown):")
        print(f"   1. Tracked features < {self.tracked_feature_threshold}")
        print(f"   2. Distance traveled > {self.distance_threshold:.3f}m (30cm)")
        print(f"   3. Rotation > {math.degrees(self.rotation_threshold):.1f}° (30°)")
        print(f"   4. Feature coverage < {self.feature_coverage_threshold*100:.0f}%")
        print(f"   + Minimum {self.keyframe_cooldown} frames between keyframes")
        print(f"✅ Storage: Only current KF in RAM, rest on disk")
        print(f"✅ Depth: Every 2nd frame")
        print(f"✅ Loop Closure: Every {self.loop_closure_check_interval}th keyframe (multi-threaded)")
        print(f"✅ Multi-threading: 4 CPU cores")
        print(f"✅ Dynamic Rotation Sensitivity: Base {self.BASE_ROTATION_SENSITIVITY} @ {self.BASE_FPS}fps")
        if IN_COLAB:
            print(f"\n📱 Colab Mode: Live map display every 30 frames")
            print(f"   Set max_frames parameter to limit processing (None = process entire video)")
        else:
            print("\n⌨️ [ESC] Quit | [S] Save map | [P] Print stats | [+/-] Adjust cooldown")
        print("="*110 + "\n")

        fps = 0.0
        prev_time = time.time()
        frame_counter = 0

        # Colab display setup
        if IN_COLAB:
            display_handle = display.display(display_id=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if max_frames and frame_counter > max_frames:
                print(f"\n⏹️ Reached max_frames limit: {max_frames}")
                break

            frame = cv2.resize(frame, (target_w, target_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.update(gray, frame)

            now = time.time()
            dt = now - prev_time
            prev_time = now

            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else 1.0 / dt
                # Adjust rotation sensitivity dynamically based on current FPS
                self.adjust_rotation_sensitivity(fps)

            # Display handling
            if IN_COLAB:
                # Update display every 30 frames in Colab (to avoid overwhelming the notebook)
                if frame_counter % 30 == 0:
                    top_map = self.draw_top_view(scale=100, size=900, fps=fps)
                    top_map_rgb = cv2.cvtColor(top_map, cv2.COLOR_BGR2RGB)
                    display_handle.update(Image.fromarray(top_map_rgb))

                    # Print stats
                    print(f"Frame: {self.frame_count} | FPS: {fps:.1f} | "
                          f"Rot.Sens: {self.ROTATION_SENSITIVITY:.3f} | "
                          f"Dist: {self.path_distance:.2f}m | "
                          f"KFs: {len(self.keyframes_info) + (1 if self.current_keyframe else 0)}")
            else:
                # Local display - show both camera view and top view
                vis = self.draw_camera_view(frame, self.prev_depth)
                cv2.imshow("Camera View - Tuned Keyframes", vis)

                top_map = self.draw_top_view(scale=100, size=900, fps=fps)
                cv2.imshow("Top View - Balanced Keyframe Density", top_map)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == ord('s') or key == ord('S'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    map_img = self.draw_top_view(scale=100, size=900, fps=None)
                    map_filename = f"slam_tuned_{timestamp}.png"
                    cv2.imwrite(map_filename, map_img)
                    print(f"✅ Saved: {map_filename}")
                elif key == ord('p') or key == ord('P'):
                    print(f"\n📊 STATISTICS:")
                    print(f"   Frame: {self.frame_count}")
                    print(f"   Distance: {self.path_distance:.3f}m")
                    print(f"   FPS: {fps:.1f}")
                    print(f"   Rotation Sensitivity: {self.ROTATION_SENSITIVITY:.3f}")
                    print(f"   Keyframes (stored): {len(self.keyframes_info)}")
                    print(f"   Cooldown setting: {self.keyframe_cooldown} frames\n")
                elif key == ord('+') or key == ord('='):
                    self.keyframe_cooldown += 10
                    print(f"⬆️ Cooldown increased to {self.keyframe_cooldown} frames (fewer keyframes)")
                elif key == ord('-') or key == ord('_'):
                    self.keyframe_cooldown = max(5, self.keyframe_cooldown - 10)
                    print(f"⬇️ Cooldown decreased to {self.keyframe_cooldown} frames (more keyframes)")

        cap.release()
        if not IN_COLAB:
            cv2.destroyAllWindows()

        # Shutdown thread pool
        self.executor.shutdown(wait=True)

        print("\n💾 Finalizing...")
        if self.current_keyframe is not None:
            self.storage_manager.save_keyframe(self.current_keyframe)
            self.keyframes_info[str(self.current_keyframe.id)] = {
                'pose': self.current_keyframe.pose.tolist(),
                'num_features': len(self.current_keyframe.features_3d),
                'timestamp': self.current_keyframe.timestamp
            }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_map = self.draw_top_view(scale=100, size=900, fps=None)
        cv2.imwrite(f"slam_final_{timestamp}.png", final_map)
        
        print(f"✅ Total keyframes: {len(self.keyframes_info)}")
        print(f"✅ Loop closures detected: {len(self.loop_closure_matches)}")
        print(f"✅ Total distance: {self.path_distance:.2f}m")

        if len(self.loop_closure_matches) > 0:
            print("\n🔄 Loop Closure Summary:")
            for lc in self.loop_closure_matches:
                print(f"   KF#{lc['current_kf']} matched KF#{lc['matched_kf']} ({lc['inliers']} inliers)")

# =============== MAIN ===============
def main_menu():
    print("\n" + "="*110)
    print("🗺️ TUNED KEYFRAME SLAM SYSTEM - Balanced Keyframe Density")
    print("="*110)
    print("1. Build Map (Tuned Keyframe Selection)")
    print("2. Exit")
    print("="*110)
    
    choice = input("Choose [1/2]: ").strip()
    
    if choice == '1':
        build_map_menu()

def build_map_menu():
    print("\n📹 BUILD MAP")
    print("0 - Rover Stream")
    print("1 - Webcam")
    print("2 - Video File")
    
    choice = input("Choose: ").strip()
    
    rot_sens = input("Rotation sensitivity [3.0]: ").strip()
    rotation_sensitivity = float(rot_sens) if rot_sens else 3.0
    
    cooldown = input("Keyframe cooldown frames [30]: ").strip()
    keyframe_cooldown = int(cooldown) if cooldown else 30
    
    slam = TunedFeatureTrackingSLAM(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        min_features_per_frame=30,
        max_features_per_frame=150,
        rotation_sensitivity=rotation_sensitivity,
        feature_lifetime_frames=50,
        max_depth_threshold=3.0,
        spatial_grid_size=0.10,
        tracked_feature_threshold=20,    # TUNED UP from 10
        distance_threshold=0.3,           # TUNED UP from 0.1m
        rotation_threshold=30.0,          # TUNED UP from 15°
        feature_coverage_threshold=0.6,   # TUNED UP from 0.3
        keyframe_cooldown=keyframe_cooldown,
        local_map_size=10
    )
    
    if choice == '0':
        rover_url = input("Rover URL [http://10.47.11.127:8080/video_feed]: ").strip()
        rover_url = rover_url if rover_url else "http://10.47.11.127:8080/video_feed"
        slam.run_slam(source=None, use_rover_stream=True, rover_url=rover_url)
    elif choice == '1':
        slam.run_slam(source=0)
    elif choice == '2':
        path = input("Video path: ").strip()
        if os.path.exists(path):
            slam.run_slam(source=path)
        else:
            print("❌ File not found")

# =============== GOOGLE COLAB CONVENIENCE FUNCTION ===============
def run_colab_slam(video_path,
                   max_frames=None,
                   target_w=1280,
                   target_h=720,
                   rotation_sensitivity=1.05,
                   keyframe_cooldown=30):
    """
    Convenience function for running SLAM in Google Colab

    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process (None = entire video)
        target_w: Target width (default 1280)
        target_h: Target height (default 720)
        rotation_sensitivity: Base rotation sensitivity (default 1.05 for 15fps reference)
        keyframe_cooldown: Minimum frames between keyframes (default 30)

    Example usage in Colab:
        from google.colab import drive
        drive.mount('/content/drive')

        run_colab_slam(
            video_path='/content/drive/MyDrive/your_video.mp4',
            max_frames=1000,  # Process first 1000 frames
            target_w=1280,
            target_h=720,
            rotation_sensitivity=1.05,
            keyframe_cooldown=30
        )
    """
    print(f"🎬 Loading video: {video_path}")
    print(f"📐 Target resolution: {target_w}x{target_h}")
    print(f"🎯 Max frames: {max_frames if max_frames else 'All'}")
    print(f"🔄 Base rotation sensitivity: {rotation_sensitivity} @ 15fps")

    # Calculate camera intrinsics based on target resolution
    cx = target_w / 2.0
    cy = target_h / 2.0
    fx = target_w * 0.8  # Approximate focal length
    fy = target_h * 0.8

    slam = TunedFeatureTrackingSLAM(
        fx=fx, fy=fy, cx=cx, cy=cy,
        min_features_per_frame=30,
        max_features_per_frame=200,  # Increased for higher resolution
        rotation_sensitivity=rotation_sensitivity,
        feature_lifetime_frames=50,
        max_depth_threshold=3.0,
        spatial_grid_size=0.10,
        tracked_feature_threshold=20,
        distance_threshold=0.3,
        rotation_threshold=30.0,
        feature_coverage_threshold=0.6,
        keyframe_cooldown=keyframe_cooldown,
        local_map_size=10
    )

    slam.run_slam(
        source=video_path,
        target_w=target_w,
        target_h=target_h,
        max_frames=max_frames
    )

    # Return SLAM object for analysis
    return slam

if __name__ == "__main__":
    if not IN_COLAB:
        main_menu()
    else:
        print("\n" + "="*80)
        print("🌐 Running in Google Colab")
        print("="*80)
        print("Use the run_colab_slam() function to process your video:")
        print("")
        print("Example:")
        print("  slam = run_colab_slam(")
        print("      video_path='your_video.mp4',")
        print("      max_frames=1000,")
        print("      target_w=1280,")
        print("      target_h=720,")
        print("      rotation_sensitivity=1.05")
        print("  )")
        print("="*80)
