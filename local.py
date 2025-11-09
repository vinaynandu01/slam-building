# ===============================================================================
# SLAM SYSTEM - BUILD or LOCALIZE (FIXED DESCRIPTOR DTYPE)
# ===============================================================================
# FIX: Descriptor dtype mismatch - convert to uint8 numpy array
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
# localize_only.py
# Lightweight localization-only tool.
# Usage: run and follow prompts. Press 'q' in the visualization window to return to prompt,
# press 'Esc' in the window to quit the program entirely.


# ---------- KeyFrame ----------
class KeyFrame:
    def __init__(self, frame_id, pose, features_3d, descriptors,
                 keypoints_2d, fx, fy, cx, cy, depth_scale=DEPTH_SCALE, max_depth=3.0):
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
        descriptors = None
        if data.get('descriptors') is not None:
            descriptors = np.array(data['descriptors'], dtype=np.uint8)
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
            max_depth=3.0
        )
        kf.timestamp = data.get('timestamp', time.time())
        return kf

# ---------- Storage Manager ----------
class KeyframeStorageManager:
    def __init__(self, storage_dir="keyframes_storage"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
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

    def load_keyframe(self, kf_id):
        try:
            if str(kf_id) not in self.index.get('keyframes', {}):
                return None
            filename = self.index['keyframes'][str(kf_id)]['filename']
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return KeyFrame.from_dict(data)
        except Exception:
            return None

# ---------- SLAM (localization-only subset) ----------
class SLAM:
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        self.storage_manager = KeyframeStorageManager()
        self.keyframes_info = {}
        self.spatial_grid = {}
        self.pose = np.zeros(3, dtype=float)   # not used for building but used for visualization
        # ORB for feature extraction on query images
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.orb = cv2.ORB_create(nfeatures=500)
        self.frame_count = 0

    def feature_extraction(self, gray):
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
        kp, desc = self.orb.detectAndCompute(enhanced, None)
        if kp is None or desc is None or len(kp) == 0:
            return [], None
        # keep best by response
        pairs = list(zip(kp, desc))
        pairs.sort(key=lambda x: -x[0].response)
        top = pairs[:min(len(pairs), 500)]
        kp = [p[0] for p in top]
        desc = np.array([p[1] for p in top], dtype=np.uint8)
        return kp, desc

    def localize_image(self, image_path):
        print(f"\n📷 Loading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Cannot load image")
            return None, None, 0, 0
        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp_test, desc_test = self.feature_extraction(gray)
        if desc_test is None or len(desc_test) == 0:
            print("❌ No features extracted from test image")
            return None, None, 0, 0
        print(f"✅ Extracted {len(kp_test)} features from test image")

        all_kf_ids = self.storage_manager.get_all_keyframes()
        if not all_kf_ids:
            print("❌ No keyframes in storage")
            return None, None, 0, 0

        best_match_kf_id = None
        best_inliers = 0
        best_confidence = 0.0
        best_pnp_pose = None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for kf_id in all_kf_ids:
            kf = self.storage_manager.load_keyframe(kf_id)
            if kf is None or kf.descriptors is None:
                continue
            try:
                kf_descriptors = np.array(kf.descriptors, dtype=np.uint8)
                desc_test_uint8 = np.array(desc_test, dtype=np.uint8)
                matches = bf.knnMatch(desc_test_uint8, kf_descriptors, k=2)
            except cv2.error as e:
                print(f"   ⚠️ Error matching KF#{kf_id}: {e}")
                continue

            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 10:
                continue

            # build 3D-2D correspondences
            object_points = []
            image_points = []
            for match in good_matches:
                kf_feat_idx = match.trainIdx
                curr_feat_idx = match.queryIdx
                if kf_feat_idx < len(kf.features_3d):
                    x_world, y_world, z_world = kf.features_3d[kf_feat_idx]
                    object_points.append([x_world, y_world, z_world])
                    if curr_feat_idx < len(kp_test):
                        u, v = kp_test[curr_feat_idx].pt
                        image_points.append([u, v])

            if len(object_points) < 10:
                continue

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, None,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99
            )

            if success and inliers is not None:
                inlier_count = len(inliers)
                match_confidence = len(good_matches) / (len(kf.descriptors) + 1e-6)
                print(f"   KF#{kf_id}: {inlier_count} inliers, conf={match_confidence*100:.1f}%")
                if inlier_count > best_inliers:
                    best_inliers = inlier_count
                    best_match_kf_id = kf_id
                    best_confidence = match_confidence
                    best_pnp_pose = {
                        'rvec': rvec.flatten(),
                        'tvec': tvec.flatten(),
                        'keyframe_pose': kf.pose.copy()
                    }

        if best_match_kf_id is not None:
            print(f"\n✅ BEST MATCH: KF#{best_match_kf_id}  (inliers={best_inliers})")
            return best_match_kf_id, best_pnp_pose, best_confidence, best_inliers
        else:
            print("❌ No good match found")
            return None, None, 0, 0

    def _spatial_grid_key(self, x, y):
        return (int(x / 0.10), int(y / 0.10))

    def draw_top_view_with_localization(self, matched_kf_id, pnp_pose, scale=100, size=900):
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        center_x, center_y = size // 2, size // 2
        cam_screen_x, cam_screen_y = center_x, int(size * 0.70)

        def project(x, y):
            dx = x - self.pose[0]
            dy = y - self.pose[1]
            sx = int(cam_screen_x + dx * scale)
            sy = int(cam_screen_y - dy * scale)
            return sx, sy

        # draw map features
        for feat in self.spatial_grid.values():
            sx, sy = project(feat['x'], feat['y'])
            if 0 <= sx < size and 0 <= sy < size:
                brightness = int(np.clip(200.0 / (feat['z'] + 0.5), 40, 255))
                cv2.circle(canvas, (sx, sy), 2, (brightness, brightness, brightness), -1)

        # draw keyframes as a small line (top-down rectangular representation simplified to a line)
        KEYFRAME_COLOR = (255, 0, 0)
        LINE_LENGTH = 30
        for kf_id_str, kf_info in self.keyframes_info.items():
            kf_id = int(kf_id_str)
            kf_pose = np.array(kf_info['pose'])
            kf_x, kf_y, kf_yaw = kf_pose
            kf_sx, kf_sy = project(kf_x, kf_y)
            if 0 <= kf_sx < size and 0 <= kf_sy < size:
                perp_angle = kf_yaw + math.pi / 2
                line_ex = int(kf_sx + LINE_LENGTH * math.cos(perp_angle))
                line_ey = int(kf_sy - LINE_LENGTH * math.sin(perp_angle))
                cv2.line(canvas, (kf_sx, kf_sy), (line_ex, line_ey), KEYFRAME_COLOR, 3, cv2.LINE_AA)
                if matched_kf_id is not None and kf_id == matched_kf_id:
                    cv2.circle(canvas, (kf_sx, kf_sy), 12, (0, 255, 255), 3)
                    cv2.putText(canvas, f"KF{kf_id}*MATCHED*", (kf_sx - 80, kf_sy - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(canvas, f"KF{kf_id}", (kf_sx + 8, kf_sy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEYFRAME_COLOR, 1)

        # show localized position marker if matched
        if matched_kf_id is not None and pnp_pose is not None:
            kf = self.storage_manager.load_keyframe(matched_kf_id)
            if kf is not None:
                loc_x, loc_y, loc_yaw = kf.pose
                loc_sx, loc_sy = project(loc_x, loc_y)
                if 0 <= loc_sx < size and 0 <= loc_sy < size:
                    cv2.circle(canvas, (loc_sx, loc_sy), 10, (0, 0, 255), 2)
                    arrow_len = 50
                    arrow_ex = int(loc_sx + arrow_len * math.cos(loc_yaw))
                    arrow_ey = int(loc_sy - arrow_len * math.sin(loc_yaw))
                    cv2.arrowedLine(canvas, (loc_sx, loc_sy), (arrow_ex, arrow_ey), (0, 0, 255), 2, tipLength=0.3)
                    cv2.putText(canvas, f"🎯 LOCALIZED (KF#{matched_kf_id})", (loc_sx - 120, loc_sy - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.circle(canvas, (cam_screen_x, cam_screen_y), 8, (0, 255, 0), -1)
        cv2.putText(canvas, "ALREADY-BUILT MAP - LOCALIZATION VIEW", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas

# ---------- Helper to load map/keyframes into SLAM ----------
def load_map_into_slam(slam):
    all_kf_ids = slam.storage_manager.get_all_keyframes()
    if not all_kf_ids:
        print("❌ No keyframes found in storage (keyframes_index.json empty or missing).")
        return False
    for kf_id in all_kf_ids:
        kf = slam.storage_manager.load_keyframe(kf_id)
        if kf:
            slam.keyframes_info[str(kf.id)] = {
                'pose': kf.pose.tolist(),
                'num_features': len(kf.features_3d),
                'timestamp': kf.timestamp
            }
            # add features to spatial grid (simple aggregation)
            for x_world, y_world, z_depth in kf.features_3d:
                key = slam._spatial_grid_key(x_world, y_world)
                if key not in slam.spatial_grid:
                    slam.spatial_grid[key] = {
                        'x': x_world, 'y': y_world, 'z': z_depth,
                        'descriptor': None, 'quality': 1.0, 'frame': 0
                    }
    return True

# ---------- Main interactive loop ----------
def main_loop():
    print("\n🔍 LOCALIZATION-ONLY MODE")
    print("Map/keyframes will be loaded from 'keyframes_storage' directory (keyframes_index.json).")
    slam = SLAM(fx=600.0, fy=600.0, cx=320.0, cy=240.0)

    loaded = load_map_into_slam(slam)
    if not loaded:
        print("Cannot continue without existing keyframes. Exiting.")
        return

    while True:
        user_in = input("\nEnter image path to localize (or 'q' to quit): ").strip()
        if user_in.lower() == 'q':
            print("Goodbye.")
            break

        img_path = user_in
        if not os.path.exists(img_path):
            print("❌ Image not found. Try again.")
            continue

        matched_kf_id, pnp_pose, confidence, inliers = slam.localize_image(img_path)

        canvas = slam.draw_top_view_with_localization(matched_kf_id, pnp_pose, scale=100, size=900)
        win_name = "LOCALIZATION RESULT - Press 'q' to return to prompt, 'Esc' to quit"
        cv2.imshow(win_name, canvas)

        # Wait for user to press 'q' to go back to prompt, 'Esc' to exit, or any other key to continue (prompt)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                # close window and ask for image path again
                cv2.destroyAllWindows()
                break
            elif key == 27:
                # Esc: quit program entirely
                cv2.destroyAllWindows()
                print("Exiting.")
                return
            else:
                # any other key: close window and go back to prompt
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    main_loop()
