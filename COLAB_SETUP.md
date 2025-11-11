# Google Colab Setup Guide for SLAM with SIFT

## Features
- ✅ SIFT feature detection (Scale-Invariant Feature Transform)
- ✅ GPU acceleration (CUDA support for MiDaS depth estimation)
- ✅ Live map visualization with arrows showing trajectory
- ✅ Dynamic rotation sensitivity adjustment based on FPS
- ✅ Supports 1280x720 @ 30fps video
- ✅ Keyframe-based SLAM with loop closure detection

## Quick Start in Google Colab

### 1. Setup Environment

```python
# Clone or upload the repository
!git clone https://github.com/YOUR_USERNAME/slam-building.git
%cd slam-building

# Install dependencies
!pip install torch torchvision opencv-python-headless numpy Pillow

# Verify GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### 2. Upload Your Video

```python
from google.colab import files
import shutil

# Upload video file
uploaded = files.upload()
video_filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {video_filename}")
```

**OR** mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Use video from your Drive
video_path = '/content/drive/MyDrive/your_video.mp4'
```

### 3. Run SLAM

```python
# Import the SLAM module
import sys
sys.path.append('/content/slam-building')
from tuned import run_colab_slam

# Run SLAM with your video
slam = run_colab_slam(
    video_path='your_video.mp4',  # or video_filename
    max_frames=None,               # None = process entire video, or set limit like 1000
    target_w=1280,                 # Video width
    target_h=720,                  # Video height
    rotation_sensitivity=1.05,     # Base rotation sensitivity at 15fps reference
    keyframe_cooldown=30           # Minimum frames between keyframes
)
```

### 4. View Results

The live map will update every 30 frames showing:
- **Orange trajectory**: Path taken by the camera
- **Blue circles with lines**: Stored keyframes (perpendicular lines show orientation)
- **Cyan circle**: Current keyframe in memory
- **Green circle with arrow**: Current camera position and direction

Stats printed every 30 frames:
- Frame number
- Current FPS
- Adjusted rotation sensitivity
- Total distance traveled
- Number of keyframes created

### 5. Save Final Map

```python
# Save the final map visualization
import cv2
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_map = slam.draw_top_view(scale=100, size=900, fps=None)
map_filename = f"slam_final_{timestamp}.png"
cv2.imwrite(map_filename, final_map)

print(f"✅ Saved: {map_filename}")

# Download the map
from google.colab import files
files.download(map_filename)
```

### 6. Access SLAM Data

```python
# Get trajectory data
print(f"Total keyframes: {len(slam.keyframes_info)}")
print(f"Total distance: {slam.path_distance:.2f}m")
print(f"Loop closures: {len(slam.loop_closure_matches)}")

# Get keyframe positions
for kf_id, kf_info in slam.keyframes_info.items():
    pose = kf_info['pose']
    print(f"KF#{kf_id}: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}")
```

## Understanding Dynamic Rotation Sensitivity

The rotation sensitivity adjusts automatically based on FPS:

```
adjusted_sensitivity = base_sensitivity × (reference_fps / current_fps)
```

- **Reference FPS**: 15fps (midpoint of 12-19fps from your laptop)
- **Base sensitivity**: 1.05 (your calibrated value)
- **At 30fps**: sensitivity will be `1.05 × (15/30) = 0.525` (LOWER for higher FPS)
- **At 12fps**: sensitivity will be `1.05 × (15/12) = 1.313` (HIGHER for lower FPS)

**Why inverse?** Higher FPS = less time between frames = smaller optical flow displacements. We need LOWER sensitivity to compensate and maintain consistent rotation behavior!

## Parameters Explained

### `max_frames`
- `None`: Process entire video
- `1000`: Process only first 1000 frames (good for testing)

### `rotation_sensitivity`
- Default: `1.05` (calibrated for 15fps)
- Lower values = less sensitive to rotation
- Higher values = more sensitive to rotation
- **Adjusts automatically with FPS!**

### `keyframe_cooldown`
- Default: `30` frames
- Minimum number of frames between creating keyframes
- Higher = fewer keyframes (sparser map)
- Lower = more keyframes (denser map)

### `target_w`, `target_h`
- Video resolution to resize to
- Default: `1280x720` (HD)
- Lower resolution = faster processing but less detail
- Higher resolution = slower but more detailed features

## Troubleshooting

### Out of Memory Error
```python
# Reduce resolution
slam = run_colab_slam(
    video_path='video.mp4',
    target_w=640,
    target_h=480,
    max_frames=500
)
```

### Processing Too Slow
```python
# Process every Nth frame (modify in tuned.py)
# Change: self.depth_compute_every_n_frames = 2
# To: self.depth_compute_every_n_frames = 5
```

### Not Enough Keyframes
```python
# Reduce cooldown
slam = run_colab_slam(
    video_path='video.mp4',
    keyframe_cooldown=15  # Create keyframes more frequently
)
```

### Too Many Keyframes (Running Out of Space)
```python
# Increase cooldown
slam = run_colab_slam(
    video_path='video.mp4',
    keyframe_cooldown=50  # Create keyframes less frequently
)
```

## Advanced: Custom SLAM Configuration

```python
from tuned import TunedFeatureTrackingSLAM

# Create SLAM instance with custom parameters
slam = TunedFeatureTrackingSLAM(
    fx=1024.0,                      # Focal length X
    fy=576.0,                       # Focal length Y
    cx=640.0,                       # Principal point X
    cy=360.0,                       # Principal point Y
    min_features_per_frame=30,      # Minimum features to detect
    max_features_per_frame=200,     # Maximum features to detect
    rotation_sensitivity=1.05,      # Base rotation sensitivity
    feature_lifetime_frames=50,     # How long features stay visible
    max_depth_threshold=3.0,        # Max depth in meters
    tracked_feature_threshold=20,   # Min tracked features before relocalization
    distance_threshold=0.3,         # Distance to trigger new keyframe (meters)
    rotation_threshold=30.0,        # Rotation to trigger new keyframe (degrees)
    feature_coverage_threshold=0.6, # Coverage threshold for keyframes
    keyframe_cooldown=30            # Min frames between keyframes
)

# Run with custom configuration
slam.run_slam(
    source='video.mp4',
    target_w=1280,
    target_h=720,
    max_frames=None
)
```

## Performance Tips

1. **Enable GPU**: Always check GPU is available
2. **Limit frames**: Use `max_frames` for testing
3. **Adjust resolution**: Lower resolution for faster processing
4. **Monitor memory**: Watch RAM usage in Colab
5. **Save periodically**: Save intermediate results for long videos

## Example Complete Workflow

```python
# 1. Setup
!git clone https://github.com/YOUR_REPO/slam-building.git
%cd slam-building
!pip install -q torch torchvision opencv-python-headless numpy Pillow

# 2. Upload video
from google.colab import files
uploaded = files.upload()
video_file = list(uploaded.keys())[0]

# 3. Run SLAM
from tuned import run_colab_slam

slam = run_colab_slam(
    video_path=video_file,
    max_frames=1000,  # Test with 1000 frames first
    target_w=1280,
    target_h=720,
    rotation_sensitivity=1.05,
    keyframe_cooldown=30
)

# 4. Save and download results
import cv2
from datetime import datetime
from google.colab import files

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
final_map = slam.draw_top_view(scale=100, size=900)
filename = f"slam_map_{timestamp}.png"
cv2.imwrite(filename, final_map)
files.download(filename)

print(f"\n📊 FINAL STATISTICS:")
print(f"   Frames processed: {slam.frame_count}")
print(f"   Total distance: {slam.path_distance:.2f}m")
print(f"   Keyframes created: {len(slam.keyframes_info)}")
print(f"   Loop closures: {len(slam.loop_closure_matches)}")
```

## What You'll See

### Console Output
```
🌐 Running in Google Colab - Using IPython display
🔧 Initializing MiDaS...
✅ CUDA - GPU: Tesla T4
🎬 Loading video: your_video.mp4
📐 Target resolution: 1280x720
🎯 Max frames: 1000
🔄 Base rotation sensitivity: 1.05 @ 15fps

==================================================
🎯 TUNED SLAM - BALANCED KEYFRAME SELECTION (SIFT)
==================================================
✅ Environment: Google Colab (GPU)
✅ Keyframe triggers (ANY met + cooldown):
   1. Tracked features < 20
   2. Distance traveled > 0.300m (30cm)
   3. Rotation > 30.0° (30°)
   4. Feature coverage < 60%
   + Minimum 30 frames between keyframes
✅ Dynamic Rotation Sensitivity: Base 1.05 @ 15.0fps

Frame: 30 | FPS: 28.4 | Rot.Sens: 1.988 | Dist: 0.05m | KFs: 1
Frame: 60 | FPS: 29.1 | Rot.Sens: 2.037 | Dist: 0.12m | KFs: 1
...
```

### Live Display
You'll see the top-down map updating every 30 frames with:
- Real-time trajectory visualization
- Keyframe positions marked with blue circles
- Current camera position and orientation
- Distance traveled markers

## Notes
- First run will download MiDaS model (~100MB)
- GPU significantly speeds up depth estimation
- Processing time: ~0.5-1 second per frame on GPU
- Expected FPS: 25-35 on Tesla T4 GPU
- Storage: ~1-5MB per keyframe depending on features
