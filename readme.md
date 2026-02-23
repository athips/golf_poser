# Full Pipeline Usage Guide

This guide shows how to run the complete pose estimation pipeline with all postprocessing features enabled.

## Quick Start (All Features Enabled)

By default, **all features are enabled**, including:
- ✅ Pattern Recognition (lighting/background adaptation)
- ✅ One-Euro Filter (trajectory smoothing)
- ✅ Club Tracking
- ✅ Joint Interpolation
- ✅ Person Detection (YOLO)

### Basic Command

```bash
python -m 1_golfpose_reproduce.main --video path/to/your/video.mp4 --output-video output.mp4
```

Or if running as a script:

```bash
python main.py --video path/to/your/video.mp4 --output-video output.mp4
```

## Full Command with All Options

### Complete Example

```bash
python -m 1_golfpose_reproduce.main \
    --video "C:\Users\User\Documents\test_data_api\FO_008.mp4" \
    --output-video "output_pose.mp4" \
    --output-npz "results.npz" \
    --output-overlay "overlay_output.mp4" \
    --frame-skip 1 \
    --device cuda \
    --one-euro-min-cutoff 1.0 \
    --one-euro-beta 0.7 \
    --one-euro-d-cutoff 1.0 \
    --overlay-threshold 0.3 \
    --right-handed
```

### Minimal Example (Just Keypoints)

```bash
python -m 1_golfpose_reproduce.main \
    --video input.mp4 \
    --output-npz results.npz
```

## Command-Line Arguments

### Required Arguments
- `--video`: Path to input video file (required)

### Output Options
- `--output-video`: Save video with pose overlay
- `--output-npz`: Save results as compressed NumPy file (.npz)
- `--output-overlay`: Save overlay visualization video
- `--overlay-json`: Path to pose_overlays.json (auto-created if not provided)

### Processing Options
- `--frame-skip N`: Process every Nth frame (default: 1 = all frames)
- `--device`: `cuda` or `cpu` (default: `cuda`)
- `--detect-view-per-frame`: Detect view type for each frame (slower)

### Feature Toggles (Disable Features)
- `--no-person-det`: Disable YOLO person detection
- `--no-club-tracking`: Disable club tracking
- `--no-interp`: Disable joint interpolation
- `--no-one-euro`: Disable One-Euro Filter
- `--no-pattern-recognition`: Disable pattern recognition

### One-Euro Filter Parameters
- `--one-euro-min-cutoff FLOAT`: Minimum cutoff frequency in Hz (default: 1.0)
  - Lower = more smoothing (good for slow movements)
  - Higher = less smoothing (good for fast movements)
- `--one-euro-beta FLOAT`: Speed coefficient (default: 0.7)
  - Higher = more responsive to fast motion
  - Lower = less responsive
- `--one-euro-d-cutoff FLOAT`: Derivative cutoff frequency in Hz (default: 1.0)

### Overlay Options
- `--overlay-threshold FLOAT`: Keypoint confidence threshold for overlay (default: 0.3)
- `--left-handed`: Mark as left-handed golfer
- `--right-handed`: Mark as right-handed golfer (default)

### Model Paths (Optional)
- `--view-model`: Path to view detection model (default: `models/classify_view.onnx`)
- `--pose-fo`: Path to Face-On pose model (default: `models/golfer_fo2.onnx`)
- `--pose-dl`: Path to Down-the-Line pose model (default: `models/golfer_dl.onnx`)
- `--yolo`: Path to YOLO model (default: `models/yolo11n.pt`)

## Processing Pipeline Order

The pipeline processes videos in this order:

1. **Pattern Recognition** (if enabled)
   - Analyzes lighting conditions (bright/normal/dark/variable)
   - Analyzes background complexity
   - Applies adaptive preprocessing (CLAHE, brightness adjustment, etc.)

2. **Pose Estimation**
   - View detection
   - Person detection (YOLO)
   - Pose keypoint prediction

3. **Club Tracking** (if enabled)
   - Detects club in video
   - Stitches club joints
   - Filters hand distance

4. **Joint Interpolation** (if enabled)
   - Fills in missing frames using B-spline interpolation

5. **One-Euro Filter** (if enabled)
   - Smooths keypoint trajectories over time
   - Adapts to motion speed dynamically

## Example Use Cases

### 1. Full Processing with Custom Smoothing

For fast golf swings, use higher beta for more responsiveness:

```bash
python -m 1_golfpose_reproduce.main \
    --video swing.mp4 \
    --output-video output.mp4 \
    --output-npz results.npz \
    --one-euro-beta 1.0 \
    --one-euro-min-cutoff 1.5
```

### 2. Processing with Pattern Recognition Only

Disable One-Euro Filter but keep pattern recognition:

```bash
python -m 1_golfpose_reproduce.main \
    --video swing.mp4 \
    --output-video output.mp4 \
    --no-one-euro
```

### 3. Fast Processing (Skip Some Features)

Process every 2nd frame, disable club tracking:

```bash
python -m 1_golfpose_reproduce.main \
    --video swing.mp4 \
    --output-npz results.npz \
    --frame-skip 2 \
    --no-club-tracking
```

### 4. CPU Processing

If you don't have CUDA:

```bash
python -m 1_golfpose_reproduce.main \
    --video swing.mp4 \
    --output-video output.mp4 \
    --device cpu
```

### 5. Maximum Smoothing (Slow Movements)

For very smooth trajectories:

```bash
python -m 1_golfpose_reproduce.main \
    --video swing.mp4 \
    --output-video output.mp4 \
    --one-euro-min-cutoff 0.5 \
    --one-euro-beta 0.3
```

## Output Files

### `.npz` File Contents
- `keypoints`: Array of shape (num_frames, num_joints, 2) - x, y coordinates
- `confidences`: Array of shape (num_frames, num_joints) - confidence scores
- `view`: Detected view type ("DL" or "FO")
- `fps`: Video frame rate
- `num_frames`: Number of frames processed
- `frame_indices`: Frame indices array

### Loading Results

```python
import numpy as np

# Load results
data = np.load('results.npz')
keypoints = data['keypoints']  # (num_frames, num_joints, 2)
confidences = data['confidences']  # (num_frames, num_joints)
view = data['view']
fps = data['fps']
```

## Troubleshooting

### If One-Euro Filter Fails
The pipeline will continue without it. Check logs for warnings.

### If Pattern Recognition Fails
The pipeline will continue without adaptive preprocessing. Original frames will be used.

### Memory Issues
- Use `--frame-skip 2` or higher to process fewer frames
- Disable club tracking: `--no-club-tracking`
- Use CPU: `--device cpu` (slower but uses less GPU memory)

### Slow Processing
- Increase `--frame-skip` to process fewer frames
- Disable club tracking if not needed: `--no-club-tracking`
- Use GPU: `--device cuda`

## Tips for Best Results

1. **For Golf Swings**: Use default One-Euro parameters (beta=0.7) - they're tuned for golf motion
2. **For Dark Videos**: Pattern recognition will automatically enhance dark frames
3. **For Cluttered Backgrounds**: Pattern recognition will enhance contrast
4. **For Variable Lighting**: Pattern recognition uses adaptive histogram equalization
5. **For Fast Swings**: Increase `--one-euro-beta` to 1.0 or higher
6. **For Slow Swings**: Decrease `--one-euro-min-cutoff` to 0.5

## Python API Usage

You can also use the pipeline programmatically:

```python
from 1_golfpose_reproduce.main import PosePredictor

# Initialize predictor
predictor = PosePredictor(
    view_model_path='models/classify_view.onnx',
    pose_fo_model_path='models/golfer_fo2.onnx',
    pose_dl_model_path='models/golfer_dl.onnx',
    yolo_model_path='models/yolo11n.pt',
    device='cuda'
)

# Process video with all features enabled
results = predictor.process_video(
    video_path='input.mp4',
    output_path='output.mp4',
    frame_skip=1,
    enable_club_tracking=True,
    enable_interpolation=True,
    enable_one_euro_filter=True,
    enable_pattern_recognition=True,
    one_euro_min_cutoff=1.0,
    one_euro_beta=0.7,
    one_euro_d_cutoff=1.0
)

# Access results
keypoints = results['keypoints']  # (num_frames, num_joints, 2)
confidences = results['confidences']  # (num_frames, num_joints)
view = results['view']
fps = results['fps']
```
