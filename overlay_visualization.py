"""
Visualization script to overlay pose results on video
"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import argparse

from .data_classes import View


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.3,
    joint_radius: int = 5,
    line_thickness: int = 2
) -> np.ndarray:
    """
    Draw skeleton on image.
    
    Args:
        image: Input image
        keypoints: Keypoints array (num_joints, 2)
        confidences: Confidence array (num_joints,)
        threshold: Minimum confidence to draw
        joint_radius: Radius of joint circles
        line_thickness: Thickness of skeleton lines
        
    Returns:
        Image with skeleton drawn
    """
    img = image.copy()
    
    # Skeleton connections for golf pose (19 keypoints)
    # Format: (start_joint, end_joint)
    skeleton_connections = [
        (0, 1), (0, 2),  # head to eyes
        (3, 4), (4, 10), (3, 9), (9, 10),  # shoulders and hips
        (3, 5), (5, 7), (4, 6), (6, 8),  # arms
        (9, 11), (11, 13), (10, 12), (12, 14),  # legs
        (15, 16), (16, 17), (17, 18)  # club
    ]
    
    # Colors for different body parts
    colors = {
        'head': (255, 0, 255),      # Magenta
        'torso': (0, 255, 0),       # Green
        'left_arm': (255, 165, 0),  # Orange
        'right_arm': (0, 165, 255), # Blue
        'left_leg': (255, 0, 0),    # Red
        'right_leg': (0, 0, 255),   # Blue
        'club': (0, 255, 255),      # Cyan
    }
    
    # Draw connections
    for start_idx, end_idx in skeleton_connections:
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            confidences[start_idx] > threshold and confidences[end_idx] > threshold):
            
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            
            # Choose color based on body part
            if start_idx in [0, 1, 2]:
                color = colors['head']
            elif start_idx in [3, 4, 9, 10]:
                color = colors['torso']
            elif start_idx in [3, 5, 7]:  # Left arm (from left shoulder)
                color = colors['left_arm']
            elif start_idx in [4, 6, 8]:  # Right arm (from right shoulder)
                color = colors['right_arm']
            elif start_idx in [9, 11, 13]:  # Left leg (from left hip)
                color = colors['left_leg']
            elif start_idx in [10, 12, 14]:  # Right leg (from right hip)
                color = colors['right_leg']
            elif start_idx in [15, 16, 17, 18]:  # Club
                color = colors['club']
            else:
                color = (255, 255, 255)  # White default
            
            cv2.line(img, pt1, pt2, color, line_thickness)
    
    # Draw keypoints
    for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
        if conf > threshold:
            pt = tuple(kp.astype(int))
            # Color based on confidence
            color_intensity = int(255 * conf)
            color = (0, color_intensity, 255 - color_intensity)
            cv2.circle(img, pt, joint_radius, color, -1)
            # Draw joint number (optional, for debugging)
            # cv2.putText(img, str(i), (pt[0]+5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    
    return img


def draw_club_trajectory(
    image: np.ndarray,
    trajectory: list,
    current_frame_idx: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    max_points: int = None,
    max_jump_distance: float = 100.0  # Maximum distance between consecutive points (reduced for smoother lines)
) -> np.ndarray:
    """
    Draw club head trajectory path up to current frame with outlier filtering.
    
    Args:
        image: Input image
        trajectory: List of (x, y) tuples or None for invalid frames
        current_frame_idx: Current frame index (draw trajectory up to this frame)
        color: Trajectory color (default: green)
        thickness: Line thickness
        max_points: Maximum number of points to draw (None = all up to current frame)
        max_jump_distance: Maximum allowed distance between consecutive points (pixels)
        
    Returns:
        Image with trajectory drawn
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Get valid trajectory points up to current frame with frame indices
    valid_points_with_indices = []
    for i in range(min(current_frame_idx + 1, len(trajectory))):
        if trajectory[i] is not None:
            pt = trajectory[i]
            # Check bounds
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                valid_points_with_indices.append((i, pt))
    
    if len(valid_points_with_indices) < 2:
        return img
    
    # Filter out outliers (points that jump too far from previous point)
    filtered_points = [valid_points_with_indices[0]]  # Always include first point
    
    for i in range(1, len(valid_points_with_indices)):
        prev_idx, prev_pt = valid_points_with_indices[i-1]
        curr_idx, curr_pt = valid_points_with_indices[i]
        
        # Calculate distance between consecutive points
        distance = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)
        
        # Check if jump is reasonable (considering frame gap)
        frame_gap = curr_idx - prev_idx
        # Allow larger jumps if frames are skipped, but still limit
        max_allowed_distance = max_jump_distance * (1 + frame_gap * 0.1)
        
        if distance <= max_allowed_distance:
            filtered_points.append((curr_idx, curr_pt))
        # If jump is too large, skip this point (don't add it)
    
    if len(filtered_points) < 2:
        return img
    
    # Limit number of points if specified
    if max_points is not None and len(filtered_points) > max_points:
        # Take evenly spaced points
        step = len(filtered_points) // max_points
        filtered_points = filtered_points[::max(1, step)]
    
    # Draw trajectory as smooth curves using polylines
    if len(filtered_points) >= 2:
        # Extract just the points (without indices) and ensure they're integers
        points_list = []
        for _, pt in filtered_points:
            if pt is not None:
                # Ensure integer coordinates
                points_list.append((int(round(pt[0])), int(round(pt[1]))))
        
        if len(points_list) >= 2:
            points_array = np.array(points_list, dtype=np.int32)
            
            # Draw as a polyline for smoother appearance
            # Use cv2.polylines with isClosed=False for open curve
            cv2.polylines(img, [points_array], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
            
            # Also draw individual segments with anti-aliasing for extra smoothness
            for i in range(len(points_list) - 1):
                pt1 = points_list[i]
                pt2 = points_list[i + 1]
                # Use LINE_AA for anti-aliased smooth lines
                cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    
    # Draw current club head position as a larger circle
    if current_frame_idx < len(trajectory) and trajectory[current_frame_idx] is not None:
        current_pos = trajectory[current_frame_idx]
        # Ensure integer coordinates
        current_pos_int = (int(round(current_pos[0])), int(round(current_pos[1])))
        if 0 <= current_pos_int[0] < w and 0 <= current_pos_int[1] < h:
            cv2.circle(img, current_pos_int, 8, color, -1)  # Filled circle
            cv2.circle(img, current_pos_int, 8, (255, 255, 255), 2)  # White outline
    
    return img


def smooth_trajectory(
    trajectory: list,
    window_size: int = 15,
    max_jump: float = 100.0
) -> list:
    """
    Smooth trajectory using multiple aggressive passes: outlier removal, Gaussian filter, and spline.
    
    Args:
        trajectory: List of (x, y) tuples or None for invalid frames
        window_size: Size of filter window (must be odd)
        max_jump: Maximum allowed jump between consecutive points (pixels)
        
    Returns:
        Smoothed trajectory list
    """
    if len(trajectory) < 3:
        return trajectory
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    half_window = window_size // 2
    smoothed = trajectory.copy()
    
    # Step 1: Remove obvious outliers (points that jump too far)
    for i in range(1, len(trajectory) - 1):
        if smoothed[i] is None:
            continue
        
        prev_pt = smoothed[i-1]
        next_pt = smoothed[i+1]
        curr_pt = smoothed[i]
        
        if prev_pt is None or next_pt is None:
            continue
        
        # Check distance to previous and next points
        dist_prev = np.sqrt((curr_pt[0] - prev_pt[0])**2 + (curr_pt[1] - prev_pt[1])**2)
        dist_next = np.sqrt((curr_pt[0] - next_pt[0])**2 + (curr_pt[1] - next_pt[1])**2)
        
        # If both distances are too large, this is likely an outlier
        if dist_prev > max_jump * 1.5 and dist_next > max_jump * 1.5:
            # Interpolate from neighbors (keep as float for now, will convert later)
            smoothed[i] = ((prev_pt[0] + next_pt[0]) / 2, (prev_pt[1] + next_pt[1]) / 2)
    
    # Step 2: Apply Gaussian-weighted moving average (smoother than median)
    final_smoothed = smoothed.copy()
    for i in range(len(final_smoothed)):
        if final_smoothed[i] is None:
            continue
        
        # Collect neighboring valid points with Gaussian weights
        neighbors = []
        weights = []
        for j in range(max(0, i - half_window), min(len(final_smoothed), i + half_window + 1)):
            if final_smoothed[j] is not None:
                # Gaussian weight (stronger smoothing)
                dist = abs(j - i)
                sigma = half_window / 2.0
                weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
                neighbors.append(final_smoothed[j])
                weights.append(weight)
        
        if len(neighbors) >= 3:
            # Weighted average
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()  # Normalize
                
                x_avg = sum(n[0] * w for n, w in zip(neighbors, weights))
                y_avg = sum(n[1] * w for n, w in zip(neighbors, weights))
                
                # Convert to integers for OpenCV compatibility
                final_smoothed[i] = (int(round(x_avg)), int(round(y_avg)))
    
    return final_smoothed


def draw_swing_phases(
    image: np.ndarray,
    frame_idx: int,
    swing_phases: Dict,
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    """
    Draw swing phase markers on image.
    
    Args:
        image: Input image
        frame_idx: Current frame index
        swing_phases: Dictionary with swing phase frame indices
        color: Color for phase markers
        
    Returns:
        Image with phase markers
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Draw vertical line at current frame if it's a phase
    phase_names = {
        'TakeAway': 'Takeaway',
        'TopBackswing': 'Top',
        'Impact': 'Impact'
    }
    
    for phase_key, phase_name in phase_names.items():
        phase_frame = swing_phases.get(phase_key)
        if phase_frame is not None and phase_frame == frame_idx:
            # Draw vertical line
            cv2.line(img, (w//2, 0), (w//2, h), color, 3)
            # Draw text
            cv2.putText(img, phase_name, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return img


def draw_info_text(
    image: np.ndarray,
    frame_idx: int,
    total_frames: int,
    view: View,
    fps: float,
    right_handed: bool
) -> np.ndarray:
    """
    Draw information text on image.
    
    Args:
        image: Input image
        frame_idx: Current frame index
        total_frames: Total number of frames
        view: View type
        fps: Video framerate
        right_handed: Whether right-handed
        
    Returns:
        Image with info text
    """
    img = image.copy()
    h, w = img.shape[:2]
    
    # Background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Text information
    info_lines = [
        f"Frame: {frame_idx}/{total_frames}",
        f"View: {view}",
        f"FPS: {fps:.1f}",
        f"Handed: {'Right' if right_handed else 'Left'}"
    ]
    
    y_offset = 20
    for i, line in enumerate(info_lines):
        cv2.putText(img, line, (10, y_offset + i * 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img


def create_overlay_video(
    video_path: str,
    keypoints_path: str,
    confidences_path: str,
    overlay_json_path: str,
    output_path: str,
    draw_skeleton_overlay: bool = True,
    draw_phases: bool = True,
    draw_info: bool = True,
    keypoint_threshold: float = 0.3
):
    """
    Create video with pose overlays.
    
    Args:
        video_path: Path to input video
        keypoints_path: Path to keypoints .npy file
        confidences_path: Path to confidences .npy file
        overlay_json_path: Path to overlay JSON file
        output_path: Path to save output video
        draw_skeleton: Whether to draw skeleton
        draw_phases: Whether to draw swing phases
        draw_info: Whether to draw info text
    """
    # Load data
    print("Loading data...")
    keypoints = np.load(keypoints_path)  # (num_frames, num_joints, 2)
    confidences = np.load(confidences_path)  # (num_frames, num_joints)
    
    with open(overlay_json_path, 'r') as f:
        overlay_dict = json.load(f)
    
    # Extract metadata
    metadata = overlay_dict.get('Metadata', {})
    view_str = metadata.get('View', 'DL')
    view = View.DL if view_str == 'DL' else View.FO
    right_handed = metadata.get('RightHanded', True)
    fps = metadata.get('Framerate', 30.0)
    
    swing_phases = overlay_dict.get('visualaid', {}).get('SwingPhases', {})
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate and fix coordinate scaling
    print(f"Video dimensions: {width}x{height}")
    print(f"Keypoints shape: {keypoints.shape}")
    
    # Check if coordinates are normalized (0-1 range) or need scaling
    if keypoints.size > 0:
        # Filter out zeros and invalid values
        valid_mask = (keypoints[:, :, 0] > 0) & (keypoints[:, :, 1] > 0)
        valid_keypoints = keypoints[valid_mask]
        
        if len(valid_keypoints) > 0:
            kp_min = np.min(valid_keypoints)
            kp_max = np.max(valid_keypoints)
            kp_mean = np.mean(valid_keypoints)
            print(f"Keypoint coordinate range: [{kp_min:.2f}, {kp_max:.2f}], mean: {kp_mean:.2f}")
            
            # Check first frame sample
            if len(keypoints) > 0:
                sample_kp = keypoints[0]
                valid_sample = sample_kp[(sample_kp[:, 0] > 0) & (sample_kp[:, 1] > 0)]
                if len(valid_sample) > 0:
                    print(f"Sample first frame keypoints (first 3 valid):")
                    for i, kp in enumerate(valid_sample[:3]):
                        print(f"  Joint: ({kp[0]:.2f}, {kp[1]:.2f})")
            
            # If coordinates are in normalized range (0-1) or very small, scale them
            if kp_max <= 1.5:  # Likely normalized (0-1) or very small values
                print("⚠️  WARNING: Keypoints appear to be normalized (0-1 range). Scaling to video dimensions...")
                keypoints[:, :, 0] = keypoints[:, :, 0] * width
                keypoints[:, :, 1] = keypoints[:, :, 1] * height
                new_valid = keypoints[valid_mask]
                print(f"✓ Scaled keypoint range: [{np.min(new_valid):.2f}, {np.max(new_valid):.2f}]")
            elif kp_max < width * 0.1 and kp_max < height * 0.1:  # Coordinates are too small for video size
                print(f"⚠️  WARNING: Keypoints appear to be in wrong scale (max={kp_max:.2f} vs video {width}x{height}).")
                print("   Attempting to scale...")
                # Estimate scale factor based on ratio
                scale_x = width / kp_max if kp_max > 0 else 1.0
                scale_y = height / kp_max if kp_max > 0 else 1.0
                keypoints[:, :, 0] = keypoints[:, :, 0] * scale_x
                keypoints[:, :, 1] = keypoints[:, :, 1] * scale_y
                new_valid = keypoints[valid_mask]
                print(f"✓ Scaled keypoint range: [{np.min(new_valid):.2f}, {np.max(new_valid):.2f}]")
            else:
                print(f"✓ Keypoints appear to be in correct pixel space")
        
        # Clamp coordinates to video bounds (but preserve zeros/invalid)
        valid_x = keypoints[:, :, 0] > 0
        valid_y = keypoints[:, :, 1] > 0
        keypoints[valid_x, 0] = np.clip(keypoints[valid_x, 0], 0, width - 1)
        keypoints[valid_y, 1] = np.clip(keypoints[valid_y, 1], 0, height - 1)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    print(f"Processing {total_frames} frames...")
    
    # Extract club head trajectory (keypoint 18)
    # Keypoints should already be filtered during detection in pose_predictor
    CLUB_HEAD_IDX = 18
    club_head_trajectory = []
    club_head_confidences = []
    valid_club_points = 0
    
    for i in range(len(keypoints)):
        if CLUB_HEAD_IDX < len(keypoints[i]) and CLUB_HEAD_IDX < len(confidences[i]):
            kp = keypoints[i][CLUB_HEAD_IDX]
            conf = confidences[i][CLUB_HEAD_IDX]
            # Only include valid keypoints (coordinates > 0 means it passed threshold during detection)
            if kp[0] > 0 and kp[1] > 0:
                club_head_trajectory.append((int(kp[0]), int(kp[1])))
                club_head_confidences.append(conf)
                valid_club_points += 1
            else:
                club_head_trajectory.append(None)
                club_head_confidences.append(0.0)
        else:
            club_head_trajectory.append(None)
            club_head_confidences.append(0.0)
    
    # Apply aggressive smoothing to trajectory
    if valid_club_points > 3:
        # First pass: remove outliers and apply Gaussian smoothing
        smoothed_trajectory = smooth_trajectory(club_head_trajectory, window_size=15, max_jump=100.0)
        
        # Second pass: apply spline smoothing if scipy is available
        try:
            from scipy.interpolate import UnivariateSpline, interp1d
            # Extract valid points for spline
            valid_indices = [i for i, pt in enumerate(smoothed_trajectory) if pt is not None]
            if len(valid_indices) >= 4:  # Need at least 4 points for spline
                valid_points = np.array([smoothed_trajectory[i] for i in valid_indices])
                x_coords = valid_points[:, 0]
                y_coords = valid_points[:, 1]
                
                # Use cubic spline interpolation for very smooth curves
                # Higher s = smoother but less accurate to original points
                # Use a much higher smoothing factor for very smooth results
                smoothing_factor = len(valid_indices) * 50  # Increased from 10 to 50
                spline_x = UnivariateSpline(valid_indices, x_coords, s=smoothing_factor, k=3)
                spline_y = UnivariateSpline(valid_indices, y_coords, s=smoothing_factor, k=3)
                
                # Evaluate spline at all indices (including gaps)
                all_indices = np.arange(len(smoothed_trajectory))
                x_smooth = spline_x(all_indices)
                y_smooth = spline_y(all_indices)
                
                # Update trajectory with spline-smoothed values (fill gaps too)
                # Convert to integers for OpenCV compatibility
                for i in range(len(smoothed_trajectory)):
                    smoothed_trajectory[i] = (int(round(x_smooth[i])), int(round(y_smooth[i])))
                
                print(f"Applied aggressive spline smoothing to trajectory: {len(valid_indices)} points")
        except ImportError:
            print("scipy not available, using basic smoothing only")
        
        club_head_trajectory = smoothed_trajectory
        print(f"Applied trajectory smoothing: {valid_club_points} points")
    
    print(f"Club head trajectory: {valid_club_points}/{len(keypoints)} valid points")
    
    frame_idx = 0
    keypoint_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get corresponding keypoints (handle frame skipping)
        if keypoint_idx < len(keypoints):
            kp = keypoints[keypoint_idx]
            conf = confidences[keypoint_idx]
        else:
            # Use last frame's keypoints if we've run out
            kp = keypoints[-1]
            conf = confidences[-1]
        
        # Draw club head trajectory (up to current frame)
        if draw_skeleton_overlay and keypoint_idx < len(club_head_trajectory):
            # Use frame_idx for trajectory drawing to match video frames
            # Reduced max_jump_distance since we've already smoothed the trajectory
            frame = draw_club_trajectory(frame, club_head_trajectory, frame_idx, max_jump_distance=50.0)
        
        # Draw overlays
        if draw_skeleton_overlay:
            frame = draw_skeleton(frame, kp, conf, threshold=keypoint_threshold)
        
        if draw_phases:
            frame = draw_swing_phases(frame, frame_idx, swing_phases)
        
        if draw_info:
            frame = draw_info_text(frame, frame_idx, total_frames, view, fps, right_handed)
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
        
        # Advance keypoint index (assuming 1:1 mapping for now)
        if keypoint_idx < len(keypoints) - 1:
            keypoint_idx += 1
    
    cap.release()
    out.release()
    print(f"Saved overlay video to: {output_path}")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Overlay pose results on video')
    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--keypoints', default='output/keypoints.npy',
                       help='Path to keypoints .npy file')
    parser.add_argument('--confidences', default='output/confidences.npy',
                       help='Path to confidences .npy file')
    parser.add_argument('--overlay-json', default='output/pose_overlays.json',
                       help='Path to overlay JSON file')
    parser.add_argument('--output', default='output/overlay_video.mp4',
                       help='Path to save output video')
    parser.add_argument('--no-skeleton', action='store_true',
                       help='Disable skeleton drawing')
    parser.add_argument('--no-phases', action='store_true',
                       help='Disable swing phase markers')
    parser.add_argument('--no-info', action='store_true',
                       help='Disable info text')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Keypoint confidence threshold (default: 0.3)')
    
    args = parser.parse_args()
    
    create_overlay_video(
        video_path=args.video_path,
        keypoints_path=args.keypoints,
        confidences_path=args.confidences,
        overlay_json_path=args.overlay_json,
        output_path=args.output,
        draw_skeleton_overlay=not args.no_skeleton,
        draw_phases=not args.no_phases,
        draw_info=not args.no_info,
        keypoint_threshold=args.threshold
    )


if __name__ == "__main__":
    main()

