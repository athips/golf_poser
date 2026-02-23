"""
Utility functions for pose estimation
"""
import numpy as np
from typing import Tuple, Optional
from .data_classes import View


def expand_bounding_box(
    bbox: np.ndarray,
    view: View,
    image_height: int,
    image_width: int
) -> np.ndarray:
    """
    Expand bounding box to include full swing motion.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        view: View type (FO or DL)
        image_height: Image height
        image_width: Image width
        
    Returns:
        Expanded bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    
    if view == View.FO:
        # Face On: expand more horizontally
        expand_x = box_width * 2.0
        expand_y_top = box_height * 0.3
        expand_y_bottom = box_height * 0.1
    else:  # DL
        # Down the Line: expand more vertically
        expand_x = box_width * 1.6
        expand_y_top = box_height * 0.5
        expand_y_bottom = box_height * 0.1
    
    # Expand
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y_top)
    new_x2 = min(image_width, x2 + expand_x)
    new_y2 = min(image_height, y2 + expand_y_bottom)
    
    return np.array([new_x1, new_y1, new_x2, new_y2])


def rescale_pose_coordinates(
    keypoints: np.ndarray,
    crop_offset: np.ndarray,
    scale: float,
    pad: Tuple[int, int]
) -> np.ndarray:
    """
    Rescale pose coordinates from model space to original image space.
    
    Args:
        keypoints: Keypoints in model space (num_joints, 2)
        crop_offset: Crop offset [x, y]
        scale: Scale factor used in preprocessing
        pad: Padding (pad_w, pad_h)
        
    Returns:
        Rescaled keypoints in original image space
    """
    pad_w, pad_h = pad
    
    # Remove padding
    keypoints_rescaled = keypoints.copy()
    keypoints_rescaled[:, 0] = (keypoints_rescaled[:, 0] - pad_w) / scale
    keypoints_rescaled[:, 1] = (keypoints_rescaled[:, 1] - pad_h) / scale
    
    # Add crop offset
    keypoints_rescaled[:, 0] += crop_offset[0]
    keypoints_rescaled[:, 1] += crop_offset[1]
    
    return keypoints_rescaled


def filter_joints(
    joints: np.ndarray,
    framerate: float,
    cutoff_hz: float = 10.0
) -> np.ndarray:
    """
    Apply low-pass filter to joint coordinates.
    
    Args:
        joints: Joint coordinates (num_frames, num_joints, 2) or 1D signal (num_frames,)
        framerate: Video framerate
        cutoff_hz: Cutoff frequency in Hz
        
    Returns:
        Filtered joints or signal
    """
    from scipy.signal import butter, filtfilt
    
    if cutoff_hz >= framerate / 2:
        raise ValueError(f"Cutoff frequency ({cutoff_hz} Hz) must be less than Nyquist frequency ({framerate/2} Hz)")
    
    order = 2
    nyq = 0.5 * framerate
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Handle 1D signal (single array)
    if joints.ndim == 1:
        return filtfilt(b, a, joints)
    
    # Handle 2D array (num_frames, num_features) - treat as single joint with multiple features
    if joints.ndim == 2:
        if joints.shape[1] == 1:
            # Single feature, filter along time axis
            return filtfilt(b, a, joints[:, 0])
        else:
            # Multiple features, filter each
            filtered = np.zeros_like(joints)
            for feat in range(joints.shape[1]):
                filtered[:, feat] = filtfilt(b, a, joints[:, feat])
            return filtered
    
    # Handle 3D array (num_frames, num_joints, 2) - original behavior
    filtered = np.zeros_like(joints)
    for j in range(joints.shape[1]):
        for coord in range(joints.shape[2]):
            filtered[:, j, coord] = filtfilt(b, a, joints[:, j, coord])
    
    return filtered


def interpolate_missing_joints(
    joints: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.3
) -> np.ndarray:
    """
    Interpolate missing joints using linear interpolation.
    
    Args:
        joints: Joint coordinates (num_frames, num_joints, 2)
        confidences: Confidence scores (num_frames, num_joints)
        threshold: Confidence threshold below which joint is considered missing
        
    Returns:
        Interpolated joints
    """
    from scipy.interpolate import interp1d
    
    interpolated = joints.copy()
    num_frames, num_joints, _ = joints.shape
    
    for j in range(num_joints):
        # Find frames with valid detections
        valid_mask = confidences[:, j] > threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) < 2:
            continue  # Not enough data to interpolate
        
        # Interpolate x and y separately
        for coord in range(2):
            valid_values = joints[valid_indices, j, coord]
            
            # Create interpolation function
            interp_func = interp1d(
                valid_indices,
                valid_values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            
            # Interpolate for all frames
            all_indices = np.arange(num_frames)
            interpolated[all_indices, j, coord] = interp_func(all_indices)
    
    return interpolated

