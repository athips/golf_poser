"""
Joint interpolation utilities for filling in missing frames.

This module implements B-spline interpolation to interpolate joint positions
for frames that were not processed or have missing detections.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.interpolate import UnivariateSpline
import logging

logger = logging.getLogger(__name__)


class InterpolationError(Exception):
    """Base exception for interpolation errors."""
    pass


class NotEnoughData(InterpolationError):
    """Raised when there are not enough data points for interpolation."""
    def __init__(self, message: str, n_samples: int, required: int):
        self.n_samples = n_samples
        self.required = required
        super().__init__(message)


def interpolate_joint(
    joint: np.ndarray,
    downsampling_vector: np.ndarray,
    joint_conf: Optional[np.ndarray] = None,
    s_val: float = 0.0,
    k_val: int = 3
) -> Tuple[np.ndarray, Optional[Exception]]:
    """
    Interpolates the position of a single joint of a skeleton.
    
    Args:
        joint: A 2D numpy array with the x and y coordinates of a single joint.
               Shape (N, 2) where N > k_val
        downsampling_vector: A 1D numpy array indicating which frames of the original
                            video were used for inference.
        joint_conf: Optional 1D numpy array with a confidence score for each frame.
        s_val: Parameter that controls the smoothness of the spline. Default is 0.
               If s_val is negative infinity, uses confidence-weighted interpolation.
        k_val: Degree of the spline. Default is 3 (cubic).
    
    Returns:
        Tuple of (interpolated_joint, error)
        interpolated_joint: 2D array with interpolated coordinates for all frames, or None
        error: Exception if interpolation failed, or None
    """
    n = len(joint)
    if n <= k_val:
        return None, NotEnoughData(
            f"Must have more samples ({n}) than the order of the B-spline ({k_val})",
            n, k_val + 1
        )
    
    # Get valid indices (where joint was detected)
    valid_indices = downsampling_vector
    
    # Interpolate x and y coordinates separately
    try:
        if s_val == float('-inf') and joint_conf is not None:
            # Confidence-weighted interpolation
            # Use square of confidence as weights
            weights = joint_conf ** 2
            # Filter out zero confidence points
            valid_mask = weights > 0
            if np.sum(valid_mask) <= k_val:
                # Fall back to unweighted interpolation
                weights = None
            else:
                weights = weights[valid_mask]
                valid_indices = valid_indices[valid_mask]
                joint = joint[valid_mask]
        else:
            weights = None
        
        # Interpolate x coordinates
        if weights is not None:
            spline_x = UnivariateSpline(
                valid_indices, joint[:, 0],
                s=s_val if s_val != float('-inf') else 0,
                w=weights,
                k=k_val
            )
        else:
            spline_x = UnivariateSpline(
                valid_indices, joint[:, 0],
                s=s_val if s_val != float('-inf') else 0,
                k=k_val
            )
        
        # Interpolate y coordinates
        if weights is not None:
            spline_y = UnivariateSpline(
                valid_indices, joint[:, 1],
                s=s_val if s_val != float('-inf') else 0,
                w=weights,
                k=k_val
            )
        else:
            spline_y = UnivariateSpline(
                valid_indices, joint[:, 1],
                s=s_val if s_val != float('-inf') else 0,
                k=k_val
            )
        
        # Generate interpolated values for all frames
        n_frames = int(downsampling_vector.max()) + 1
        frame_indices = np.arange(n_frames)
        
        interpolated_x = spline_x(frame_indices)
        interpolated_y = spline_y(frame_indices)
        
        interpolated_joint = np.stack([interpolated_x, interpolated_y], axis=1)
        
        return interpolated_joint, None
        
    except Exception as e:
        return None, InterpolationError(f"Interpolation failed: {str(e)}")


def interpolate_full(
    downsampled_skeleton: np.ndarray,
    inference_frameindex: np.ndarray,
    joint_confs: np.ndarray,
    s_val: float = 0.0,
    conf_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, Optional[Exception]]:
    """
    Interpolates the position of all joints of a downsampled skeleton for all frames.
    
    Args:
        downsampled_skeleton: A 3D numpy array with the x and y coordinates of all joints
                             of the downsampled skeleton for the frames selected for inference.
                             Shape (N-frames, N-joints, 2)
        inference_frameindex: A vector of indices indicating which frames of the original
                             video have been sampled
        joint_confs: An array (N-frames, N-joints) with a confidence score for each joint
                    in each frame
        s_val: Parameter that controls the smoothness of the spline. Default is 0.
               If float('-inf'), uses confidence-weighted interpolation.
        conf_threshold: The confidence threshold below which a joint is not considered
                       for interpolation.
    
    Returns:
        Tuple of (interpolated_joints, start_end_sequence, error)
        interpolated_joints: Array containing the interpolated joints for all frames.
                           Shape (max_frame+1, N-joints, 2)
        start_end_sequence: Array containing the start and end frame each joint is detected.
                          Shape (N-joints, 2)
        error: Exception if interpolation failed, or None
    """
    n_frames = int(inference_frameindex.max()) + 1
    n_joints = downsampled_skeleton.shape[1]
    
    interpolated_joints = np.empty((n_frames, n_joints, 2))
    start_end_sequence = np.zeros((n_joints, 2), dtype=int)
    
    try:
        for j in range(n_joints):
            # Extract joint positions and confidences for this joint
            joint_positions = downsampled_skeleton[:, j, :]
            joint_confidences = joint_confs[:, j]
            
            # Filter by confidence threshold
            valid_mask = joint_confidences >= conf_threshold
            valid_positions = joint_positions[valid_mask]
            valid_indices = inference_frameindex[valid_mask]
            valid_confs = joint_confidences[valid_mask]
            
            if len(valid_positions) == 0:
                # No valid detections for this joint
                interpolated_joints[:, j, :] = -1  # Mark as invalid
                start_end_sequence[j, :] = [-1, -1]
                continue
            
            # Record start and end frames
            start_end_sequence[j, 0] = int(valid_indices.min())
            start_end_sequence[j, 1] = int(valid_indices.max())
            
            # Interpolate this joint
            interpolated_joint, error = interpolate_joint(
                valid_positions,
                valid_indices,
                joint_conf=valid_confs if s_val == float('-inf') else None,
                s_val=s_val,
                k_val=3
            )
            
            if error is not None:
                logger.warning(f"Interpolation failed for joint {j}: {error}")
                # Fill with last known position or -1
                if len(valid_positions) > 0:
                    last_pos = valid_positions[-1]
                    interpolated_joints[:, j, :] = last_pos
                else:
                    interpolated_joints[:, j, :] = -1
            else:
                interpolated_joints[:, j, :] = interpolated_joint
        
        return interpolated_joints, start_end_sequence, None
        
    except Exception as e:
        return interpolated_joints, start_end_sequence, InterpolationError(f"Full interpolation failed: {str(e)}")






