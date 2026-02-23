"""
One-Euro Filter for smoothing keypoint trajectories over time.

The One-Euro Filter is a low-pass filter that adapts its cutoff frequency
based on the speed of motion, making it ideal for smoothing trajectories
with varying speeds (e.g., golf swings with different phases).

Reference: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input
in Interactive Systems" by Casiez et al.
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OneEuroFilter:
    """
    One-Euro Filter for smoothing a single signal value over time.
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        freq: float = 30.0
    ):
        """
        Initialize One-Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency for the low-pass filter (Hz).
                       Controls the minimum amount of smoothing.
            beta: Speed coefficient. Higher values increase responsiveness to
                  fast motion. 0.0 means no speed adaptation.
            d_cutoff: Cutoff frequency for the derivative (speed) filter (Hz).
            freq: Sampling frequency (Hz). Should match video FPS.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        
        # Internal state
        self.x_prev = None  # Previous filtered value
        self.dx_prev = None  # Previous derivative (speed)
        self.alpha_d = self._alpha(d_cutoff)  # Alpha for derivative filter
        
    def _alpha(self, cutoff: float) -> float:
        """Calculate alpha parameter for exponential smoothing."""
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a single value.
        
        Args:
            x: Current input value
            t: Optional timestamp (if None, assumes constant time step)
            
        Returns:
            Filtered value
        """
        if self.x_prev is None:
            # First value: no filtering
            self.x_prev = x
            self.dx_prev = 0.0
            return x
        
        # Update time step if provided
        if t is not None:
            dt = t - getattr(self, 't_prev', t)
            if dt > 0:
                self.freq = 1.0 / dt
                self.alpha_d = self._alpha(self.d_cutoff)
            self.t_prev = t
        
        # Estimate derivative (speed) using exponential smoothing
        dx = (x - self.x_prev) * self.freq
        dx_hat = self.alpha_d * dx + (1.0 - self.alpha_d) * self.dx_prev
        
        # Adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff)
        
        # Apply low-pass filter
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        
        return x_hat
    
    def reset(self):
        """Reset filter state."""
        self.x_prev = None
        self.dx_prev = None


class OneEuroFilter2D:
    """
    One-Euro Filter for 2D coordinates (x, y).
    """
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        freq: float = 30.0
    ):
        """
        Initialize 2D One-Euro Filter.
        
        Args:
            min_cutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            d_cutoff: Derivative cutoff frequency (Hz)
            freq: Sampling frequency (Hz)
        """
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
    
    def filter(self, point: np.ndarray, t: Optional[float] = None) -> np.ndarray:
        """
        Filter a 2D point.
        
        Args:
            point: 2D point as numpy array [x, y]
            t: Optional timestamp
            
        Returns:
            Filtered 2D point
        """
        x_filtered = self.filter_x.filter(float(point[0]), t)
        y_filtered = self.filter_y.filter(float(point[1]), t)
        return np.array([x_filtered, y_filtered], dtype=np.float32)
    
    def reset(self):
        """Reset filter state."""
        self.filter_x.reset()
        self.filter_y.reset()


def apply_one_euro_filter(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    fps: float = 30.0,
    min_cutoff: float = 1.0,
    beta: float = 0.7,
    d_cutoff: float = 1.0,
    conf_threshold: float = 0.3,
    invalid_value: float = -1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply One-Euro Filter to smooth keypoint trajectories over time.
    
    Args:
        keypoints: Keypoints array of shape (num_frames, num_joints, 2)
        confidences: Confidence array of shape (num_frames, num_joints)
        fps: Video frame rate (Hz)
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing.
        beta: Speed coefficient. Higher = more responsive to fast motion.
        d_cutoff: Derivative cutoff frequency (Hz)
        conf_threshold: Only filter keypoints with confidence above this threshold
        invalid_value: Value that indicates invalid keypoints (e.g., -1.0)
        
    Returns:
        Tuple of (filtered_keypoints, confidences)
        filtered_keypoints: Smoothed keypoints array (num_frames, num_joints, 2)
        confidences: Unchanged confidence array
    """
    num_frames, num_joints, _ = keypoints.shape
    filtered_keypoints = keypoints.copy()
    
    # Create a filter for each joint
    filters = []
    for j in range(num_joints):
        filters.append(OneEuroFilter2D(
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            freq=fps
        ))
    
    # Apply filter frame by frame
    dt = 1.0 / fps if fps > 0 else None
    
    for i in range(num_frames):
        t = i * dt if dt is not None else None
        
        for j in range(num_joints):
            kp = keypoints[i, j, :]
            conf = confidences[i, j]
            
            # Only filter valid keypoints with sufficient confidence
            if conf >= conf_threshold and not np.allclose(kp, invalid_value):
                filtered_kp = filters[j].filter(kp, t)
                filtered_keypoints[i, j, :] = filtered_kp
            else:
                # Reset filter for invalid keypoints
                filters[j].reset()
                # Keep original (invalid) value
                filtered_keypoints[i, j, :] = kp
    
    logger.info(f"Applied One-Euro Filter to {num_frames} frames, {num_joints} joints")
    return filtered_keypoints, confidences
