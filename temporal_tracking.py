"""
Temporal Tracking and Sequence Consistency for Golf Pose Estimation.

This module ensures temporal consistency in pose estimation by:
1. Tracking keypoints across frames using optical flow and Kalman filtering
2. Enforcing biomechanical constraints (bone length, joint angles)
3. Validating motion patterns based on golf swing biomechanics
4. Correcting temporal inconsistencies and sudden jumps
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import logging
from collections import deque

logger = logging.getLogger(__name__)


# Golf pose skeleton structure (19 keypoints)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # head to eyes
    (3, 4), (4, 10), (3, 9), (9, 10),  # shoulders and hips
    (3, 5), (5, 7), (4, 6), (6, 8),  # arms
    (9, 11), (11, 13), (10, 12), (12, 14),  # legs
    (15, 16), (16, 17), (17, 18)  # club
]

# Bone pairs for length consistency checking
BONE_PAIRS = [
    (3, 5), (5, 7),  # left arm segments
    (4, 6), (6, 8),  # right arm segments
    (9, 11), (11, 13),  # left leg segments
    (10, 12), (12, 14),  # right leg segments
    (3, 4),  # shoulder width
    (9, 10),  # hip width
]

# Joint angle limits (in degrees) for biomechanical constraints
# Format: (joint_idx, min_angle, max_angle, parent_idx, child_idx)
JOINT_ANGLE_LIMITS = {
    # Elbow angles (should be between 0-180 degrees)
    5: (0, 180, 3, 7),  # left elbow
    6: (0, 180, 4, 8),  # right elbow
    # Knee angles
    11: (0, 180, 9, 13),  # left knee
    12: (0, 180, 10, 14),  # right knee
}


class KalmanTracker:
    """
    Kalman filter for tracking a single 2D keypoint.
    """
    
    def __init__(self, initial_pos: np.ndarray, process_noise: float = 0.03, measurement_noise: float = 0.3):
        """
        Initialize Kalman filter for 2D point tracking.
        
        Args:
            initial_pos: Initial position [x, y]
            process_noise: Process noise covariance (motion uncertainty)
            measurement_noise: Measurement noise covariance (observation uncertainty)
        """
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x' = x + vx
            [0, 1, 0, 1],  # y' = y + vy
            [0, 0, 1, 0],   # vx' = vx
            [0, 0, 0, 1]    # vy' = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we observe x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initialize state
        self.kf.statePre = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float32)
        self.kf.statePost = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0], dtype=np.float32)
        
        self.last_position = initial_pos.copy()
        self.is_initialized = True
    
    def predict(self) -> np.ndarray:
        """Predict next position."""
        prediction = self.kf.predict()
        return np.array([prediction[0], prediction[1]], dtype=np.float32)
    
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Update filter with new measurement."""
        corrected = self.kf.correct(measurement.astype(np.float32))
        self.last_position = np.array([corrected[0], corrected[1]], dtype=np.float32)
        return self.last_position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        state = self.kf.statePost
        return np.array([state[2], state[3]], dtype=np.float32)


class OpticalFlowTracker:
    """
    Optical flow-based tracking for keypoints.
    """
    
    def __init__(self):
        """Initialize optical flow tracker."""
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.prev_gray = None
        self.prev_points = None
    
    def track(self, current_frame: np.ndarray, keypoints: np.ndarray, confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track keypoints using optical flow.
        
        Args:
            current_frame: Current frame (BGR)
            keypoints: Keypoints to track (num_joints, 2)
            confidences: Confidence scores (num_joints,)
            
        Returns:
            Tuple of (tracked_keypoints, tracked_confidences)
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None or self.prev_points is None:
            # First frame - initialize
            self.prev_gray = gray
            valid_mask = confidences > 0.3
            self.prev_points = keypoints[valid_mask].reshape(-1, 1, 2).astype(np.float32)
            return keypoints.copy(), confidences.copy()
        
        # Track points using Lucas-Kanade optical flow
        if len(self.prev_points) > 0:
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None, **self.lk_params
            )
            
            # Update tracked keypoints
            tracked_keypoints = keypoints.copy()
            tracked_confidences = confidences.copy()
            
            # Update successfully tracked points
            valid_idx = 0
            for i in range(len(keypoints)):
                if confidences[i] > 0.3:
                    if status[valid_idx] == 1 and error[valid_idx] < 50.0:
                        # Good tracking
                        tracked_keypoints[i] = next_points[valid_idx, 0]
                        tracked_confidences[i] = min(1.0, confidences[i] + 0.1)  # Boost confidence
                    valid_idx += 1
        
        # Update for next frame
        self.prev_gray = gray
        valid_mask = confidences > 0.3
        self.prev_points = keypoints[valid_mask].reshape(-1, 1, 2).astype(np.float32)
        
        return tracked_keypoints, tracked_confidences


class BiomechanicalConstraints:
    """
    Enforces biomechanical constraints on pose.
    """
    
    def __init__(self, bone_length_tolerance: float = 0.15, max_velocity: float = 200.0):
        """
        Initialize biomechanical constraints.
        
        Args:
            bone_length_tolerance: Maximum allowed change in bone length (15% of original)
            max_velocity: Maximum allowed keypoint velocity (pixels per frame)
        """
        self.bone_length_tolerance = bone_length_tolerance
        self.max_velocity = max_velocity
        self.bone_lengths = {}  # Cache of bone lengths
    
    def compute_bone_length(self, kp1: np.ndarray, kp2: np.ndarray) -> float:
        """Compute distance between two keypoints."""
        return np.linalg.norm(kp2 - kp1)
    
    def check_bone_length_consistency(
        self,
        keypoints: np.ndarray,
        prev_keypoints: Optional[np.ndarray],
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check and correct bone length consistency.
        
        Args:
            keypoints: Current keypoints (num_joints, 2)
            prev_keypoints: Previous frame keypoints (num_joints, 2) or None
            confidences: Confidence scores (num_joints,)
            
        Returns:
            Tuple of (corrected_keypoints, corrected_confidences)
        """
        corrected_keypoints = keypoints.copy()
        corrected_confidences = confidences.copy()
        
        if prev_keypoints is None:
            # Initialize bone lengths
            for start_idx, end_idx in BONE_PAIRS:
                if confidences[start_idx] > 0.3 and confidences[end_idx] > 0.3:
                    length = self.compute_bone_length(keypoints[start_idx], keypoints[end_idx])
                    self.bone_lengths[(start_idx, end_idx)] = length
            return corrected_keypoints, corrected_confidences
        
        # Check bone lengths
        for start_idx, end_idx in BONE_PAIRS:
            if confidences[start_idx] < 0.3 or confidences[end_idx] < 0.3:
                continue
            
            current_length = self.compute_bone_length(keypoints[start_idx], keypoints[end_idx])
            
            if (start_idx, end_idx) in self.bone_lengths:
                expected_length = self.bone_lengths[(start_idx, end_idx)]
                length_change = abs(current_length - expected_length) / max(expected_length, 1.0)
                
                if length_change > self.bone_length_tolerance:
                    # Bone length changed too much - correct it
                    direction = (keypoints[end_idx] - keypoints[start_idx])
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        # Adjust end point to maintain bone length
                        corrected_keypoints[end_idx] = keypoints[start_idx] + direction * expected_length
                        corrected_confidences[end_idx] = max(0.3, confidences[end_idx] - 0.1)
            else:
                # Store initial bone length
                self.bone_lengths[(start_idx, end_idx)] = current_length
        
        return corrected_keypoints, corrected_confidences
    
    def check_joint_angles(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check joint angles against biomechanical limits.
        
        Args:
            keypoints: Current keypoints (num_joints, 2)
            confidences: Confidence scores (num_joints,)
            
        Returns:
            Tuple of (corrected_keypoints, corrected_confidences)
        """
        corrected_keypoints = keypoints.copy()
        corrected_confidences = confidences.copy()
        
        for joint_idx, (min_angle, max_angle, parent_idx, child_idx) in JOINT_ANGLE_LIMITS.items():
            if (confidences[joint_idx] < 0.3 or 
                confidences[parent_idx] < 0.3 or 
                confidences[child_idx] < 0.3):
                continue
            
            # Compute joint angle
            vec1 = keypoints[joint_idx] - keypoints[parent_idx]
            vec2 = keypoints[child_idx] - keypoints[joint_idx]
            
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                
                # Check if angle is within limits
                if angle < min_angle or angle > max_angle:
                    # Angle out of range - reduce confidence
                    corrected_confidences[joint_idx] = max(0.2, confidences[joint_idx] - 0.2)
        
        return corrected_keypoints, corrected_confidences
    
    def check_velocity_limits(
        self,
        keypoints: np.ndarray,
        prev_keypoints: np.ndarray,
        confidences: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check keypoint velocities against maximum limits.
        
        Args:
            keypoints: Current keypoints (num_joints, 2)
            prev_keypoints: Previous frame keypoints (num_joints, 2)
            confidences: Confidence scores (num_joints,)
            
        Returns:
            Tuple of (corrected_keypoints, corrected_confidences)
        """
        corrected_keypoints = keypoints.copy()
        corrected_confidences = confidences.copy()
        
        for i in range(len(keypoints)):
            if confidences[i] < 0.3:
                continue
            
            velocity = np.linalg.norm(keypoints[i] - prev_keypoints[i])
            
            if velocity > self.max_velocity:
                # Velocity too high - likely an error
                # Interpolate between previous and current position
                alpha = self.max_velocity / max(velocity, 1.0)
                corrected_keypoints[i] = prev_keypoints[i] + alpha * (keypoints[i] - prev_keypoints[i])
                corrected_confidences[i] = max(0.2, confidences[i] - 0.2)
        
        return corrected_keypoints, corrected_confidences


class TemporalTracker:
    """
    Main temporal tracking system combining Kalman filtering, optical flow, and biomechanical constraints.
    """
    
    def __init__(
        self,
        num_joints: int = 19,
        use_kalman: bool = True,
        use_optical_flow: bool = True,
        use_biomechanical: bool = True,
        max_velocity: float = 200.0
    ):
        """
        Initialize temporal tracker.
        
        Args:
            num_joints: Number of keypoints
            use_kalman: Enable Kalman filtering
            use_optical_flow: Enable optical flow tracking
            use_biomechanical: Enable biomechanical constraints
            max_velocity: Maximum allowed keypoint velocity
        """
        self.num_joints = num_joints
        self.use_kalman = use_kalman
        self.use_optical_flow = use_optical_flow
        self.use_biomechanical = use_biomechanical
        
        # Initialize trackers
        self.kalman_trackers = [None] * num_joints
        self.optical_flow_tracker = OpticalFlowTracker() if use_optical_flow else None
        self.biomechanical = BiomechanicalConstraints(max_velocity=max_velocity) if use_biomechanical else None
        
        # History for temporal smoothing
        self.keypoint_history = deque(maxlen=5)  # Store last 5 frames
        self.confidence_history = deque(maxlen=5)
        
        self.prev_keypoints = None
        self.prev_confidences = None
    
    def track_frame(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        frame: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track keypoints for a single frame.
        
        Args:
            keypoints: Detected keypoints (num_joints, 2)
            confidences: Confidence scores (num_joints,)
            frame: Optional current frame for optical flow
            
        Returns:
            Tuple of (tracked_keypoints, tracked_confidences)
        """
        tracked_keypoints = keypoints.copy()
        tracked_confidences = confidences.copy()
        
        # Initialize Kalman filters for new keypoints
        if self.prev_keypoints is None:
            self.prev_keypoints = keypoints.copy()
            self.prev_confidences = confidences.copy()
            
            # Initialize Kalman filters
            if self.use_kalman:
                for i in range(self.num_joints):
                    if confidences[i] > 0.3:
                        self.kalman_trackers[i] = KalmanTracker(keypoints[i])
            
            self.keypoint_history.append(keypoints.copy())
            self.confidence_history.append(confidences.copy())
            return tracked_keypoints, tracked_confidences
        
        # Step 1: Optical flow tracking (if enabled and frame provided)
        if self.use_optical_flow and frame is not None:
            tracked_keypoints, tracked_confidences = self.optical_flow_tracker.track(
                frame, tracked_keypoints, tracked_confidences
            )
        
        # Step 2: Kalman filtering (if enabled)
        if self.use_kalman:
            for i in range(self.num_joints):
                if tracked_confidences[i] > 0.3:
                    if self.kalman_trackers[i] is None:
                        # Initialize new tracker
                        self.kalman_trackers[i] = KalmanTracker(tracked_keypoints[i])
                    else:
                        # Predict and update
                        predicted = self.kalman_trackers[i].predict()
                        # Use weighted combination of prediction and measurement
                        alpha = 0.7  # Trust measurement more
                        tracked_keypoints[i] = alpha * tracked_keypoints[i] + (1 - alpha) * predicted
                        tracked_keypoints[i] = self.kalman_trackers[i].update(tracked_keypoints[i])
                else:
                    # Low confidence - use prediction if available
                    if self.kalman_trackers[i] is not None:
                        predicted = self.kalman_trackers[i].predict()
                        tracked_keypoints[i] = predicted
                        tracked_confidences[i] = 0.3  # Low confidence for predicted points
        
        # Step 3: Biomechanical constraints
        if self.use_biomechanical:
            # Check bone length consistency
            tracked_keypoints, tracked_confidences = self.biomechanical.check_bone_length_consistency(
                tracked_keypoints, self.prev_keypoints, tracked_confidences
            )
            
            # Check joint angles
            tracked_keypoints, tracked_confidences = self.biomechanical.check_joint_angles(
                tracked_keypoints, tracked_confidences
            )
            
            # Check velocity limits
            tracked_keypoints, tracked_confidences = self.biomechanical.check_velocity_limits(
                tracked_keypoints, self.prev_keypoints, tracked_confidences
            )
        
        # Step 4: Temporal smoothing using history
        if len(self.keypoint_history) >= 3:
            # Use median filter for temporal smoothing
            history_array = np.array(list(self.keypoint_history))
            for i in range(self.num_joints):
                if tracked_confidences[i] > 0.3:
                    # Compute median of last few frames
                    recent_positions = history_array[-3:, i, :]
                    median_pos = np.median(recent_positions, axis=0)
                    # Blend with current position
                    tracked_keypoints[i] = 0.6 * tracked_keypoints[i] + 0.4 * median_pos
        
        # Update history
        self.prev_keypoints = tracked_keypoints.copy()
        self.prev_confidences = tracked_confidences.copy()
        self.keypoint_history.append(tracked_keypoints.copy())
        self.confidence_history.append(tracked_confidences.copy())
        
        return tracked_keypoints, tracked_confidences
    
    def reset(self):
        """Reset tracker state."""
        self.kalman_trackers = [None] * self.num_joints
        self.keypoint_history.clear()
        self.confidence_history.clear()
        self.prev_keypoints = None
        self.prev_confidences = None
        if self.optical_flow_tracker:
            self.optical_flow_tracker.prev_gray = None
            self.optical_flow_tracker.prev_points = None


def apply_temporal_tracking(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    frames: Optional[List[np.ndarray]] = None,
    fps: float = 30.0,
    use_kalman: bool = True,
    use_optical_flow: bool = True,
    use_biomechanical: bool = True,
    max_velocity: float = 200.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply temporal tracking to keypoint sequences.
    
    Args:
        keypoints: Keypoints array (num_frames, num_joints, 2)
        confidences: Confidence array (num_frames, num_joints)
        frames: Optional list of video frames for optical flow
        fps: Video frame rate
        use_kalman: Enable Kalman filtering
        use_optical_flow: Enable optical flow tracking
        use_biomechanical: Enable biomechanical constraints
        max_velocity: Maximum allowed keypoint velocity (pixels per frame)
        
    Returns:
        Tuple of (tracked_keypoints, tracked_confidences)
    """
    num_frames, num_joints, _ = keypoints.shape
    
    # Adjust max_velocity based on fps
    max_velocity_per_frame = max_velocity / fps if fps > 0 else max_velocity
    
    # Initialize tracker
    tracker = TemporalTracker(
        num_joints=num_joints,
        use_kalman=use_kalman,
        use_optical_flow=use_optical_flow,
        use_biomechanical=use_biomechanical,
        max_velocity=max_velocity_per_frame
    )
    
    tracked_keypoints = np.zeros_like(keypoints)
    tracked_confidences = np.zeros_like(confidences)
    
    # Process each frame
    for i in range(num_frames):
        frame = frames[i] if frames is not None and i < len(frames) else None
        
        kp, conf = tracker.track_frame(
            keypoints[i],
            confidences[i],
            frame=frame
        )
        
        tracked_keypoints[i] = kp
        tracked_confidences[i] = conf
    
    logger.info(f"Applied temporal tracking to {num_frames} frames, {num_joints} joints")
    return tracked_keypoints, tracked_confidences
