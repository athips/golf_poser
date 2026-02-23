"""
Club detection and tracking utilities.

This module implements the full club detection pipeline including:
- Background subtraction
- Club cone vector calculation
- Polygon masking
- Line detection
- Blob detection
- Trust point calculation
- Club joint stitching
"""

import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)


def find_person_height(joints: np.ndarray) -> float:
    """
    Calculate the approximate height of a person based on specific joint coordinates.
    
    Args:
        joints: Array of shape (num_frames, num_joints, 2)
        
    Returns:
        Estimated height (distance between shoulders and feet midpoints)
    """
    # Use first frame for height estimation
    # Assuming joints: 0=head, 1=left_eye, 2=right_eye, 3=left_shoulder, 4=right_shoulder,
    #                  9=left_hip, 10=right_hip, 11=left_knee, 12=right_knee, 13=left_ankle, 14=right_ankle
    pose_joints = joints[0]  # First frame
    
    # Midpoint of shoulders (indices 3, 4)
    mid_shoulders = (pose_joints[3] + pose_joints[4]) / 2
    
    # Midpoint of feet/ankles (indices 13, 14)
    mid_feet = (pose_joints[13] + pose_joints[14]) / 2
    
    height = np.linalg.norm(mid_shoulders - mid_feet)
    return height


def rotate_vector(vectors: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a collection of 2D vectors by a specified angle.
    
    Args:
        vectors: Array of shape (N, 2) representing N 2D vectors
        angle: Rotation angle in degrees (counterclockwise)
        
    Returns:
        Array of shape (N, 2) containing the rotated vectors
    """
    angle_rad = np.radians(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    rotated_vectors = vectors @ rotation_matrix.T
    return rotated_vectors


def find_club_cone_vectors(club_joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes two club cone vectors based on club joint positions.
    
    This function calculates two vectors that represent the club's "cone" 
    by using three club handle joints. The vectors are rotated ±35 degrees
    from the mean club handle vector and scaled by person height.
    
    Args:
        club_joints: Array of shape (num_frames, num_joints, 2) containing 
                     joint coordinates. Last 4 joints should be club joints:
                     [handle1, handle2, handle3, club_head]
    
    Returns:
        Tuple of two arrays, each of shape (num_frames, 4):
            - club_handle_vector_counterclockwise: [x_start, x_end, y_start, y_end]
            - club_handle_vector_clockwise: [x_start, x_end, y_start, y_end]
    """
    raw_person_height = find_person_height(club_joints)
    
    # Extract club handle joints (indices -4, -3, -2)
    club_handle1 = club_joints[:, -4, :]  # First handle point
    club_handle2 = club_joints[:, -3, :]  # Second handle point
    club_handle3 = club_joints[:, -2, :]  # Third handle point
    
    # Calculate vectors between handle points
    club_handle_vector1 = club_handle2 - club_handle1
    club_handle_vector2 = club_handle3 - club_handle2
    club_handle_vector3 = club_handle3 - club_handle1
    
    # Mean club handle vector
    mean_club_handle_vector = (club_handle_vector1 + club_handle_vector2 + club_handle_vector3) / 3
    
    # Rotate by ±35 degrees
    mean_club_handle_vector_counterclockwise = rotate_vector(mean_club_handle_vector, 35)
    mean_club_handle_vector_clockwise = rotate_vector(mean_club_handle_vector, -35)
    
    # Normalize and scale by person height * 2.5
    norm_ccw = np.linalg.norm(mean_club_handle_vector_counterclockwise, axis=-1, keepdims=True)
    norm_cw = np.linalg.norm(mean_club_handle_vector_clockwise, axis=-1, keepdims=True)
    
    # Avoid division by zero
    norm_ccw = np.where(norm_ccw > 0, norm_ccw, 1.0)
    norm_cw = np.where(norm_cw > 0, norm_cw, 1.0)
    
    club_handle_vector_counterclockwise = (mean_club_handle_vector_counterclockwise / norm_ccw) * raw_person_height * 2.5
    club_handle_vector_clockwise = (mean_club_handle_vector_clockwise / norm_cw) * raw_person_height * 2.5
    
    # Stack into [x_start, x_end, y_start, y_end] format
    club_handle_vector_counterclockwise = np.stack([
        club_handle1[:, 0],
        club_handle1[:, 0] + club_handle_vector_counterclockwise[:, 0],
        club_handle1[:, 1],
        club_handle1[:, 1] + club_handle_vector_counterclockwise[:, 1]
    ], axis=1)
    
    club_handle_vector_clockwise = np.stack([
        club_handle1[:, 0],
        club_handle1[:, 0] + club_handle_vector_clockwise[:, 0],
        club_handle1[:, 1],
        club_handle1[:, 1] + club_handle_vector_clockwise[:, 1]
    ], axis=1)
    
    return club_handle_vector_counterclockwise, club_handle_vector_clockwise


def mog2_background_subtraction(frames: np.ndarray) -> np.ndarray:
    """
    Performs background subtraction on video frames using KNN algorithm.
    
    Processes frames in reverse order to build up the background model,
    then returns masks in original frame order.
    
    Args:
        frames: Array of video frames with shape (frames, height, width, channels)
                or (frames, channels, height, width)
        
    Returns:
        Array of binary masks with shape (frames, height, width)
    """
    # Handle different input formats
    if frames.ndim == 4:
        if frames.shape[1] == 3 or frames.shape[1] == 1:  # (frames, channels, height, width)
            frames_transposed = frames.transpose(0, 2, 3, 1)
        else:  # (frames, height, width, channels)
            frames_transposed = frames
    else:
        raise ValueError(f"Expected 4D array, got shape {frames.shape}")
    
    knn = cv2.createBackgroundSubtractorKNN(history=20, detectShadows=False)
    masks = []
    
    # Process in reverse order
    for frame in frames_transposed[::-1]:
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        mask_knn = knn.apply(frame)
        masks.append(mask_knn)
    
    # Reverse back to original order
    masks = np.array(masks)[::-1]
    return masks


def detect_lines(mask: np.ndarray) -> np.ndarray:
    """
    Detects lines in a binary mask using the Hough transform.
    
    Args:
        mask: Binary mask of shape (height, width)
    
    Returns:
        Array of detected lines with shape (num_lines, 4) where each line is [x1, y1, x2, y2]
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 360,
        threshold=30,
        minLineLength=20,
        maxLineGap=400
    )
    
    if lines is None:
        return np.array([]).reshape(0, 4)
    
    return lines.reshape(-1, 4)


def polygon_masking(line_cw: np.ndarray, line_ccw: np.ndarray, image_shape: Tuple) -> np.ndarray:
    """
    Creates a binary mask for a polygon defined by two lines.
    
    Args:
        line_cw: Array of shape (N, 4) containing clockwise line coordinates [x1, x2, y1, y2]
        line_ccw: Array of shape (N, 4) containing counter-clockwise line coordinates [x1, x2, y1, y2]
        image_shape: Tuple of (frames, height, width) or (height, width) defining output mask dimensions
    
    Returns:
        Binary mask array of shape (frames, height, width) where polygon area is True
    """
    if len(image_shape) == 3:
        num_frames, height, width = image_shape
    else:
        num_frames = line_cw.shape[0]
        height, width = image_shape
    
    mask = np.zeros((num_frames, height, width), dtype=np.uint8)
    
    for i in range(num_frames):
        # Extract line coordinates
        # line format: [x_start, x_end, y_start, y_end]
        x1_cw, x2_cw, y1_cw, y2_cw = line_cw[i]
        x1_ccw, x2_ccw, y1_ccw, y2_ccw = line_ccw[i]
        
        # Create polygon from line endpoints
        polygon = np.array([
            [x1_cw, y1_cw],   # Start of clockwise line
            [x2_cw, y2_cw],   # End of clockwise line
            [x2_ccw, y2_ccw], # End of counter-clockwise line
            [x1_ccw, y1_ccw]  # Start of counter-clockwise line
        ], dtype=np.int32)
        
        cv2.fillPoly(mask[i], [polygon], 1)
    
    return mask.astype(bool)


def polygon_masking_outer(line_cw: np.ndarray, line_ccw: np.ndarray, image_shape: Tuple) -> np.ndarray:
    """
    Creates a binary mask for the outer part of the polygon.
    
    Uses the end of lines and 1/4 of the way from start to end.
    
    Args:
        line_cw: Array of shape (N, 4) containing clockwise line coordinates [x1, x2, y1, y2]
        line_ccw: Array of shape (N, 4) containing counter-clockwise line coordinates [x1, x2, y1, y2]
        image_shape: Tuple of (frames, height, width) or (height, width)
    
    Returns:
        Binary mask array of shape (frames, height, width)
    """
    if len(image_shape) == 3:
        num_frames, height, width = image_shape
    else:
        num_frames = line_cw.shape[0]
        height, width = image_shape
    
    mask = np.zeros((num_frames, height, width), dtype=np.uint8)
    
    for i in range(num_frames):
        x1_cw, x2_cw, y1_cw, y2_cw = line_cw[i]
        x1_ccw, x2_ccw, y1_ccw, y2_ccw = line_ccw[i]
        
        # Calculate 1/4 point along lines
        line_cw_start = np.array([x1_cw, y1_cw])
        line_cw_end = np.array([x2_cw, y2_cw])
        line_ccw_start = np.array([x1_ccw, y1_ccw])
        line_ccw_end = np.array([x2_ccw, y2_ccw])
        
        line_cw_part = line_cw_start + (line_cw_end - line_cw_start) / 4
        line_ccw_part = line_ccw_start + (line_ccw_end - line_ccw_start) / 4
        
        # Create polygon from outer points
        polygon = np.array([
            line_cw_end,
            line_cw_part,
            line_ccw_part,
            line_ccw_end
        ], dtype=np.int32)
        
        cv2.fillPoly(mask[i], [polygon], 1)
    
    return mask.astype(bool)


def polygon_mask_skeleton(joints: np.ndarray, image_shape: Tuple) -> np.ndarray:
    """
    Creates a binary mask for a skeleton defined by joint positions.
    
    Args:
        joints: Array of shape (frames, num_joints, 2) containing joint coordinates [x, y]
        image_shape: Tuple of (frames, height, width) or (height, width)
    
    Returns:
        Binary mask array of shape (frames, height, width) where skeleton area is inverted (True = not skeleton)
    """
    if len(image_shape) == 3:
        num_frames, height, width = image_shape
    else:
        num_frames = joints.shape[0]
        height, width = image_shape
    
    skeleton_mask = np.zeros((num_frames, height, width), dtype=np.uint8)
    
    # Skeleton connections for masking (based on original implementation)
    skeleton_connections = [
        (0, 1), (0, 2),  # head to eyes
        (3, 4), (4, 10), (3, 9), (9, 10),  # shoulders and hips
        (3, 5), (5, 7), (4, 6), (6, 8),  # arms
        (9, 11), (11, 13), (10, 12), (12, 14),  # legs
    ]
    
    for frame in range(num_frames):
        for i, (start_idx, end_idx) in enumerate(skeleton_connections):
            if start_idx >= joints.shape[1] or end_idx >= joints.shape[1]:
                continue
            
            # Determine thickness based on connection type
            if i > 11:
                thickness = 50
            elif i > 3:
                thickness = 20
            else:
                thickness = 30
            
            start_point = tuple(map(int, joints[frame, start_idx]))
            end_point = tuple(map(int, joints[frame, end_idx]))
            
            # Check if points are valid (not negative)
            if start_point[0] >= 0 and start_point[1] >= 0 and end_point[0] >= 0 and end_point[1] >= 0:
                cv2.line(skeleton_mask[frame], start_point, end_point, 1, thickness)
    
    skeleton_mask = skeleton_mask.astype(bool)
    skeleton_mask = np.invert(skeleton_mask)  # Invert: True = not skeleton area
    return skeleton_mask


def blob_detection(mask: np.ndarray, return_mask: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detects blobs in a binary mask using SimpleBlobDetector.
    
    Args:
        mask: Binary mask of shape (height, width)
        return_mask: If True, also return the mask with blob centers drawn
    
    Returns:
        Tuple of (blob_centers, mask_with_blobs)
        blob_centers: Array of shape (num_blobs, 2) with [x, y] coordinates, or None
        mask_with_blobs: Mask with blob centers drawn, or None
    """
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 10000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(mask)
    
    if len(keypoints) == 0:
        return None, (mask.copy() if return_mask else None)
    
    blob_centers = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    
    mask_with_blobs = None
    if return_mask:
        mask_with_blobs = mask.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(mask_with_blobs, (x, y), 5, 255, -1)
    
    return blob_centers, mask_with_blobs


def sort_lines(lines: np.ndarray, line_clockwise: np.ndarray, line_counterclockwise: np.ndarray) -> np.ndarray:
    """
    Sorts lines based on their position relative to the club handle lines.
    
    Args:
        lines: Array of detected lines with shape (num_lines, 4) [x1, y1, x2, y2]
        line_clockwise: Array of shape (4,) containing clockwise line [x1, x2, y1, y2]
        line_counterclockwise: Array of shape (4,) containing counter-clockwise line [x1, x2, y1, y2]
    
    Returns:
        Array of sorted/validated lines
    """
    if lines.shape[0] == 0:
        return lines
    
    # Calculate line vectors
    linevector_cw = np.array([line_clockwise[1] - line_clockwise[0], line_clockwise[3] - line_clockwise[2]])
    linevector_ccw = np.array([line_counterclockwise[1] - line_counterclockwise[0], 
                               line_counterclockwise[3] - line_counterclockwise[2]])
    
    # Calculate angle between cone vectors
    angle_cone = np.arctan2(np.cross(linevector_cw, linevector_ccw), np.dot(linevector_cw, linevector_ccw))
    
    # Calculate vectors and midpoints for detected lines
    found_line_vectors = lines[:, 2:4] - lines[:, :2]
    found_line_midpoints = (lines[:, :2] + lines[:, 2:4]) / 2
    
    # Calculate angles relative to cone vectors
    valid_lines = []
    for i, (line_vec, line_mid) in enumerate(zip(found_line_vectors, found_line_midpoints)):
        angle_line_cw = np.arctan2(np.cross(line_vec, linevector_cw), np.dot(line_vec, linevector_cw))
        angle_line_ccw = np.arctan2(np.cross(line_vec, linevector_ccw), np.dot(line_vec, linevector_ccw))
        
        # Check if line is within cone
        valid_angle = (np.abs(angle_line_cw) < np.abs(angle_cone)) & (np.abs(angle_line_ccw) < np.abs(angle_cone))
        
        # Check position relative to cone
        vec_to_point_cw = line_mid - np.array([line_clockwise[0], line_clockwise[2]])
        vec_to_point_ccw = line_mid - np.array([line_counterclockwise[0], line_counterclockwise[2]])
        
        side_cw = np.cross(linevector_cw, vec_to_point_cw) > 0
        side_ccw = np.cross(linevector_ccw, vec_to_point_ccw) < 0
        valid_position = side_cw & side_ccw
        
        if valid_angle and valid_position:
            valid_lines.append(lines[i])
    
    if len(valid_lines) == 0:
        return np.array([]).reshape(0, 4)
    
    return np.array(valid_lines)


def angle_of_lines(lines: List[np.ndarray]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[np.ndarray]]]:
    """
    Computes statistical metrics for lines detected in multiple frames.
    
    Args:
        lines: List of arrays, each containing detected lines for a frame
    
    Returns:
        Tuple of:
            - mean_lines_angle: List of median angles per frame
            - max_lines_length: List of maximum line lengths per frame
            - median_endpoints: List of median endpoints per frame
    """
    mean_lines_angle = []
    max_lines_length = []
    median_endpoints = []
    
    reference_vector = np.array([-1, 0])
    
    for framelines in lines:
        if len(framelines) == 0:
            mean_lines_angle.append(None)
            max_lines_length.append(None)
            median_endpoints.append(None)
            continue
        
        lines_angle = []
        lines_length = []
        endpoints = []
        
        for line in framelines:
            line_vector = line[2:4] - line[:2]
            angle_line = np.arctan2(np.cross(line_vector, reference_vector), np.dot(line_vector, reference_vector))
            lines_angle.append(angle_line)
            lines_length.append(np.linalg.norm(line_vector))
            endpoints.append(line[2:4])  # Endpoint [x2, y2]
        
        mean_lines_angle.append(np.median(lines_angle))
        max_lines_length.append(np.max(lines_length))
        median_endpoints.append(np.median(endpoints, axis=0))
    
    return mean_lines_angle, max_lines_length, median_endpoints


def find_trust_points(
    median_endpoints: List[Optional[np.ndarray]],
    blob_centers: List[Optional[np.ndarray]],
    mean_angle: List[Optional[float]],
    midhands: np.ndarray
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """
    Finds trust points (club head positions) by combining line endpoints and blob detections.
    
    Args:
        median_endpoints: List of median line endpoints per frame
        blob_centers: List of blob center arrays per frame
        mean_angle: List of mean line angles per frame
        midhands: Array of shape (num_frames, 2) with mid-hand positions
    
    Returns:
        Tuple of (trust_points, endpoint_trust)
        trust_points: List of trust point arrays [x, y] per frame
        endpoint_trust: List of confidence scores per frame
    """
    trust_points = []
    endpoint_trust = []
    
    for i in range(len(median_endpoints)):
        # Use median endpoint if available
        if median_endpoints[i] is not None:
            # Check if endpoint is valid (not NaN, within reasonable bounds)
            if not np.isnan(median_endpoints[i]).any() and np.all(median_endpoints[i] >= 0):
                trust_points.append(median_endpoints[i])
                # Trust score based on whether blob was also detected
                if blob_centers[i] is not None and len(blob_centers[i]) > 0:
                    endpoint_trust.append(1.0)
                else:
                    endpoint_trust.append(0.8)
            else:
                trust_points.append(None)
                endpoint_trust.append(0.0)
        # Fallback: use blob center if no line endpoint
        elif blob_centers[i] is not None and len(blob_centers[i]) > 0:
            # Use the blob center closest to expected club head position
            blob_center = blob_centers[i]
            if blob_center.ndim == 2 and len(blob_center) > 0:
                # Use first blob or closest to hand
                if len(blob_center) == 1:
                    trust_points.append(blob_center[0])
                else:
                    # Find blob closest to expected position (extended from hand)
                    hand_pos = midhands[i] if i < len(midhands) else None
                    if hand_pos is not None and not np.isnan(hand_pos).any():
                        distances = np.linalg.norm(blob_center - hand_pos, axis=1)
                        closest_idx = np.argmin(distances)
                        trust_points.append(blob_center[closest_idx])
                    else:
                        trust_points.append(blob_center[0])
                endpoint_trust.append(0.6)  # Lower confidence for blob-only
            else:
                trust_points.append(None)
                endpoint_trust.append(0.0)
        else:
            trust_points.append(None)
            endpoint_trust.append(0.0)
    
    return trust_points, endpoint_trust


def detect_club_in_video(
    frames: np.ndarray,
    line_clockwise: np.ndarray,
    line_counterclockwise: np.ndarray,
    all_joints: np.ndarray
) -> Tuple[List[Optional[np.ndarray]], List[float]]:
    """
    Main function to detect club in video using computer vision.
    
    Args:
        frames: Array of video frames with shape (num_frames, height, width, channels)
        line_clockwise: Array of shape (num_frames, 4) with clockwise cone line [x1, x2, y1, y2]
        line_counterclockwise: Array of shape (num_frames, 4) with counter-clockwise cone line
        all_joints: Array of shape (num_frames, num_joints, 2) with all joint positions
    
    Returns:
        Tuple of (trust_points, endpoint_trust)
        trust_points: List of detected club head positions per frame
        endpoint_trust: List of confidence scores per frame
    """
    midhands = all_joints[:, -3, :]  # Mid-hand position (assuming index -3)
    
    # Background subtraction
    masks = mog2_background_subtraction(frames)
    masks_outer = masks.copy()
    
    # Create polygon masks
    image_shape = frames.shape[:3] if frames.ndim == 4 else (frames.shape[0], frames.shape[1], frames.shape[2])
    polygonmask = polygon_masking(line_clockwise, line_counterclockwise, image_shape)
    polygonmask_outer = polygon_masking_outer(line_clockwise, line_counterclockwise, image_shape)
    skeleton_mask = polygon_mask_skeleton(all_joints, image_shape)
    
    # Apply masks
    masks = masks * polygonmask * skeleton_mask
    masks_outer = masks_outer * polygonmask_outer * skeleton_mask
    
    # Process each frame
    lines = []
    blob_centers = []
    
    for i, mask in enumerate(masks):
        if i < 2:  # Skip first 2 frames
            lines.append([])
            blob_centers.append(None)
            continue
        
        # Morphological operations
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_eroded = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, open_kernel, iterations=1)
        
        # Detect lines
        lines_eroded = detect_lines(mask_eroded)
        
        if lines_eroded.shape[0] == 0:
            # Try with dilated mask
            mask_dilated = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8))
            lines_eroded = detect_lines(mask_dilated)
        
        # Sort/validate lines
        if lines_eroded.shape[0] > 0:
            lines_eroded = sort_lines(lines_eroded, line_clockwise[i], line_counterclockwise[i])
        
        lines.append(lines_eroded)
        
        # Blob detection on outer mask
        if i < len(masks_outer):
            blob_result, _ = blob_detection(masks_outer[i].astype(np.uint8), return_mask=False)
            blob_centers.append(blob_result)
        else:
            blob_centers.append(None)
    
    # Calculate statistics
    mean_angle, max_length, median_endpoints = angle_of_lines(lines)
    
    # Find trust points
    trust_points, endpoint_trust = find_trust_points(median_endpoints, blob_centers, mean_angle, midhands)
    
    return trust_points, endpoint_trust


def stitch_club_joints(
    joints: np.ndarray,
    joint_confs: np.ndarray,
    trust_points: List[Optional[np.ndarray]],
    endpoint_trust: List[float],
    min_frame: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitches detected club head positions with pose model club joints.
    
    Args:
        joints: Array of shape (num_frames, num_joints, 2) with all joint positions
        joint_confs: Array of shape (num_frames, num_joints) with confidence scores
        trust_points: List of detected club head positions per frame
        endpoint_trust: List of confidence scores per frame
        min_frame: Minimum frame index to start using trust points
    
    Returns:
        Tuple of (updated_joints, updated_joint_confs)
    """
    updated_joints = joints.copy()
    updated_joint_confs = joint_confs.copy()
    
    hosel_joints = joints[:, -1, :]  # Club head (last joint)
    
    for i, trust_point in enumerate(trust_points):
        if trust_point is None:
            continue
        if i < min_frame:
            continue
        
        # Replace hosel position with trust point
        hosel_joints[i] = trust_point
        updated_joint_confs[i, -1] = endpoint_trust[i]
    
    updated_joints[:, -1, :] = hosel_joints
    
    return updated_joints, updated_joint_confs


def filter_hand_dist(
    club_joints: np.ndarray,
    hosel_joints: np.ndarray,
    joint_confs: np.ndarray,
    pose_conf_threshold: float = 0.7,
    framerate: float = 30.0
) -> np.ndarray:
    """
    Filters club head positions based on distance from hands.
    
    Args:
        club_joints: Array of shape (num_frames, num_joints, 2) with club joints
        hosel_joints: Array of shape (num_frames, 2) with hosel (club head) positions
        joint_confs: Array of shape (num_frames, num_joints) with confidence scores
        pose_conf_threshold: Minimum confidence threshold
        framerate: Video framerate
    
    Returns:
        Updated club_joints array
    """
    from .utils import filter_joints
    
    confidence_mask = joint_confs[:, -1] >= pose_conf_threshold
    hand_positions = club_joints[confidence_mask, -3, :]  # Mid-hand position
    
    if len(hand_positions) == 0:
        return club_joints
    
    # Calculate hand-to-hosel distance
    hand_dist = np.linalg.norm(hosel_joints[confidence_mask] - hand_positions, axis=-1)
    
    # Filter the distance
    filtered_hand_dist = filter_joints(hand_dist.reshape(-1, 1), framerate, cutoff_hz=10.0)
    filtered_hand_dist = filtered_hand_dist.flatten()
    
    # Calculate direction vector from hand to hosel
    hand_hosel_vec = hosel_joints[confidence_mask] - hand_positions
    hand_hosel_norm = np.linalg.norm(hand_hosel_vec, axis=-1, keepdims=True)
    hand_hosel_norm = np.where(hand_hosel_norm > 0, hand_hosel_norm, 1.0)
    hand_hosel_vec_normalized = hand_hosel_vec / hand_hosel_norm
    
    # Update hosel positions
    updated_hosel = hand_positions + hand_hosel_vec_normalized * filtered_hand_dist[:, None]
    club_joints[confidence_mask, -1, :] = updated_hosel
    
    return club_joints

