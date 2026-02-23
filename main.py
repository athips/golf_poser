"""
2D Pose Prediction Pipeline
Reproduces functionality from kinematics_2d/posepredict.py
"""
import os
# Fix OpenMP duplicate library error - must be set before any imports
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import logging
import argparse
from pathlib import Path

from .model_wrappers import (
    ViewDetectionModel, 
    PoseEstimationModel, 
    YOLOPersonDetector
)
from .data_classes import View, BoundingBox
from .utils import expand_bounding_box, rescale_pose_coordinates

logger = logging.getLogger(__name__)


def _resolve_model_path(model_path: str, search_dirs: List[Path]) -> str:
    """
    Resolve a model path by searching common directories.
    Falls back to the original path if nothing is found.
    """
    candidate = Path(model_path)
    if candidate.is_file():
        return str(candidate)

    # Only search for non-absolute or missing paths
    if not candidate.is_absolute():
        # Try exact path under each search directory
        for root in search_dirs:
            exact = root / model_path
            if exact.is_file():
                return str(exact)

        # Try basename under each search directory
        basename = candidate.name
        for root in search_dirs:
            by_name = root / basename
            if by_name.is_file():
                return str(by_name)

    return model_path


class PosePredictor:
    """Main class for 2D pose prediction from video."""
    
    def __init__(
        self,
        view_model_path: str = 'models/classify_view.onnx',
        pose_fo_model_path: str = 'models/golfer_fo2.onnx',
        pose_dl_model_path: str = 'models/golfer_dl.onnx',
        yolo_model_path: str = 'models/yolo11n.pt',
        device: str = 'cuda',
        use_person_detection: bool = True
    ):
        """
        Initialize pose predictor.
        
        Args:
            view_model_path: Path to view detection model
            pose_fo_model_path: Path to Face-On pose model
            pose_dl_model_path: Path to Down-the-Line pose model
            yolo_model_path: Path to YOLO person detection model
            device: 'cuda' or 'cpu'
            use_person_detection: Whether to use YOLO for person detection
        """
        self.device = device
        self.use_person_detection = use_person_detection

        # Resolve model paths from common folders
        repo_root = Path(__file__).resolve().parents[1]
        search_dirs = [
            Path.cwd(),
            repo_root,
            repo_root / "model",
            repo_root / "models",
            repo_root / "onnx_models",
            repo_root / "view_models",
        ]
        view_model_path = _resolve_model_path(view_model_path, search_dirs)
        pose_fo_model_path = _resolve_model_path(pose_fo_model_path, search_dirs)
        pose_dl_model_path = _resolve_model_path(pose_dl_model_path, search_dirs)
        yolo_model_path = _resolve_model_path(yolo_model_path, search_dirs)
        
        # Load models
        logger.info("Loading models...")
        self.view_model = ViewDetectionModel(view_model_path, device)
        self.pose_fo_model = PoseEstimationModel(pose_fo_model_path, device)
        self.pose_dl_model = PoseEstimationModel(pose_dl_model_path, device)
        
        if use_person_detection:
            try:
                self.yolo_detector = YOLOPersonDetector(yolo_model_path)
            except Exception as e:
                logger.warning(f"Could not load YOLO detector: {e}")
                self.use_person_detection = False
                self.yolo_detector = None
        else:
            self.yolo_detector = None
        
        logger.info("Models loaded successfully")
    
    def detect_view(self, image: np.ndarray) -> View:
        """
        Detect view type from image.
        
        Args:
            image: Input image
            
        Returns:
            Detected View (DL or FO)
        """
        prediction = self.view_model.predict(image)
        logger.info(f"Detected view: {prediction.view} (confidence: {prediction.confidence:.2f})")
        return prediction.view
    
    def detect_person(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect person bounding box.
        
        Args:
            image: Input image
            
        Returns:
            Bounding box [x1, y1, x2, y2, confidence] or None
        """
        if not self.use_person_detection or self.yolo_detector is None:
            return None
        
        bbox = self.yolo_detector.select_best_person(image)
        return bbox
    
    def predict_pose(
        self,
        image: np.ndarray,
        view: Optional[View] = None,
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict pose from image.
        
        Args:
            image: Input image (H, W, C)
            view: Optional view type (if None, will be detected)
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (keypoints, confidences)
            keypoints: (num_joints, 2) array
            confidences: (num_joints,) array
        """
        # Detect view if not provided
        if view is None:
            view = self.detect_view(image)
        
        # Detect person if bbox not provided
        if bbox is None and self.use_person_detection:
            detected_bbox = self.detect_person(image)
            if detected_bbox is not None:
                bbox = detected_bbox[:4]  # Extract [x1, y1, x2, y2]
        
        # Expand bounding box to include full swing
        if bbox is not None:
            h, w = image.shape[:2]
            expanded_bbox = expand_bounding_box(bbox, view, h, w)
        else:
            expanded_bbox = None
        
        # Select appropriate model
        if view == View.FO:
            model = self.pose_fo_model
        else:
            model = self.pose_dl_model
        
        # Predict pose
        keypoints, confidences = model.predict(image, expanded_bbox)
        
        return keypoints, confidences
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        frame_skip: int = 1,
        detect_view_per_frame: bool = False,
        enable_club_tracking: bool = True,
        enable_interpolation: bool = True,
        enable_temporal_tracking: bool = True,
        enable_one_euro_filter: bool = True,
        enable_pattern_recognition: bool = True,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.7,
        one_euro_d_cutoff: float = 1.0,
        temporal_use_kalman: bool = True,
        temporal_use_optical_flow: bool = True,
        temporal_use_biomechanical: bool = True,
        temporal_max_velocity: float = 200.0
    ) -> Dict:
        """
        Process entire video and predict poses.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            frame_skip: Process every Nth frame
            detect_view_per_frame: Whether to detect view for each frame
            enable_club_tracking: Whether to enable club tracking
            enable_interpolation: Whether to enable joint interpolation
            enable_temporal_tracking: Whether to enable temporal tracking and sequence consistency
            enable_one_euro_filter: Whether to apply One-Euro Filter for trajectory smoothing
            enable_pattern_recognition: Whether to apply pattern recognition for lighting/background adaptation
            one_euro_min_cutoff: Minimum cutoff frequency for One-Euro Filter (Hz)
            one_euro_beta: Speed coefficient for One-Euro Filter
            one_euro_d_cutoff: Derivative cutoff frequency for One-Euro Filter (Hz)
            temporal_use_kalman: Enable Kalman filtering in temporal tracking
            temporal_use_optical_flow: Enable optical flow tracking
            temporal_use_biomechanical: Enable biomechanical constraints
            temporal_max_velocity: Maximum allowed keypoint velocity (pixels/second)
            
        Returns:
            Dictionary containing:
            - 'keypoints': List of keypoint arrays per frame
            - 'confidences': List of confidence arrays per frame
            - 'view': Detected view
            - 'fps': Video FPS
            - 'num_frames': Number of frames processed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Detect view from first frame
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
        
        view = self.detect_view(first_frame)
        logger.info(f"Processing video with view: {view}")
        
        # Initialize pattern recognition if enabled
        adaptive_preprocessor = None
        lighting_info = None
        background_info = None
        if enable_pattern_recognition:
            try:
                from .pattern_recognition import AdaptivePreprocessor, analyze_video_conditions
                adaptive_preprocessor = AdaptivePreprocessor()
                logger.info("Pattern recognition enabled - will analyze lighting and background")
            except Exception as e:
                logger.warning(f"Could not initialize pattern recognition: {e}")
                enable_pattern_recognition = False
        
        # Reset video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        all_keypoints = []
        all_confidences = []
        frame_indices = []
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = first_frame.shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Pre-analyze video conditions if pattern recognition is enabled
        if enable_pattern_recognition and adaptive_preprocessor:
            try:
                logger.info("Pre-analyzing video for lighting and background conditions...")
                # Sample frames for analysis
                cap_analysis = cv2.VideoCapture(video_path)
                sample_frames = []
                sample_count = min(30, total_frames // max(1, frame_skip))
                sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
                
                for idx in sample_indices:
                    cap_analysis.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap_analysis.read()
                    if ret:
                        sample_frames.append(frame)
                cap_analysis.release()
                
                if sample_frames:
                    from .pattern_recognition import analyze_video_conditions
                    lighting_info, background_info = analyze_video_conditions(sample_frames)
                    logger.info(f"Video analysis complete - Lighting: {lighting_info['condition']}, "
                              f"Background: {background_info['type']}")
            except Exception as e:
                logger.warning(f"Video pre-analysis failed: {e}. Will analyze per-frame.")
                lighting_info = None
                background_info = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue
            
            # Apply adaptive preprocessing if pattern recognition is enabled
            processed_frame = frame
            if enable_pattern_recognition and adaptive_preprocessor:
                try:
                    # Get person bbox if available
                    person_bbox = None
                    if self.use_person_detection:
                        detected_bbox = self.detect_person(frame)
                        if detected_bbox is not None:
                            person_bbox = detected_bbox[:4]
                    
                    processed_frame = adaptive_preprocessor.preprocess_frame(
                        frame,
                        lighting_info=lighting_info,
                        background_info=background_info,
                        person_bbox=person_bbox
                    )
                except Exception as e:
                    logger.warning(f"Adaptive preprocessing failed for frame {frame_idx}: {e}")
                    processed_frame = frame
            
            # Detect view per frame if requested
            if detect_view_per_frame:
                view = self.detect_view(processed_frame)
            
            # Predict pose
            try:
                keypoints, confidences = self.predict_pose(processed_frame, view)
                
                # Filter keypoints by confidence threshold during detection
                # Club head (index 18) uses strict threshold (0.8), others use default (0.3)
                CLUB_HEAD_IDX = 18
                CLUB_HEAD_THRESHOLD = 0.8
                DEFAULT_THRESHOLD = 0.3
                
                for j in range(len(keypoints)):
                    threshold = CLUB_HEAD_THRESHOLD if j == CLUB_HEAD_IDX else DEFAULT_THRESHOLD
                    if confidences[j] < threshold:
                        # Mark as invalid by setting coordinates to negative
                        keypoints[j] = np.array([-1, -1], dtype=np.float32)
                        confidences[j] = 0.0
                
                all_keypoints.append(keypoints)
                all_confidences.append(confidences)
                frame_indices.append(frame_idx)
                
                # Draw pose on frame if writing video
                if writer is not None:
                    frame_with_pose = self.draw_pose(frame, keypoints, confidences)
                    writer.write(frame_with_pose)
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {e}")
                # Add empty keypoints
                num_joints = 19  # Default number of joints
                all_keypoints.append(np.zeros((num_joints, 2)))
                all_confidences.append(np.zeros(num_joints))
                frame_indices.append(frame_idx)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        if writer is not None:
            writer.release()
        
        # Convert to numpy arrays
        keypoints_array = np.array(all_keypoints)  # (num_frames, num_joints, 2)
        confidences_array = np.array(all_confidences)  # (num_frames, num_joints)
        frame_indices_array = np.array(frame_indices)
        
        # Apply club tracking if enabled
        if enable_club_tracking:
            try:
                print("Applying club tracking...")
                logger.info("Applying club tracking...")
                from .club_tracking import (
                    find_club_cone_vectors,
                    detect_club_in_video,
                    stitch_club_joints,
                    filter_hand_dist
                )
                
                # Load all frames for club detection
                print(f"Loading {len(frame_indices_array)} frames for club detection...")
                cap = cv2.VideoCapture(video_path)
                all_frames = []
                for idx in frame_indices_array:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        all_frames.append(frame)
                    else:
                        all_frames.append(None)
                cap.release()
                
                # Filter out None frames and get valid indices
                valid_frames = [f for f in all_frames if f is not None]
                print(f"Loaded {len(valid_frames)} valid frames")
                
                if len(valid_frames) > 0:
                    valid_frames_array = np.array(valid_frames)
                    
                    # Extract club joints (last 4 joints: handle1, handle2, handle3, club_head)
                    club_joints = keypoints_array[:, -4:, :]
                    
                    # Find club cone vectors
                    print("Calculating club cone vectors...")
                    line_clockwise, line_counterclockwise = find_club_cone_vectors(
                        keypoints_array  # Pass all joints to calculate height
                    )
                    
                    # Detect club in video
                    print("Detecting club in video (this may take a while)...")
                    trust_points, endpoint_trust = detect_club_in_video(
                        valid_frames_array,
                        line_clockwise,
                        line_counterclockwise,
                        keypoints_array
                    )
                    
                    # Count how many trust points were found
                    valid_trust_points = sum(1 for tp in trust_points if tp is not None)
                    print(f"Found {valid_trust_points}/{len(trust_points)} valid club detections")
                    
                    # Stitch club joints
                    print("Stitching club joints...")
                    keypoints_array, confidences_array = stitch_club_joints(
                        keypoints_array,
                        confidences_array,
                        trust_points,
                        endpoint_trust
                    )
                    
                    # Filter hand distance
                    print("Filtering hand distance...")
                    hosel_joints = keypoints_array[:, -1, :]
                    keypoints_array = filter_hand_dist(
                        keypoints_array,
                        hosel_joints,
                        confidences_array,
                        pose_conf_threshold=0.8,
                        framerate=fps
                    )
                    
                    print("Club tracking completed successfully")
                    logger.info("Club tracking completed")
                else:
                    print("Warning: No valid frames loaded for club tracking")
            except Exception as e:
                import traceback
                print(f"Club tracking failed: {e}")
                print(traceback.format_exc())
                logger.warning(f"Club tracking failed: {e}. Continuing without club tracking.", exc_info=True)
        
        # Apply joint interpolation if enabled
        if enable_interpolation:
            try:
                print("Applying joint interpolation...")
                logger.info("Applying joint interpolation...")
                from .joint_interpolation import interpolate_full
                
                # Interpolate to fill in all frames (not just sampled ones)
                print(f"Interpolating {len(keypoints_array)} frames to fill gaps...")
                interpolated_joints, start_end_sequence, error = interpolate_full(
                    keypoints_array,
                    frame_indices_array,
                    confidences_array,
                    s_val=0.0,
                    conf_threshold=0.3
                )
                
                if error is None:
                    # Create full frame array
                    total_video_frames = int(frame_indices_array.max()) + 1
                    print(f"Interpolating to {total_video_frames} total frames")
                    full_keypoints = np.full((total_video_frames, keypoints_array.shape[1], 2), -1.0)
                    full_confidences = np.zeros((total_video_frames, keypoints_array.shape[1]))
                    # Fill in interpolated values
                    full_keypoints[:interpolated_joints.shape[0]] = interpolated_joints
                    
                    # Update confidences for interpolated frames
                    for j in range(keypoints_array.shape[1]):
                        start_frame = start_end_sequence[j, 0]
                        end_frame = start_end_sequence[j, 1]
                        if start_frame >= 0 and end_frame >= 0:
                            # Set confidence to 0.5 for interpolated frames
                            full_confidences[start_frame:end_frame+1, j] = 0.5
                    
                    # Keep original confidences for detected frames
                    for i, frame_idx in enumerate(frame_indices_array):
                        full_confidences[frame_idx] = confidences_array[i]
                    
                    keypoints_array = full_keypoints
                    confidences_array = full_confidences
                    frame_indices_array = np.arange(total_video_frames)
                    
                    print(f"Interpolation completed: {total_video_frames} frames")
                    logger.info(f"Interpolation completed: {total_video_frames} frames")
                else:
                    print(f"Interpolation failed: {error}")
                    logger.warning(f"Interpolation failed: {error}. Using original keypoints.")
            except Exception as e:
                import traceback
                print(f"Joint interpolation failed: {e}")
                print(traceback.format_exc())
                logger.warning(f"Joint interpolation failed: {e}. Continuing without interpolation.", exc_info=True)
        
        # Apply temporal tracking and sequence consistency if enabled
        if enable_temporal_tracking:
            try:
                print("Applying temporal tracking and sequence consistency...")
                logger.info("Applying temporal tracking...")
                from .temporal_tracking import apply_temporal_tracking
                
                # Load frames for optical flow if needed
                frames_for_tracking = None
                if temporal_use_optical_flow:
                    print("Loading frames for optical flow tracking...")
                    cap = cv2.VideoCapture(video_path)
                    frames_for_tracking = []
                    for idx in frame_indices_array:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames_for_tracking.append(frame)
                        else:
                            frames_for_tracking.append(None)
                    cap.release()
                    print(f"Loaded {len([f for f in frames_for_tracking if f is not None])} frames for tracking")
                
                keypoints_array, confidences_array = apply_temporal_tracking(
                    keypoints_array,
                    confidences_array,
                    frames=frames_for_tracking,
                    fps=fps,
                    use_kalman=temporal_use_kalman,
                    use_optical_flow=temporal_use_optical_flow,
                    use_biomechanical=temporal_use_biomechanical,
                    max_velocity=temporal_max_velocity
                )
                
                print("Temporal tracking applied successfully")
                logger.info("Temporal tracking completed")
            except Exception as e:
                import traceback
                print(f"Temporal tracking failed: {e}")
                print(traceback.format_exc())
                logger.warning(f"Temporal tracking failed: {e}. Continuing without temporal tracking.", exc_info=True)
        
        # Apply One-Euro Filter for trajectory smoothing if enabled
        if enable_one_euro_filter:
            try:
                print("Applying One-Euro Filter for trajectory smoothing...")
                logger.info("Applying One-Euro Filter...")
                from .one_euro_filter import apply_one_euro_filter
                
                keypoints_array, confidences_array = apply_one_euro_filter(
                    keypoints_array,
                    confidences_array,
                    fps=fps,
                    min_cutoff=one_euro_min_cutoff,
                    beta=one_euro_beta,
                    d_cutoff=one_euro_d_cutoff,
                    conf_threshold=0.3,
                    invalid_value=-1.0
                )
                
                print("One-Euro Filter applied successfully")
                logger.info("One-Euro Filter completed")
            except Exception as e:
                import traceback
                print(f"One-Euro Filter failed: {e}")
                print(traceback.format_exc())
                logger.warning(f"One-Euro Filter failed: {e}. Continuing without filtering.", exc_info=True)
        
        return {
            'keypoints': keypoints_array,  # (num_frames, num_joints, 2)
            'confidences': confidences_array,  # (num_frames, num_joints)
            'view': view,
            'fps': fps,
            'num_frames': len(keypoints_array),
            'frame_indices': frame_indices_array
        }
    
    @staticmethod
    def draw_pose(
        image: np.ndarray,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        threshold: float = 0.3
    ) -> np.ndarray:
        """
        Draw pose skeleton on image.
        
        Args:
            image: Input image
            keypoints: Keypoints array (num_joints, 2)
            confidences: Confidence array (num_joints,)
            threshold: Minimum confidence to draw
            
        Returns:
            Image with pose drawn
        """
        img = image.copy()
        
        # Skeleton connections for golf pose (19 keypoints)
        skeleton = [
            (0, 1), (0, 2),  # head to eyes
            (3, 4), (4, 10), (3, 9), (9, 10),  # shoulders and hips
            (3, 5), (5, 7), (4, 6), (6, 8),  # arms
            (9, 11), (11, 13), (10, 12), (12, 14),  # legs
            (15, 16), (16, 17), (17, 18)  # club
        ]
        
        # Draw connections
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                confidences[start_idx] > threshold and confidences[end_idx] > threshold):
                pt1 = tuple(keypoints[start_idx].astype(int))
                pt2 = tuple(keypoints[end_idx].astype(int))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
            if conf > threshold:
                pt = tuple(kp.astype(int))
                cv2.circle(img, pt, 5, (0, 0, 255), -1)
        
        return img


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 2D pose prediction on a video."
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-video", help="Optional path to save pose video (default: res/{video_name}_pose.mp4)")
    parser.add_argument("--output-npz", help="Optional path to save results (.npz) (default: res/{video_name}_results.npz)")
    parser.add_argument("--output-overlay", help="Optional path to save overlay video (default: res/{video_name}_overlay.mp4)")
    parser.add_argument("--overlay-json", help="Optional path to pose_overlays.json (used for overlay output)")
    parser.add_argument("--overlay-threshold", type=float, default=0.3,
                        help="Keypoint confidence threshold for overlay visualization")
    parser.add_argument("--left-handed", action="store_true",
                        help="Set RightHanded=False in overlay metadata")
    parser.add_argument("--right-handed", action="store_true",
                        help="Set RightHanded=True in overlay metadata (default)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--no-person-det", action="store_true", help="Disable YOLO person detection")
    parser.add_argument("--no-club-tracking", action="store_true", help="Disable club tracking")
    parser.add_argument("--no-interp", action="store_true", help="Disable joint interpolation")
    parser.add_argument("--no-temporal-tracking", action="store_true", help="Disable temporal tracking and sequence consistency")
    parser.add_argument("--no-one-euro", action="store_true", help="Disable One-Euro Filter for trajectory smoothing")
    parser.add_argument("--no-pattern-recognition", action="store_true", help="Disable pattern recognition for lighting/background adaptation")
    parser.add_argument("--one-euro-min-cutoff", type=float, default=1.0, help="One-Euro Filter minimum cutoff frequency (Hz)")
    parser.add_argument("--one-euro-beta", type=float, default=0.7, help="One-Euro Filter speed coefficient")
    parser.add_argument("--one-euro-d-cutoff", type=float, default=1.0, help="One-Euro Filter derivative cutoff frequency (Hz)")
    parser.add_argument("--temporal-no-kalman", action="store_true", help="Disable Kalman filtering in temporal tracking")
    parser.add_argument("--temporal-no-optical-flow", action="store_true", help="Disable optical flow tracking")
    parser.add_argument("--temporal-no-biomechanical", action="store_true", help="Disable biomechanical constraints")
    parser.add_argument("--temporal-max-velocity", type=float, default=200.0, help="Maximum allowed keypoint velocity (pixels/second)")
    parser.add_argument("--view-model", default="models/classify_view.onnx")
    parser.add_argument("--pose-fo", default="models/golfer_fo2.onnx")
    parser.add_argument("--pose-dl", default="models/golfer_dl.onnx")
    parser.add_argument("--yolo", default="models/yolo11n.pt")
    return parser


def _save_results_npz(results: Dict, output_path: str) -> None:
    output_path = str(Path(output_path))
    np.savez_compressed(
        output_path,
        keypoints=results["keypoints"],
        confidences=results["confidences"],
        view=str(results["view"]),
        fps=results["fps"],
        num_frames=results["num_frames"],
        frame_indices=results["frame_indices"],
    )


def _write_overlay_json(path_str: str, view: str, fps: float, right_handed: bool) -> None:
    overlay_dict = {
        "Metadata": {
            "View": view,
            "RightHanded": right_handed,
            "Framerate": float(fps),
        },
        "visualaid": {
            "SwingPhases": {}
        }
    }
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(path_str, "w", encoding="utf-8") as f:
        json.dump(overlay_dict, f, indent=2)


def _save_overlay_inputs(results: Dict, output_dir: Path) -> Tuple[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoints_path = output_dir / "keypoints.npy"
    confidences_path = output_dir / "confidences.npy"
    np.save(str(keypoints_path), results["keypoints"])
    np.save(str(confidences_path), results["confidences"])
    return str(keypoints_path), str(confidences_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Get project root directory (where main.py is located)
    project_root = Path(__file__).resolve().parent
    res_dir = project_root / "res"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate default output filenames based on input video name
    input_video_path = Path(args.video)
    video_stem = input_video_path.stem  # filename without extension
    
    # Set default output paths if not provided
    if args.output_video is None:
        args.output_video = str(res_dir / f"{video_stem}_pose.mp4")
    
    if args.output_npz is None:
        args.output_npz = str(res_dir / f"{video_stem}_results.npz")
    
    if args.output_overlay is None:
        args.output_overlay = str(res_dir / f"{video_stem}_overlay.mp4")

    predictor = PosePredictor(
        view_model_path=args.view_model,
        pose_fo_model_path=args.pose_fo,
        pose_dl_model_path=args.pose_dl,
        yolo_model_path=args.yolo,
        device=args.device,
        use_person_detection=not args.no_person_det,
    )

    results = predictor.process_video(
        video_path=args.video,
        output_path=args.output_video,
        frame_skip=args.frame_skip,
        enable_club_tracking=not args.no_club_tracking,
        enable_interpolation=not args.no_interp,
        enable_temporal_tracking=not args.no_temporal_tracking,
        enable_one_euro_filter=not args.no_one_euro,
        enable_pattern_recognition=not args.no_pattern_recognition,
        one_euro_min_cutoff=args.one_euro_min_cutoff,
        one_euro_beta=args.one_euro_beta,
        one_euro_d_cutoff=args.one_euro_d_cutoff,
        temporal_use_kalman=not args.temporal_no_kalman,
        temporal_use_optical_flow=not args.temporal_no_optical_flow,
        temporal_use_biomechanical=not args.temporal_no_biomechanical,
        temporal_max_velocity=args.temporal_max_velocity,
    )

    # Save results to .npz file
    _save_results_npz(results, args.output_npz)
    logger.info(f"Saved results to {args.output_npz}")

    # Create overlay video
    from .overlay_visualization import create_overlay_video

    output_overlay_path = Path(args.output_overlay)
    overlay_dir = output_overlay_path.parent
    keypoints_path, confidences_path = _save_overlay_inputs(results, overlay_dir)

    overlay_json_path = args.overlay_json
    if not overlay_json_path:
        overlay_json_path = str(overlay_dir / "pose_overlays.json")

    if args.left_handed:
        right_handed = False
    elif args.right_handed:
        right_handed = True
    else:
        right_handed = True

    if not Path(overlay_json_path).is_file():
        _write_overlay_json(overlay_json_path, str(results["view"]), results["fps"], right_handed)

    create_overlay_video(
        video_path=args.video,
        keypoints_path=keypoints_path,
        confidences_path=confidences_path,
        overlay_json_path=overlay_json_path,
        output_path=str(output_overlay_path),
        draw_skeleton_overlay=True,
        draw_phases=True,
        draw_info=True,
        keypoint_threshold=args.overlay_threshold,
    )
    
    logger.info(f"All outputs saved to: {res_dir}")
    logger.info(f"  - Pose video: {args.output_video}")
    logger.info(f"  - Results: {args.output_npz}")
    logger.info(f"  - Overlay video: {args.output_overlay}")
