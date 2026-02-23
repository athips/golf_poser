"""
Pattern recognition and adaptation for different lighting conditions and backgrounds.

This module provides utilities to:
1. Detect and adapt to different lighting conditions
2. Handle multiple backgrounds
3. Improve pose estimation robustness across varying environments
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class LightingCondition(Enum):
    """Enumeration of lighting conditions."""
    BRIGHT = "bright"
    NORMAL = "normal"
    DARK = "dark"
    VARIABLE = "variable"  # Mixed lighting conditions


class BackgroundType(Enum):
    """Enumeration of background types."""
    UNIFORM = "uniform"
    COMPLEX = "complex"
    CLUTTERED = "cluttered"


class LightingAnalyzer:
    """
    Analyzes lighting conditions in video frames.
    """
    
    def __init__(self):
        """Initialize lighting analyzer."""
        self.bright_threshold = 200  # Mean brightness threshold for bright
        self.dark_threshold = 80     # Mean brightness threshold for dark
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze lighting condition of a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Dictionary with lighting analysis:
            - 'condition': LightingCondition enum
            - 'mean_brightness': Mean brightness value
            - 'std_brightness': Standard deviation of brightness
            - 'contrast': Contrast measure
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Calculate contrast (using standard deviation as proxy)
        contrast = std_brightness
        
        # Determine lighting condition
        if mean_brightness >= self.bright_threshold:
            condition = LightingCondition.BRIGHT
        elif mean_brightness <= self.dark_threshold:
            condition = LightingCondition.DARK
        else:
            condition = LightingCondition.NORMAL
        
        return {
            'condition': condition,
            'mean_brightness': float(mean_brightness),
            'std_brightness': float(std_brightness),
            'contrast': float(contrast)
        }
    
    def analyze_video(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze lighting conditions across multiple frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Dictionary with overall lighting analysis
        """
        if not frames:
            return {
                'condition': LightingCondition.NORMAL,
                'mean_brightness': 128.0,
                'std_brightness': 50.0,
                'contrast': 50.0,
                'is_variable': False
            }
        
        analyses = [self.analyze_frame(frame) for frame in frames if frame is not None]
        
        if not analyses:
            return {
                'condition': LightingCondition.NORMAL,
                'mean_brightness': 128.0,
                'std_brightness': 50.0,
                'contrast': 50.0,
                'is_variable': False
            }
        
        mean_brightnesses = [a['mean_brightness'] for a in analyses]
        conditions = [a['condition'] for a in analyses]
        
        # Check if lighting is variable
        unique_conditions = set(conditions)
        is_variable = len(unique_conditions) > 1
        
        overall_condition = LightingCondition.VARIABLE if is_variable else conditions[0]
        
        return {
            'condition': overall_condition,
            'mean_brightness': float(np.mean(mean_brightnesses)),
            'std_brightness': float(np.std(mean_brightnesses)),
            'contrast': float(np.mean([a['contrast'] for a in analyses])),
            'is_variable': is_variable
        }


class BackgroundAnalyzer:
    """
    Analyzes background characteristics in video frames.
    """
    
    def __init__(self):
        """Initialize background analyzer."""
        self.complexity_threshold = 30.0  # Edge density threshold
    
    def analyze_frame(self, frame: np.ndarray, person_bbox: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze background characteristics of a frame.
        
        Args:
            frame: Input frame (BGR format)
            person_bbox: Optional bounding box [x1, y1, x2, y2] to exclude person region
            
        Returns:
            Dictionary with background analysis:
            - 'type': BackgroundType enum
            - 'complexity': Background complexity score
            - 'edge_density': Edge density measure
        """
        # Create mask to exclude person region if bbox provided
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        if person_bbox is not None:
            x1, y1, x2, y2 = map(int, person_bbox[:4])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # Expand mask slightly to avoid edge effects
            expand = 20
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(w, x2 + expand)
            y2 = min(h, y2 + expand)
            mask[y1:y2, x1:x2] = 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply mask
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Calculate edge density (complexity measure)
        edges = cv2.Canny(masked_gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = np.sum(mask > 0)
        edge_density = (edge_pixels / max(total_pixels, 1)) * 100.0
        
        # Calculate variance (uniformity measure)
        variance = np.var(masked_gray[mask > 0])
        
        # Determine background type
        if edge_density < 5.0:
            bg_type = BackgroundType.UNIFORM
        elif edge_density < self.complexity_threshold:
            bg_type = BackgroundType.COMPLEX
        else:
            bg_type = BackgroundType.CLUTTERED
        
        return {
            'type': bg_type,
            'complexity': float(edge_density),
            'edge_density': float(edge_density),
            'variance': float(variance)
        }


class AdaptivePreprocessor:
    """
    Adapts preprocessing based on lighting and background conditions.
    """
    
    def __init__(self):
        """Initialize adaptive preprocessor."""
        self.lighting_analyzer = LightingAnalyzer()
        self.background_analyzer = BackgroundAnalyzer()
    
    def preprocess_frame(
        self,
        frame: np.ndarray,
        lighting_info: Optional[Dict] = None,
        background_info: Optional[Dict] = None,
        person_bbox: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Preprocess frame with adaptive enhancements based on conditions.
        
        Args:
            frame: Input frame (BGR format)
            lighting_info: Optional lighting analysis results
            background_info: Optional background analysis results
            person_bbox: Optional person bounding box
            
        Returns:
            Preprocessed frame
        """
        processed = frame.copy()
        
        # Analyze if not provided
        if lighting_info is None:
            lighting_info = self.lighting_analyzer.analyze_frame(frame)
        
        if background_info is None:
            background_info = self.background_analyzer.analyze_frame(frame, person_bbox)
        
        # Apply lighting adaptation
        condition = lighting_info['condition']
        mean_brightness = lighting_info['mean_brightness']
        
        if condition == LightingCondition.DARK:
            # Enhance dark images
            processed = self._enhance_dark_image(processed)
        elif condition == LightingCondition.BRIGHT:
            # Reduce overexposure
            processed = self._reduce_brightness(processed)
        elif condition == LightingCondition.VARIABLE:
            # Apply adaptive histogram equalization
            processed = self._apply_adaptive_equalization(processed)
        
        # Apply background-specific enhancements
        bg_type = background_info['type']
        if bg_type == BackgroundType.CLUTTERED:
            # Enhance contrast for cluttered backgrounds
            processed = self._enhance_contrast(processed)
        
        return processed
    
    def _enhance_dark_image(self, frame: np.ndarray) -> np.ndarray:
        """Enhance dark images using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _reduce_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Reduce brightness of overexposed images."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Reduce value channel
        v = cv2.multiply(v, 0.85)
        
        # Merge and convert back
        hsv_reduced = cv2.merge([h, s, v])
        reduced = cv2.cvtColor(hsv_reduced, cv2.COLOR_HSV2BGR)
        
        return reduced
    
    def _apply_adaptive_equalization(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization for variable lighting."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_equalized = clahe.apply(l)
        
        lab_equalized = cv2.merge([l_equalized, a, b])
        equalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
        
        return equalized
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Enhance contrast for better visibility in cluttered backgrounds."""
        # Apply slight contrast enhancement
        alpha = 1.1  # Contrast control (1.0 = no change)
        beta = 5     # Brightness control
        
        enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        return enhanced


def analyze_video_conditions(
    frames: List[np.ndarray],
    person_bboxes: Optional[List[np.ndarray]] = None
) -> Tuple[Dict, Dict]:
    """
    Analyze lighting and background conditions for a video.
    
    Args:
        frames: List of video frames
        person_bboxes: Optional list of person bounding boxes per frame
        
    Returns:
        Tuple of (lighting_info, background_info)
    """
    lighting_analyzer = LightingAnalyzer()
    background_analyzer = BackgroundAnalyzer()
    
    # Sample frames for analysis (use every Nth frame to speed up)
    sample_rate = max(1, len(frames) // 30)  # Sample ~30 frames
    sample_frames = frames[::sample_rate]
    sample_bboxes = person_bboxes[::sample_rate] if person_bboxes else [None] * len(sample_frames)
    
    # Analyze lighting
    lighting_info = lighting_analyzer.analyze_video(sample_frames)
    
    # Analyze backgrounds
    background_analyses = []
    for frame, bbox in zip(sample_frames, sample_bboxes):
        if frame is not None:
            bg_info = background_analyzer.analyze_frame(frame, bbox)
            background_analyses.append(bg_info)
    
    if background_analyses:
        # Aggregate background info
        bg_types = [a['type'] for a in background_analyses]
        bg_complexities = [a['complexity'] for a in background_analyses]
        
        # Most common background type
        from collections import Counter
        most_common_type = Counter(bg_types).most_common(1)[0][0]
        
        background_info = {
            'type': most_common_type,
            'mean_complexity': float(np.mean(bg_complexities)),
            'is_variable': len(set(bg_types)) > 1
        }
    else:
        background_info = {
            'type': BackgroundType.COMPLEX,
            'mean_complexity': 20.0,
            'is_variable': False
        }
    
    logger.info(f"Video analysis - Lighting: {lighting_info['condition']}, "
                f"Background: {background_info['type']}")
    
    return lighting_info, background_info
