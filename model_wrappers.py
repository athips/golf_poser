"""
Model wrappers for ONNX and YOLO models
"""
import os
# Fix OpenMP duplicate library error - must be set before any imports
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple, List
import logging

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not available. YOLO person detection will be unavailable.")

from .data_classes import View, ViewPrediction

logger = logging.getLogger(__name__)


class ONNXModelWrapper:
    """Base wrapper for ONNX models."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize ONNX model wrapper.
        
        Args:
            model_path: Path to ONNX model file
            device: 'cuda' or 'cpu'
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model_path = model_path
        self.device = device
        
        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set providers
        providers = []
        available_providers = ort.get_available_providers()
        
        if device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            if device == 'cuda':
                logger.warning(f"CUDA requested but not available. Using CPU instead.")
        
        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"Loaded ONNX model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference.
        
        Args:
            x: Input array
            
        Returns:
            Output array
        """
        outputs = self.session.run([self.output_name], {self.input_name: x})
        return outputs[0]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ViewDetectionModel(ONNXModelWrapper):
    """Wrapper for view detection model (classify_view.onnx)."""
    
    def predict(self, image: np.ndarray) -> ViewPrediction:
        """
        Predict view type from image.
        
        Args:
            image: Input image (C, H, W) or (H, W, C)
            
        Returns:
            ViewPrediction with view type and confidence
        """
        # Preprocess image
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # (C, H, W)
                image = image.transpose(1, 2, 0)  # (H, W, C)
        
        # Resize to model input size (typically 224x224)
        image = cv2.resize(image, (224, 224))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Convert to (1, C, H, W)
        image = image.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference
        output = self.forward(image)
        
        # Get predictions (assuming output is [batch, num_classes])
        # Implement softmax manually (numpy doesn't have softmax)
        if output.shape[1] > 1:
            # Softmax: exp(x) / sum(exp(x))
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))  # Subtract max for numerical stability
            probs = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        else:
            probs = output
        
        class_idx = np.argmax(probs[0])
        confidence = float(probs[0][class_idx])
        
        # Map class index to View (0=DL, 1=FO typically)
        view = View.DL if class_idx == 0 else View.FO
        
        return ViewPrediction(view=view, confidence=confidence)


class PoseEstimationModel(ONNXModelWrapper):
    """Wrapper for pose estimation models (golfer_fo2.onnx, golfer_dl.onnx)."""
    
    def __init__(self, model_path: str, device: str = 'cuda', input_size: Tuple[int, int] = (384, 288)):
        """
        Initialize pose estimation model.
        
        Args:
            model_path: Path to ONNX model
            device: 'cuda' or 'cpu'
            input_size: Model input size (width, height)
        """
        super().__init__(model_path, device)
        self.input_size = input_size
    
    def preprocess(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for pose estimation.
        
        Args:
            image: Input image (H, W, C) or (C, H, W)
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (preprocessed_image, transform_info)
            transform_info contains: (crop_offset, original_size, scale)
        """
        # Handle different input formats
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # (C, H, W)
                image = image.transpose(1, 2, 0)  # (H, W, C)
        
        original_h, original_w = image.shape[:2]
        
        # Crop to bounding box if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_w, x2)
            y2 = min(original_h, y2)
            image = image[y1:y2, x1:x2]
            crop_offset = np.array([x1, y1])
        else:
            crop_offset = np.array([0, 0])
        
        # Resize while maintaining aspect ratio
        h, w = image.shape[:2]
        target_w, target_h = self.input_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        image_padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        image_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image_resized
        
        # Normalize
        image_norm = image_padded.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_norm - mean) / std
        
        # Convert to (1, C, H, W)
        image_final = image_norm.transpose(2, 0, 1)[np.newaxis, ...]
        
        transform_info = {
            'crop_offset': crop_offset,
            'original_size': np.array([original_w, original_h]),
            'crop_size': np.array([w, h]),  # Size of crop before resizing
            'scale': scale,
            'pad': (pad_w, pad_h)
        }
        
        return image_final.astype(np.float32), transform_info
    
    def postprocess(self, heatmaps: np.ndarray, transform_info: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess heatmaps to get keypoints using decode_heatmaps approach.
        
        Args:
            heatmaps: Model output heatmaps (batch, num_joints, H, W)
            transform_info: Transform information from preprocessing
            
        Returns:
            Tuple of (keypoints, confidences)
            keypoints: (num_joints, 2) array of (x, y) coordinates in original image space
            confidences: (num_joints,) array of confidence scores
        """
        if heatmaps.ndim == 4:
            heatmaps = heatmaps[0]  # Remove batch dimension if present
        
        num_joints, heatmap_h, heatmap_w = heatmaps.shape
        input_w, input_h = self.input_size
        
        # Get original image size (before cropping)
        original_size = transform_info['original_size']  # [width, height]
        orig_w, orig_h = original_size[0], original_size[1]
        
        # Get crop information
        crop_offset = transform_info['crop_offset']  # [x, y] of crop top-left
        scale = transform_info['scale']  # Scale factor used when resizing crop
        pad_w, pad_h = transform_info['pad']  # Padding added to resized crop
        
        # Get crop size (size of the cropped region before resizing)
        if 'crop_size' in transform_info:
            crop_size = transform_info['crop_size']
            crop_w, crop_h = crop_size[0], crop_size[1]
        else:
            # Calculate crop size from scale and padding (backward compatibility)
            # The model input size represents the resized crop (with padding)
            # To get back to crop size: (input_size - 2*pad) / scale
            crop_w = (input_w - 2 * pad_w) / scale
            crop_h = (input_h - 2 * pad_h) / scale
        
        # Vectorized operations for all keypoints at once (like decode_heatmaps)
        heatmaps_flat = heatmaps.reshape(num_joints, -1)
        
        # Find max indices for all keypoints at once
        max_indices = np.argmax(heatmaps_flat, axis=1)
        
        # Convert flat indices to (y, x) coordinates in heatmap
        y_coords = max_indices // heatmap_w
        x_coords = max_indices % heatmap_w
        
        # Get scores (vectorized)
        scores = heatmaps_flat[np.arange(num_joints), max_indices]
        
        # Step 1: Convert from heatmap coordinates to model input coordinates
        # Heatmap is typically smaller than model input (e.g., 72x96 vs 384x288)
        model_x = (x_coords + 0.5) * (input_w / heatmap_w)
        model_y = (y_coords + 0.5) * (input_h / heatmap_h)
        
        # Step 2: Convert from model input coordinates to cropped image coordinates
        # Remove padding and apply inverse scale
        crop_x = (model_x - pad_w) / scale
        crop_y = (model_y - pad_h) / scale
        
        # Step 3: Convert from cropped image coordinates to original image coordinates
        # Add the crop offset (top-left corner of the crop in original image)
        orig_x = crop_x + crop_offset[0]
        orig_y = crop_y + crop_offset[1]
        
        # Stack into (num_joints, 2) array
        keypoints = np.stack([orig_x, orig_y], axis=1).astype(np.float32)
        confidences = scores.astype(np.float32)
        
        return keypoints, confidences
    
    def predict(self, image: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict pose from image.
        
        Args:
            image: Input image
            bbox: Optional bounding box
            
        Returns:
            Tuple of (keypoints, confidences)
        """
        # Preprocess
        preprocessed, transform_info = self.preprocess(image, bbox)
        
        # Run inference
        heatmaps = self.forward(preprocessed)
        
        # Postprocess
        keypoints, confidences = self.postprocess(heatmaps, transform_info)
        
        return keypoints, confidences


class YOLOPersonDetector:
    """Wrapper for YOLO person detection model."""
    
    def __init__(self, model_path: str = 'models/yolo11n.pt'):
        """
        Initialize YOLO person detector.
        
        Args:
            model_path: Path to YOLO model file
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required for YOLO person detection")
        
        if not os.path.exists(model_path):
            # Try to load default model (will download if needed)
            logger.warning(f"Model not found at {model_path}, trying default")
            try:
                self.model = YOLO('yolo11n.pt')
            except Exception as e:
                raise FileNotFoundError(f"Could not load YOLO model: {e}")
        else:
            self.model = YOLO(model_path)
        
        logger.info(f"Loaded YOLO model: {model_path}")
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[np.ndarray]:
        """
        Detect persons in image.
        
        Args:
            image: Input image (H, W, C)
            conf_threshold: Confidence threshold
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2, confidence] for each person
        """
        results = self.model(image, verbose=False, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detected class is 'person' (class ID 0)
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        detections.append(np.array([x1, y1, x2, y2, conf]))
        
        return detections
    
    def select_best_person(self, image: np.ndarray, conf_threshold: float = 0.25) -> Optional[np.ndarray]:
        """
        Select the best person detection (highest confidence, most central).
        
        Args:
            image: Input image
            conf_threshold: Confidence threshold
            
        Returns:
            Bounding box [x1, y1, x2, y2, confidence] or None
        """
        detections = self.detect(image, conf_threshold)
        
        if len(detections) == 0:
            return None
        
        if len(detections) == 1:
            return detections[0]
        
        # Select based on confidence and centrality
        h, w = image.shape[:2]
        best_idx = 0
        best_score = 0
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf = det
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Score: confidence * centrality
            centrality = 1.0 - abs(center_x - w/2) / (w/2) - abs(center_y - h/2) / (h/2)
            score = conf * (0.5 + 0.5 * centrality)
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        return detections[best_idx]

