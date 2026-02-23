"""
Data classes and enums for pose estimation
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


class View(Enum):
    """Enum representing the point of view in a video."""
    DL = 0  # Down the Line
    FO = 1  # Face On
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"View.{self.name}"


class Dexterity(Enum):
    """Enum representing the dexterity of the golfer."""
    Right = 0
    Left = 1


class RunMode(Enum):
    """Enum representing different run modes."""
    ProductionMode = 0
    DebugMode = 1
    FullDebugMode = 2
    SummaryMode = 3


@dataclass
class ViewPrediction:
    """View prediction result."""
    view: View
    confidence: float


@dataclass
class BoundingBox:
    """Bounding box in xyxy format."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x1, y1, x2, y2]."""
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, confidence: float = 1.0):
        """Create from numpy array [x1, y1, x2, y2]."""
        return cls(arr[0], arr[1], arr[2], arr[3], confidence)







