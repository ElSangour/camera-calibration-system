"""
Model scripts package.
Contains model loading, configuration, and utilities.
"""

from models.scripts.yolo_loader import YOLOPersonDetector, load_yolo_model

__all__ = ["YOLOPersonDetector", "load_yolo_model"]
