"""
Models package for Traffic Heatmap project.
Contains YOLO model configurations and utilities.
"""

from models.scripts.yolo_loader import YOLOPersonDetector, load_yolo_model

__all__ = ["YOLOPersonDetector", "load_yolo_model"]
