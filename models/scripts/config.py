"""
Model Configuration Settings
Central configuration for all model-related settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===========================================
# Directory Paths
# ===========================================
MODELS_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = MODELS_DIR / "weights"
SCRIPTS_DIR = MODELS_DIR / "scripts"

# ===========================================
# YOLO Model Settings
# ===========================================
YOLO_CONFIG = {
    "model_name": os.getenv("YOLO_MODEL", "yolov8n.pt"),
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", 0.5)),
    "iou_threshold": float(os.getenv("IOU_THRESHOLD", 0.45)),
    "max_detections": int(os.getenv("MAX_DETECTIONS", 100)),
}

# ===========================================
# Available Models
# ===========================================
AVAILABLE_MODELS = {
    "nano": {
        "file": "yolov8n.pt",
        "description": "Fastest, least accurate (~6MB)",
        "speed": "⚡⚡⚡⚡⚡",
        "accuracy": "⭐⭐",
    },
    "small": {
        "file": "yolov8s.pt",
        "description": "Fast with good balance (~22MB)",
        "speed": "⚡⚡⚡⚡",
        "accuracy": "⭐⭐⭐",
    },
    "medium": {
        "file": "yolov8m.pt",
        "description": "Balanced speed/accuracy (~52MB)",
        "speed": "⚡⚡⚡",
        "accuracy": "⭐⭐⭐⭐",
    },
    "large": {
        "file": "yolov8l.pt",
        "description": "Accurate but slower (~87MB)",
        "speed": "⚡⚡",
        "accuracy": "⭐⭐⭐⭐⭐",
    },
    "xlarge": {
        "file": "yolov8x.pt",
        "description": "Most accurate, slowest (~137MB)",
        "speed": "⚡",
        "accuracy": "⭐⭐⭐⭐⭐",
    },
}

# ===========================================
# COCO Dataset Class IDs
# ===========================================
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    # ... (person is what we need for traffic heatmap)
}

PERSON_CLASS_ID = 0

# ===========================================
# Detection Settings
# ===========================================
DETECTION_CONFIG = {
    "target_classes": [PERSON_CLASS_ID],  # Only detect persons
    "min_box_area": 500,  # Minimum bounding box area to consider
    "max_box_area": None,  # Maximum bounding box area (None = no limit)
}


def get_model_path(model_name: str) -> Path:
    """Get the full path to a model file."""
    if model_name in AVAILABLE_MODELS:
        model_name = AVAILABLE_MODELS[model_name]["file"]
    
    if not model_name.endswith('.pt'):
        model_name += '.pt'
    
    return WEIGHTS_DIR / model_name


def print_available_models():
    """Print information about available models."""
    print("\n" + "=" * 60)
    print("   Available YOLOv8 Models")
    print("=" * 60)
    
    for name, info in AVAILABLE_MODELS.items():
        print(f"\n   {name.upper()} ({info['file']})")
        print(f"   {info['description']}")
        print(f"   Speed: {info['speed']}  Accuracy: {info['accuracy']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_available_models()
    print(f"\nCurrent config: {YOLO_CONFIG}")
    print(f"Weights directory: {WEIGHTS_DIR}")
