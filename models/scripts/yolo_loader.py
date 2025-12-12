"""
YOLO Model Loader and Configuration
Handles loading, configuration, and inference for YOLOv8 models.
"""

import os
from pathlib import Path
from ultralytics import YOLO


MODELS_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = MODELS_DIR / "weights"

AVAILABLE_MODELS = {
    "nano": "yolov8n.pt",      # Fastest, least accurate
    "small": "yolov8s.pt",     # Fast, good balance
    "medium": "yolov8m.pt",    # Balanced
    "large": "yolov8l.pt",     # Accurate, slower
    "xlarge": "yolov8x.pt",    # Most accurate, slowest
}

# COCO class IDs
PERSON_CLASS_ID = 0


class YOLOPersonDetector:
    """
    YOLOv8 Person Detection wrapper class.
    Handles model loading and person-specific detection.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the YOLO person detector.
        
        Args:
            model_name: Name of the YOLO model (e.g., 'yolov8n.pt', 'nano', 'small')
            confidence: Detection confidence threshold (0.0 - 1.0)
        """
        self.confidence = confidence
        self.model = self._load_model(model_name)
        self.person_class_id = PERSON_CLASS_ID
        
    def _load_model(self, model_name: str) -> YOLO:
        """
        Load YOLO model from weights directory or download if not present.
        
        Args:
            model_name: Model name or alias (nano, small, medium, large, xlarge)
            
        Returns:
            Loaded YOLO model
        """
        # Check if using alias
        if model_name.lower() in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_name.lower()]
        
        # Ensure .pt extension
        if not model_name.endswith('.pt'):
            model_name += '.pt'
        
        # Check local weights directory first
        local_model_path = WEIGHTS_DIR / model_name
        
        if local_model_path.exists():
            print(f"[OK] Loading model from: {local_model_path}")
            return YOLO(str(local_model_path))
        else:
            print(f"[INFO] Model not found locally. Downloading {model_name}...")
            print(f"   (Will be cached for future use)")
            
            # YOLO auto-downloads to default cache, we load from there
            model = YOLO(model_name)
            
            # Optionally copy to local weights directory for organization
            # model.save(str(local_model_path))
            
            return model
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame.
        
        Args:
            frame: Input image/frame (numpy array)
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                
                if cls_id == self.person_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, conf))
        
        return detections
    
    def detect_all(self, frame):
        """
        Run full YOLO detection on a frame (all classes).
        
        Args:
            frame: Input image/frame (numpy array)
            
        Returns:
            YOLO results object
        """
        return self.model(frame, conf=self.confidence, verbose=False)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model.model_name if hasattr(self.model, 'model_name') else "unknown",
            "confidence_threshold": self.confidence,
            "person_class_id": self.person_class_id,
            "weights_dir": str(WEIGHTS_DIR),
        }


def load_yolo_model(model_name: str = None, confidence: float = None) -> YOLOPersonDetector:
    """
    Factory function to load YOLO model with settings from environment.
    
    Args:
        model_name: Model name (defaults to YOLO_MODEL env var or 'yolov8n.pt')
        confidence: Confidence threshold (defaults to CONFIDENCE_THRESHOLD env var or 0.5)
        
    Returns:
        Configured YOLOPersonDetector instance
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    if model_name is None:
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
    
    if confidence is None:
        confidence = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
    
    print("=" * 50)
    print("   Loading YOLO Person Detector")
    print("=" * 50)
    print(f"   Model: {model_name}")
    print(f"   Confidence: {confidence}")
    print(f"   Weights Dir: {WEIGHTS_DIR}")
    print("=" * 50)
    
    return YOLOPersonDetector(model_name=model_name, confidence=confidence)


# Quick test
if __name__ == "__main__":
    print("Testing YOLO loader...")
    detector = load_yolo_model()
    print(f"\nModel info: {detector.get_model_info()}")
    print("\n[OK] YOLO loader working correctly!")
