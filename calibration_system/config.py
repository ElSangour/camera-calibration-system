"""
Calibration System Configuration
Manages all settings for the calibration process
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class CameraConfig:
    """Configuration for a single camera"""
    camera_id: int
    name: str
    rtsp_url: str
    enabled: bool = True
    required_points: int = 4  # Number of points required for calibration
    calibration_points_camera: List[List[float]] = field(default_factory=list)  # Points on camera view
    calibration_points_plan: List[List[float]] = field(default_factory=list)    # Corresponding points on floor plan
    homography_matrix: Optional[List[List[float]]] = None
    
    def is_calibrated(self) -> bool:
        """Check if camera has valid calibration"""
        return (len(self.calibration_points_camera) >= self.required_points and 
                len(self.calibration_points_plan) >= self.required_points and
                self.homography_matrix is not None)
    
    def calibration_progress(self) -> tuple:
        """Return (current_points, required_points) for progress tracking"""
        cam_pts = len(self.calibration_points_camera)
        plan_pts = len(self.calibration_points_plan)
        current = min(cam_pts, plan_pts)
        return (current, self.required_points)


@dataclass 
class CalibrationConfig:
    """Main configuration for the calibration system"""
    
    # Store information
    store_name: str = "My Store"
    store_plan_path: str = ""
    store_plan_width: int = 800   # Display width for floor plan
    store_plan_height: int = 600  # Display height for floor plan
    
    # Camera settings
    num_cameras: int = 1
    cameras: List[CameraConfig] = field(default_factory=list)
    
    # Calibration settings
    min_calibration_points: int = 4
    point_radius: int = 8
    point_color_camera: str = "#00FF00"  # Green for camera points
    point_color_plan: str = "#FF0000"    # Red for plan points
    line_color: str = "#FFFF00"          # Yellow for connection lines
    
    # Display settings
    camera_preview_width: int = 640
    camera_preview_height: int = 480
    
    # Output settings
    output_dir: str = "./calibration_data"
    
    def __post_init__(self):
        """Initialize cameras list if empty"""
        if not self.cameras and self.num_cameras > 0:
            self.cameras = [
                CameraConfig(
                    camera_id=i,
                    name=f"Camera {i+1}",
                    rtsp_url=""
                )
                for i in range(self.num_cameras)
            ]
    
    def add_camera(self, name: str = "", rtsp_url: str = "") -> CameraConfig:
        """Add a new camera to the configuration"""
        camera_id = len(self.cameras)
        camera = CameraConfig(
            camera_id=camera_id,
            name=name or f"Camera {camera_id + 1}",
            rtsp_url=rtsp_url
        )
        self.cameras.append(camera)
        self.num_cameras = len(self.cameras)
        return camera
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera by ID"""
        for i, cam in enumerate(self.cameras):
            if cam.camera_id == camera_id:
                self.cameras.pop(i)
                self.num_cameras = len(self.cameras)
                # Re-index remaining cameras
                for j, c in enumerate(self.cameras):
                    c.camera_id = j
                return True
        return False
    
    def get_camera(self, camera_id: int) -> Optional[CameraConfig]:
        """Get camera by ID"""
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        return None
    
    def all_cameras_calibrated(self) -> bool:
        """Check if all cameras are calibrated"""
        return all(cam.is_calibrated() for cam in self.cameras if cam.enabled)
    
    def get_calibration_summary(self) -> Dict:
        """Get summary of calibration status"""
        total = len([c for c in self.cameras if c.enabled])
        calibrated = len([c for c in self.cameras if c.enabled and c.is_calibrated()])
        return {
            "total_cameras": total,
            "calibrated": calibrated,
            "pending": total - calibrated,
            "complete": total == calibrated and total > 0
        }
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        data = {
            "store_name": self.store_name,
            "store_plan_path": self.store_plan_path,
            "store_plan_width": self.store_plan_width,
            "store_plan_height": self.store_plan_height,
            "num_cameras": self.num_cameras,
            "min_calibration_points": self.min_calibration_points,
            "point_radius": self.point_radius,
            "point_color_camera": self.point_color_camera,
            "point_color_plan": self.point_color_plan,
            "line_color": self.line_color,
            "camera_preview_width": self.camera_preview_width,
            "camera_preview_height": self.camera_preview_height,
            "output_dir": self.output_dir,
            "cameras": [asdict(cam) for cam in self.cameras]
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationConfig":
        """Create config from dictionary"""
        cameras_data = data.pop("cameras", [])
        config = cls(**data)
        config.cameras = [CameraConfig(**cam) for cam in cameras_data]
        config.num_cameras = len(config.cameras)
        return config
    
    def save(self, filepath: str = None):
        """Save configuration to JSON file"""
        if filepath is None:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, "calibration_config.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[INFO] Configuration saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "CalibrationConfig":
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"[INFO] Configuration loaded from: {filepath}")
        return cls.from_dict(data)


# Default configuration instance
def get_default_config() -> CalibrationConfig:
    """Get default calibration configuration"""
    return CalibrationConfig()


if __name__ == "__main__":
    # Test configuration
    config = CalibrationConfig(
        store_name="Test Store",
        num_cameras=3
    )
    
    # Add RTSP URLs
    config.cameras[0].rtsp_url = "rtsp://example.com/cam1"
    config.cameras[1].rtsp_url = "rtsp://example.com/cam2"
    config.cameras[2].rtsp_url = "rtsp://example.com/cam3"
    
    print("Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    print("\nCalibration Summary:")
    print(config.get_calibration_summary())
