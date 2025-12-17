"""
Multi-Camera Person Detection with Homography Transformation
Detects persons on multiple RTSP cameras and transforms positions to floor plan coordinates.
"""

import cv2
import os
import sys
import json
import time
import signal
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from threading import Thread, Lock, Event
import queue

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_yolo_model, YOLOPersonDetector
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DetectionPoint:
    """Single detection point data"""
    timestamp: str
    camera_id: int
    camera_name: str
    camera_point: Tuple[float, float]  # Bottom-center in camera frame
    plan_point: Tuple[float, float]    # Transformed point on floor plan
    confidence: float
    bbox: Tuple[int, int, int, int]    # Original bounding box


@dataclass
class CameraStream:
    """Camera stream configuration and state"""
    camera_id: int
    name: str
    rtsp_url: str
    homography_matrix: np.ndarray
    capture: Optional[cv2.VideoCapture] = None
    is_connected: bool = False
    frame_count: int = 0
    detection_count: int = 0


class MultiCameraDetector:
    """
    Multi-camera person detector with homography transformation.
    Connects to multiple RTSP cameras, detects persons, and transforms
    their positions to floor plan coordinates.
    """
    
    def __init__(
        self,
        calibration_path: str,
        model_name: str = None,
        confidence: float = None,
        show_preview: bool = False
    ):
        """
        Initialize the multi-camera detector.
        
        Args:
            calibration_path: Path to calibration JSON file or homography matrices file
            model_name: YOLO model name (defaults to env var)
            confidence: Detection confidence threshold (defaults to env var)
            show_preview: Whether to show real-time preview windows
        """
        self.calibration_path = Path(calibration_path)
        self.show_preview = show_preview
        self.cameras: Dict[int, CameraStream] = {}
        self.detections: List[DetectionPoint] = []
        self.detections_lock = Lock()
        self.stop_event = Event()
        self.store_name = ""
        self.plan_path = ""
        
        # Load YOLO model
        print("[INFO] Loading YOLO model...")
        self.detector = load_yolo_model(model_name, confidence)
        
        # Load calibration data
        self._load_calibration()
        
    def _load_calibration(self):
        """Load calibration data from JSON file"""
        print(f"[INFO] Loading calibration from: {self.calibration_path}")
        
        if not self.calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {self.calibration_path}")
        
        with open(self.calibration_path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a full calibration file or just matrices
        if 'matrices' in data:
            # Homography matrices file
            self.store_name = data.get('store_name', 'unknown')
            self._load_from_matrices_file(data)
        elif 'cameras' in data:
            # Full calibration file
            self.store_name = data.get('metadata', {}).get('store_name', 'unknown')
            self.plan_path = data.get('store', {}).get('plan_path', '')
            self._load_from_calibration_file(data)
        else:
            raise ValueError("Invalid calibration file format")
        
        print(f"[OK] Loaded {len(self.cameras)} cameras for store: {self.store_name}")
    
    def _load_from_matrices_file(self, data: dict):
        """Load camera data from homography matrices file"""
        matrices = data.get('matrices', {})
        
        for key, cam_data in matrices.items():
            # Extract camera ID from key (e.g., "camera_1" -> 1)
            try:
                camera_id = int(key.split('_')[1])
            except (IndexError, ValueError):
                print(f"[WARN] Invalid camera key: {key}, skipping")
                continue
            
            matrix = np.array(cam_data['matrix'])
            name = cam_data.get('name', f'Camera {camera_id}')
            
            # We need RTSP URL from somewhere - try to load full calibration
            rtsp_url = self._find_rtsp_url(camera_id)
            
            if rtsp_url:
                self.cameras[camera_id] = CameraStream(
                    camera_id=camera_id,
                    name=name,
                    rtsp_url=rtsp_url,
                    homography_matrix=matrix
                )
            else:
                print(f"[WARN] No RTSP URL found for camera {camera_id}, skipping")
    
    def _load_from_calibration_file(self, data: dict):
        """Load camera data from full calibration file"""
        cameras = data.get('cameras', [])
        
        for cam in cameras:
            camera_id = cam['camera_id']
            rtsp_url = cam.get('rtsp_url', '')
            name = cam.get('name', f'Camera {camera_id}')
            
            # Get homography matrix
            matrix_data = cam.get('calibration', {}).get('homography_matrix')
            if matrix_data is None:
                print(f"[WARN] No homography matrix for camera {camera_id}, skipping")
                continue
            
            matrix = np.array(matrix_data)
            
            if rtsp_url:
                self.cameras[camera_id] = CameraStream(
                    camera_id=camera_id,
                    name=name,
                    rtsp_url=rtsp_url,
                    homography_matrix=matrix
                )
    
    def _find_rtsp_url(self, camera_id: int) -> Optional[str]:
        """Try to find RTSP URL for a camera from calibration files"""
        # Look for full calibration file in same directory
        calibration_dir = self.calibration_path.parent
        
        for f in calibration_dir.glob(f"calibration_{self.store_name}*.json"):
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    for cam in data.get('cameras', []):
                        if cam['camera_id'] == camera_id:
                            return cam.get('rtsp_url')
            except Exception:
                continue
        
        return None
    
    def connect_cameras(self) -> Dict[int, bool]:
        """
        Connect to all configured cameras.
        
        Returns:
            Dictionary mapping camera_id to connection success status
        """
        results = {}
        
        for camera_id, camera in self.cameras.items():
            print(f"[INFO] Connecting to {camera.name} ({camera.rtsp_url[:50]}...)")
            
            try:
                cap = cv2.VideoCapture(camera.rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    # Test read
                    ret, _ = cap.read()
                    if ret:
                        camera.capture = cap
                        camera.is_connected = True
                        results[camera_id] = True
                        print(f"[OK] Connected to {camera.name}")
                    else:
                        cap.release()
                        results[camera_id] = False
                        print(f"[ERROR] Failed to read from {camera.name}")
                else:
                    results[camera_id] = False
                    print(f"[ERROR] Failed to connect to {camera.name}")
                    
            except Exception as e:
                results[camera_id] = False
                print(f"[ERROR] Exception connecting to {camera.name}: {e}")
        
        connected = sum(1 for v in results.values() if v)
        print(f"[INFO] Connected to {connected}/{len(self.cameras)} cameras")
        
        return results
    
    def disconnect_cameras(self):
        """Disconnect all cameras"""
        for camera in self.cameras.values():
            if camera.capture is not None:
                camera.capture.release()
                camera.is_connected = False
        print("[INFO] All cameras disconnected")
    
    def transform_point(self, camera_point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
        """
        Transform a point from camera coordinates to floor plan coordinates.
        
        Args:
            camera_point: (x, y) point in camera frame
            H: 3x3 homography matrix
            
        Returns:
            (x, y) point in floor plan coordinates
        """
        # Convert to homogeneous coordinates
        pt = np.array([camera_point[0], camera_point[1], 1.0])
        
        # Apply homography
        transformed = H @ pt
        
        # Convert back from homogeneous coordinates
        if transformed[2] != 0:
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
        else:
            x, y = camera_point  # Fallback
        
        return (float(x), float(y))
    
    def get_feet_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Get the bottom-center point of a bounding box (approximate feet position).
        
        Args:
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (x, y) bottom-center point
        """
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_bottom = y2  # Bottom of bounding box
        return (x_center, y_bottom)
    
    def process_frame(self, camera: CameraStream, frame: np.ndarray) -> List[DetectionPoint]:
        """
        Process a single frame from a camera.
        
        Args:
            camera: Camera stream object
            frame: Video frame
            
        Returns:
            List of detection points
        """
        detections = self.detector.detect_persons(frame)
        points = []
        
        timestamp = datetime.now().isoformat()
        
        for bbox_conf in detections:
            x1, y1, x2, y2, conf = bbox_conf
            
            # Get feet position (bottom-center)
            camera_point = self.get_feet_position((x1, y1, x2, y2))
            
            # Transform to floor plan coordinates
            plan_point = self.transform_point(camera_point, camera.homography_matrix)
            
            point = DetectionPoint(
                timestamp=timestamp,
                camera_id=camera.camera_id,
                camera_name=camera.name,
                camera_point=camera_point,
                plan_point=plan_point,
                confidence=conf,
                bbox=(x1, y1, x2, y2)
            )
            points.append(point)
        
        return points
    
    def draw_detections(self, frame: np.ndarray, detections: List[DetectionPoint]) -> np.ndarray:
        """
        Draw detection boxes and points on frame.
        
        Args:
            frame: Video frame
            detections: List of detection points
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw feet position
            feet_x, feet_y = int(det.camera_point[0]), int(det.camera_point[1])
            cv2.circle(annotated, (feet_x, feet_y), 5, (0, 0, 255), -1)
            
            # Draw label
            label = f"Conf: {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw plan coordinates
            plan_label = f"Plan: ({det.plan_point[0]:.0f}, {det.plan_point[1]:.0f})"
            cv2.putText(annotated, plan_label, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated
    
    def run(self, duration_seconds: int = None, max_frames: int = None):
        """
        Run detection on all connected cameras.
        
        Args:
            duration_seconds: How long to run (None for indefinite)
            max_frames: Maximum frames to process per camera (None for indefinite)
        """
        if not any(cam.is_connected for cam in self.cameras.values()):
            print("[ERROR] No cameras connected. Call connect_cameras() first.")
            return
        
        start_time = time.time()
        total_detections = 0
        
        print("[INFO] Starting detection loop...")
        print("[INFO] Press Ctrl+C to stop")
        
        if duration_seconds:
            print(f"[INFO] Will run for {duration_seconds} seconds")
        if max_frames:
            print(f"[INFO] Will process max {max_frames} frames per camera")
        
        try:
            while not self.stop_event.is_set():
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    print("[INFO] Duration limit reached")
                    break
                
                for camera_id, camera in self.cameras.items():
                    if not camera.is_connected or camera.capture is None:
                        continue
                    
                    # Check frame limit
                    if max_frames and camera.frame_count >= max_frames:
                        continue
                    
                    ret, frame = camera.capture.read()
                    if not ret:
                        print(f"[WARN] Failed to read frame from {camera.name}")
                        continue
                    
                    camera.frame_count += 1
                    
                    # Process frame
                    frame_detections = self.process_frame(camera, frame)
                    
                    if frame_detections:
                        with self.detections_lock:
                            self.detections.extend(frame_detections)
                        camera.detection_count += len(frame_detections)
                        total_detections += len(frame_detections)
                    
                    # Show preview if enabled
                    if self.show_preview:
                        annotated = self.draw_detections(frame, frame_detections)
                        # Resize for display
                        display = cv2.resize(annotated, (640, 480))
                        cv2.imshow(f"{camera.name}", display)
                
                # Check if all cameras reached frame limit
                if max_frames and all(cam.frame_count >= max_frames for cam in self.cameras.values() if cam.is_connected):
                    print("[INFO] Frame limit reached for all cameras")
                    break
                
                # Handle preview window events
                if self.show_preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Quit requested")
                        break
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            if self.show_preview:
                cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        print(f"\n[INFO] Detection complete:")
        print(f"   - Duration: {elapsed:.1f} seconds")
        print(f"   - Total detections: {total_detections}")
        for camera in self.cameras.values():
            if camera.is_connected:
                print(f"   - {camera.name}: {camera.frame_count} frames, {camera.detection_count} detections")
    
    def stop(self):
        """Signal the detection loop to stop"""
        self.stop_event.set()
    
    def save_detections(self, output_path: str = None) -> str:
        """
        Save detections to JSON file.
        
        Args:
            output_path: Output file path (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"detections_{self.store_name}_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.detections_lock:
            data = {
                "metadata": {
                    "store_name": self.store_name,
                    "created_at": datetime.now().isoformat(),
                    "total_detections": len(self.detections),
                    "cameras": {
                        cam.camera_id: {
                            "name": cam.name,
                            "frame_count": cam.frame_count,
                            "detection_count": cam.detection_count
                        }
                        for cam in self.cameras.values()
                    }
                },
                "plan_path": self.plan_path,
                "detections": [
                    {
                        "timestamp": d.timestamp,
                        "camera_id": d.camera_id,
                        "camera_name": d.camera_name,
                        "camera_point": list(d.camera_point),
                        "plan_point": list(d.plan_point),
                        "confidence": d.confidence,
                        "bbox": list(d.bbox)
                    }
                    for d in self.detections
                ]
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Saved {len(self.detections)} detections to: {output_path}")
        return str(output_path)
    
    def get_detections(self) -> List[DetectionPoint]:
        """Get all collected detections"""
        with self.detections_lock:
            return list(self.detections)
    
    def clear_detections(self):
        """Clear all collected detections"""
        with self.detections_lock:
            self.detections.clear()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-camera person detection with homography transformation"
    )
    parser.add_argument(
        "--calibration", "-c",
        required=True,
        help="Path to calibration JSON file (full or matrices only)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=None,
        help="Duration in seconds to run (default: indefinite)"
    )
    parser.add_argument(
        "--frames", "-f",
        type=int,
        default=None,
        help="Maximum frames per camera (default: indefinite)"
    )
    parser.add_argument(
        "--preview", "-p",
        action="store_true",
        help="Show real-time preview windows"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="YOLO model name (default: from env)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold (default: from env)"
    )
    
    args = parser.parse_args()
    
    # Setup signal handler for graceful shutdown
    detector = None
    
    def signal_handler(sig, frame):
        print("\n[INFO] Shutdown signal received...")
        if detector:
            detector.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize detector
        detector = MultiCameraDetector(
            calibration_path=args.calibration,
            model_name=args.model,
            confidence=args.confidence,
            show_preview=args.preview
        )
        
        # Connect to cameras
        results = detector.connect_cameras()
        
        if not any(results.values()):
            print("[ERROR] Failed to connect to any cameras")
            return 1
        
        # Run detection
        detector.run(
            duration_seconds=args.duration,
            max_frames=args.frames
        )
        
        # Save detections
        detector.save_detections(args.output)
        
        # Cleanup
        detector.disconnect_cameras()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
