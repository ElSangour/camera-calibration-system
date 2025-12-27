"""
Camera Manager
Handles multiple RTSP camera connections and frame capture
"""

import cv2
import threading
import time
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraStream:
    """Represents a single camera stream"""
    camera_id: int
    rtsp_url: str
    name: str = ""
    cap: Optional[cv2.VideoCapture] = None
    is_connected: bool = False
    last_frame: Optional[np.ndarray] = None
    frame_width: int = 0
    frame_height: int = 0
    fps: float = 0
    error_message: str = ""


class CameraManager:
    """
    Manages multiple RTSP camera connections
    Supports async frame capture and reconnection
    """
    
    def __init__(self, preview_width: int = 640, preview_height: int = 480):
        self.cameras: Dict[int, CameraStream] = {}
        self.preview_width = preview_width
        self.preview_height = preview_height
        self._lock = threading.Lock()
        self._running = False
        self._threads: Dict[int, threading.Thread] = {}
        self._frame_callbacks: Dict[int, List[Callable]] = {}
    
    def add_camera(self, camera_id: int, rtsp_url: str, name: str = "") -> bool:
        """Add a camera to the manager"""
        with self._lock:
            if camera_id in self.cameras:
                print(f"[WARN] Camera {camera_id} already exists, updating URL")
                self.cameras[camera_id].rtsp_url = rtsp_url
                self.cameras[camera_id].name = name or f"Camera {camera_id + 1}"
                return True
            
            self.cameras[camera_id] = CameraStream(
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                name=name or f"Camera {camera_id + 1}"
            )
            self._frame_callbacks[camera_id] = []
            print(f"[INFO] Added camera {camera_id}: {name}")
            return True
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera from the manager"""
        self.disconnect_camera(camera_id)
        with self._lock:
            if camera_id in self.cameras:
                del self.cameras[camera_id]
                if camera_id in self._frame_callbacks:
                    del self._frame_callbacks[camera_id]
                print(f"[INFO] Removed camera {camera_id}")
                return True
        return False
    
    def connect_camera(self, camera_id: int) -> bool:
        """Connect to a single camera"""
        with self._lock:
            if camera_id not in self.cameras:
                print(f"[ERROR] Camera {camera_id} not found")
                return False
            
            camera = self.cameras[camera_id]
        
        if not camera.rtsp_url:
            camera.error_message = "No RTSP URL configured"
            print(f"[ERROR] Camera {camera_id}: No RTSP URL")
            return False
        
        try:
            print(f"[INFO] Connecting to camera {camera_id}: {camera.rtsp_url[:50]}...")
            cap = cv2.VideoCapture(camera.rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
            
            if not cap.isOpened():
                camera.error_message = "Failed to connect"
                camera.is_connected = False
                print(f"[ERROR] Camera {camera_id}: Failed to connect")
                return False
            
            # Read test frame
            ret, frame = cap.read()
            if not ret:
                camera.error_message = "Failed to read frame"
                camera.is_connected = False
                cap.release()
                print(f"[ERROR] Camera {camera_id}: Failed to read frame")
                return False
            
            # Update camera info
            camera.cap = cap
            camera.is_connected = True
            camera.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            camera.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            camera.fps = cap.get(cv2.CAP_PROP_FPS) or 25
            camera.last_frame = frame
            camera.error_message = ""
            
            print(f"[OK] Camera {camera_id} connected: {camera.frame_width}x{camera.frame_height} @ {camera.fps:.1f} FPS")
            return True
            
        except Exception as e:
            camera.error_message = str(e)
            camera.is_connected = False
            print(f"[ERROR] Camera {camera_id}: {e}")
            return False
    
    def disconnect_camera(self, camera_id: int):
        """Disconnect a single camera"""
        # Stop capture thread if running
        if camera_id in self._threads:
            self._threads[camera_id] = None
        
        with self._lock:
            if camera_id in self.cameras:
                camera = self.cameras[camera_id]
                if camera.cap is not None:
                    camera.cap.release()
                    camera.cap = None
                camera.is_connected = False
                print(f"[INFO] Camera {camera_id} disconnected")
    
    def connect_all(self) -> Dict[int, bool]:
        """Connect all cameras"""
        results = {}
        for camera_id in self.cameras:
            results[camera_id] = self.connect_camera(camera_id)
        return results
    
    def disconnect_all(self):
        """Disconnect all cameras"""
        self.stop_capture()
        for camera_id in list(self.cameras.keys()):
            self.disconnect_camera(camera_id)
    
    def get_frame(self, camera_id: int, resize: bool = True) -> Optional[np.ndarray]:
        """Get current frame from camera (non-blocking - returns last frame if read would block)"""
        with self._lock:
            if camera_id not in self.cameras:
                return None
            
            camera = self.cameras[camera_id]
            
            if not camera.is_connected or camera.cap is None:
                return camera.last_frame  # Return last known frame
            
            # Try to read frame, but don't block if buffer is empty
            # Set a very short timeout by checking if frame is available
            ret, frame = camera.cap.read()
            if ret:
                camera.last_frame = frame
                if resize:
                    frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                return frame
            else:
                # If read failed, return last frame (non-blocking)
                # Connection might be lost, but don't mark as disconnected immediately
                # to avoid flickering on temporary network issues
                if camera.last_frame is not None:
                    frame = camera.last_frame.copy()
                    if resize:
                        frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                    return frame
                return None
    
    def capture_frame(self, camera_id: int) -> Optional[np.ndarray]:
        """Capture a single frame (full resolution) for calibration"""
        with self._lock:
            if camera_id not in self.cameras:
                return None
            
            camera = self.cameras[camera_id]
            
            if not camera.is_connected or camera.cap is None:
                return None
            
            ret, frame = camera.cap.read()
            if ret:
                camera.last_frame = frame
                return frame
            return None
    
    def register_frame_callback(self, camera_id: int, callback: Callable):
        """Register callback for new frames"""
        if camera_id in self._frame_callbacks:
            self._frame_callbacks[camera_id].append(callback)
    
    def _capture_loop(self, camera_id: int):
        """Background capture loop for a camera"""
        while self._running and camera_id in self._threads:
            frame = self.get_frame(camera_id, resize=True)
            if frame is not None and camera_id in self._frame_callbacks:
                for callback in self._frame_callbacks[camera_id]:
                    try:
                        callback(camera_id, frame)
                    except Exception as e:
                        print(f"[ERROR] Frame callback error: {e}")
            time.sleep(0.033)  # ~30 FPS
    
    def start_capture(self, camera_ids: List[int] = None):
        """Start background capture for cameras"""
        self._running = True
        if camera_ids is None:
            camera_ids = list(self.cameras.keys())
        
        for camera_id in camera_ids:
            if camera_id not in self._threads or self._threads[camera_id] is None:
                thread = threading.Thread(target=self._capture_loop, args=(camera_id,), daemon=True)
                self._threads[camera_id] = thread
                thread.start()
    
    def stop_capture(self):
        """Stop all background capture"""
        self._running = False
        time.sleep(0.1)
        self._threads.clear()
    
    def get_camera_info(self, camera_id: int) -> Optional[Dict]:
        """Get camera information"""
        with self._lock:
            if camera_id not in self.cameras:
                return None
            
            camera = self.cameras[camera_id]
            return {
                "camera_id": camera.camera_id,
                "name": camera.name,
                "rtsp_url": camera.rtsp_url,
                "is_connected": camera.is_connected,
                "resolution": f"{camera.frame_width}x{camera.frame_height}",
                "fps": camera.fps,
                "error": camera.error_message
            }
    
    def get_all_cameras_info(self) -> List[Dict]:
        """Get info for all cameras"""
        return [self.get_camera_info(cid) for cid in self.cameras]
    
    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect_all()


if __name__ == "__main__":
    # Test camera manager
    manager = CameraManager()
    
    # Add test camera
    manager.add_camera(0, "rtsp://example.com/stream1", "Test Camera 1")
    
    print("\nCamera Info:")
    print(manager.get_camera_info(0))
