"""
Traffic Heatmap - Main Application
Integrated Qt GUI for calibration, detection, and heatmap visualization
"""

import sys
import os
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from threading import Thread, Event

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QTabWidget, QFileDialog, QMessageBox, QGroupBox, QScrollArea,
    QComboBox, QProgressBar, QStatusBar, QMenuBar, QMenu,
    QDialog, QFormLayout, QDialogButtonBox, QTextEdit, QSplitter,
    QFrame, QListWidget, QListWidgetItem, QStackedWidget, QSlider,
    QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QAction, QFont, QPixmap, QColor, QImage

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_system.config import CalibrationConfig, CameraConfig
from calibration_system.camera_manager import CameraManager
from calibration_system.homography import HomographyCalculator
from calibration_system.storage import CalibrationStorage
from calibration_system.widgets import PointSelectorWidget, CameraPreviewWidget


# =============================================================================
# Heatmap Generator (Integrated)
# =============================================================================

COLORMAPS = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'viridis': cv2.COLORMAP_VIRIDIS,
    'plasma': cv2.COLORMAP_PLASMA,
    'inferno': cv2.COLORMAP_INFERNO,
    'turbo': cv2.COLORMAP_TURBO,
    'rainbow': cv2.COLORMAP_RAINBOW,
    'cool': cv2.COLORMAP_COOL,
}


class HeatmapGenerator:
    """Generate heatmaps from detection data"""
    
    def __init__(self, plan_image: np.ndarray):
        self.plan_image = plan_image
        self.height, self.width = plan_image.shape[:2]
    
    def generate(
        self,
        points: List[tuple],
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.6,
        sigma: float = 20.0
    ) -> np.ndarray:
        """Generate heatmap from points"""
        if not points:
            return self.plan_image.copy()
        
        # Create density map
        density = np.zeros((self.height, self.width), dtype=np.float32)
        
        for x, y in points:
            ix, iy = int(x), int(y)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                density[iy, ix] += 1
        
        # Apply Gaussian blur
        if sigma > 0:
            ksize = int(sigma * 6) | 1
            ksize = max(3, ksize)
            density = cv2.GaussianBlur(density, (ksize, ksize), sigma)
        
        # Normalize
        if density.max() > 0:
            density = (density / density.max() * 255).astype(np.uint8)
        else:
            density = np.zeros_like(density, dtype=np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(density, colormap)
        
        # Create alpha mask
        alpha_mask = density.astype(np.float32) / 255.0 * alpha
        alpha_mask = np.stack([alpha_mask] * 3, axis=-1)
        
        # Blend
        result = (
            self.plan_image.astype(np.float32) * (1 - alpha_mask) +
            heatmap.astype(np.float32) * alpha_mask
        ).astype(np.uint8)
        
        return result


# =============================================================================
# Worker Threads
# =============================================================================

class ModelLoaderWorker(QObject):
    """Worker thread for loading YOLO model"""
    model_loaded = pyqtSignal(object)  # detector
    error = pyqtSignal(str)
    
    def load_model(self):
        """Load YOLO model in background"""
        try:
            from models import load_yolo_model
            detector = load_yolo_model()
            self.model_loaded.emit(detector)
        except Exception as e:
            self.error.emit(str(e))


class CameraConnectionWorker(QObject):
    """Worker thread for connecting cameras"""
    camera_connected = pyqtSignal(int, object)  # camera_id, capture
    connection_failed = pyqtSignal(int, str)  # camera_id, error
    finished = pyqtSignal()
    
    def __init__(self, cameras: Dict, parent=None):
        super().__init__(parent)
        self.cameras = cameras
    
    def connect_all(self):
        """Connect to all cameras in background"""
        for cam_id, cam_data in self.cameras.items():
            url = cam_data.get('url')
            if not url:
                continue
            
            try:
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.camera_connected.emit(cam_id, cap)
                    else:
                        cap.release()
                        self.connection_failed.emit(cam_id, "Failed to read frame")
                else:
                    self.connection_failed.emit(cam_id, "Failed to open connection")
            except Exception as e:
                self.connection_failed.emit(cam_id, str(e))
        
        self.finished.emit()


class DetectionWorker(QObject):
    """Worker thread for running multi-camera detection"""
    
    frame_processed = pyqtSignal(int, np.ndarray, list)  # camera_id, frame, detections
    detection_added = pyqtSignal(dict)  # single detection
    status_update = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, cameras: Dict, detector, parent=None):
        super().__init__(parent)
        self.cameras = cameras  # {camera_id: {url, matrix, capture}}
        self.detector = detector
        self.is_running = False
        self.stop_flag = Event()
    
    def run(self):
        """Main detection loop"""
        self.is_running = True
        self.stop_flag.clear()
        
        try:
            import time
            last_frame_time = {}  # Track last frame time per camera for throttling
            
            while not self.stop_flag.is_set():
                frame_processed_this_cycle = False
                current_time = time.time()
                
                for cam_id, cam_data in self.cameras.items():
                    if self.stop_flag.is_set():
                        break
                    
                    capture = cam_data.get('capture')
                    if capture is None or not capture.isOpened():
                        continue
                    
                    # Throttle frame processing to ~10 FPS per camera to avoid overwhelming GUI
                    if cam_id in last_frame_time:
                        if current_time - last_frame_time[cam_id] < 0.1:  # 10 FPS max
                            continue
                    
                    # Use non-blocking read
                    ret, frame = capture.read()
                    if not ret:
                        continue
                    
                    last_frame_time[cam_id] = current_time
                    
                    # Run detection
                    detections = self.detector.detect_persons(frame)
                    
                    # Process detections
                    detection_list = []
                    H = np.array(cam_data['matrix'])
                    
                    for det in detections:
                        x1, y1, x2, y2, conf = det
                        
                        # Get feet position (bottom center)
                        feet_x = (x1 + x2) / 2
                        feet_y = y2
                        
                        # Transform to plan coordinates
                        pt = np.array([feet_x, feet_y, 1.0])
                        transformed = H @ pt
                        if transformed[2] != 0:
                            plan_x = transformed[0] / transformed[2]
                            plan_y = transformed[1] / transformed[2]
                        else:
                            plan_x, plan_y = feet_x, feet_y
                        
                        detection = {
                            'timestamp': datetime.now().isoformat(),
                            'camera_id': cam_id,
                            'camera_point': (feet_x, feet_y),
                            'plan_point': (plan_x, plan_y),
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2)
                        }
                        detection_list.append(detection)
                        self.detection_added.emit(detection)
                    
                    self.frame_processed.emit(cam_id, frame, detection_list)
                    frame_processed_this_cycle = True
                
                # Yield to event loop if no frames processed (prevents tight loop)
                if not frame_processed_this_cycle:
                    time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except Exception as e:
            self.error.emit(str(e))
        
        finally:
            self.is_running = False
            self.finished.emit()
    
    def stop(self):
        """Stop detection loop"""
        self.stop_flag.set()


# =============================================================================
# Heatmap Tab Widget
# =============================================================================

class HeatmapTab(QWidget):
    """Tab for heatmap visualization"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plan_image = None
        self.plan_path = ""
        self.detections = []
        self.generator = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Plan selection
        plan_group = QGroupBox("Floor Plan")
        plan_layout = QVBoxLayout()
        
        plan_btn_layout = QHBoxLayout()
        self.plan_path_label = QLabel("No plan loaded")
        self.plan_path_label.setWordWrap(True)
        plan_layout.addWidget(self.plan_path_label)
        
        self.load_plan_btn = QPushButton("Load Floor Plan")
        self.load_plan_btn.clicked.connect(self._load_plan)
        plan_layout.addWidget(self.load_plan_btn)
        
        plan_group.setLayout(plan_layout)
        left_layout.addWidget(plan_group)
        
        # Detection data
        data_group = QGroupBox("Detection Data")
        data_layout = QVBoxLayout()
        
        self.data_path_label = QLabel("No data loaded")
        self.data_path_label.setWordWrap(True)
        data_layout.addWidget(self.data_path_label)
        
        self.load_data_btn = QPushButton("Load Detection Data")
        self.load_data_btn.clicked.connect(self._load_detections)
        data_layout.addWidget(self.load_data_btn)
        
        self.detection_count_label = QLabel("Points: 0")
        data_layout.addWidget(self.detection_count_label)
        
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)
        
        # Heatmap settings
        settings_group = QGroupBox("Heatmap Settings")
        settings_layout = QFormLayout()
        
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(list(COLORMAPS.keys()))
        self.colormap_combo.setCurrentText('jet')
        self.colormap_combo.currentTextChanged.connect(self._update_heatmap)
        settings_layout.addRow("Colormap:", self.colormap_combo)
        
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(60)
        self.alpha_slider.valueChanged.connect(self._update_heatmap)
        self.alpha_label = QLabel("60%")
        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(self.alpha_slider)
        alpha_layout.addWidget(self.alpha_label)
        settings_layout.addRow("Transparency:", alpha_layout)
        
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(1.0, 100.0)
        self.sigma_spin.setValue(20.0)
        self.sigma_spin.setSingleStep(5.0)
        self.sigma_spin.valueChanged.connect(self._update_heatmap)
        settings_layout.addRow("Blur Sigma:", self.sigma_spin)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        
        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()
        
        self.generate_btn = QPushButton("Generate Heatmap")
        self.generate_btn.clicked.connect(self._generate_heatmap)
        self.generate_btn.setEnabled(False)
        actions_layout.addWidget(self.generate_btn)
        
        self.save_btn = QPushButton("Save Heatmap")
        self.save_btn.clicked.connect(self._save_heatmap)
        self.save_btn.setEnabled(False)
        actions_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self._clear_data)
        actions_layout.addWidget(self.clear_btn)
        
        actions_group.setLayout(actions_layout)
        left_layout.addWidget(actions_group)
        
        left_layout.addStretch()
        layout.addWidget(left_panel)
        
        # Right panel - Display
        self.display_label = QLabel("Load a floor plan and detection data to generate heatmap")
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_label.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self.display_label.setMinimumSize(800, 600)
        layout.addWidget(self.display_label, 1)
    
    def _load_plan(self):
        """Load floor plan image"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Floor Plan",
            "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filepath:
            self.plan_image = cv2.imread(filepath)
            if self.plan_image is not None:
                self.plan_path = filepath
                self.plan_path_label.setText(os.path.basename(filepath))
                self.generator = HeatmapGenerator(self.plan_image)
                self._display_image(self.plan_image)
                self._check_ready()
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")
    
    def _load_detections(self):
        """Load detection data from JSON"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Detection Data",
            "output", "JSON Files (*.json)"
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.detections = []
                for det in data.get('detections', []):
                    plan_point = det.get('plan_point')
                    if plan_point and len(plan_point) == 2:
                        self.detections.append(tuple(plan_point))
                
                self.data_path_label.setText(os.path.basename(filepath))
                self.detection_count_label.setText(f"Points: {len(self.detections)}")
                
                # Try to load plan from detection data
                if self.plan_image is None:
                    plan_path = data.get('plan_path')
                    if plan_path and os.path.exists(plan_path):
                        self.plan_image = cv2.imread(plan_path)
                        if self.plan_image is not None:
                            self.plan_path = plan_path
                            self.plan_path_label.setText(os.path.basename(plan_path))
                            self.generator = HeatmapGenerator(self.plan_image)
                            self._display_image(self.plan_image)
                
                self._check_ready()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load detections: {e}")
    
    def _check_ready(self):
        """Check if ready to generate heatmap"""
        ready = self.plan_image is not None and len(self.detections) > 0
        self.generate_btn.setEnabled(ready)
    
    def _update_heatmap(self):
        """Update heatmap with new settings"""
        self.alpha_label.setText(f"{self.alpha_slider.value()}%")
        if self.plan_image is not None and len(self.detections) > 0:
            self._generate_heatmap()
    
    def _generate_heatmap(self):
        """Generate and display heatmap in background thread"""
        if self.generator is None or not self.detections:
            return
        
        # Disable button during generation
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        QApplication.processEvents()
        
        # Generate in background to avoid blocking UI
        colormap_name = self.colormap_combo.currentText()
        colormap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)
        alpha = self.alpha_slider.value() / 100.0
        sigma = self.sigma_spin.value()
        
        # For small datasets, generate directly (fast enough)
        # For large datasets, could use QThread but heatmap generation is usually fast
        try:
            result = self.generator.generate(
                self.detections,
                colormap=colormap,
                alpha=alpha,
                sigma=sigma
            )
            
            self._display_image(result)
            self.save_btn.setEnabled(True)
            self.current_heatmap = result
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to generate heatmap: {e}")
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("Generate Heatmap")
    
    def _display_image(self, image: np.ndarray):
        """Display image in label"""
        if image is None:
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        
        # Scale to fit
        label_size = self.display_label.size()
        scale = min(label_size.width() / w, label_size.height() / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if scale < 1.0:
            rgb = cv2.resize(rgb, (new_w, new_h))
            h, w = new_h, new_w
        
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.display_label.setPixmap(pixmap)
    
    def _save_heatmap(self):
        """Save heatmap to file"""
        if not hasattr(self, 'current_heatmap'):
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"heatmap_{timestamp}.png"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Heatmap",
            f"output/{default_name}",
            "PNG Images (*.png);;JPEG Images (*.jpg)"
        )
        
        if filepath:
            cv2.imwrite(filepath, self.current_heatmap)
            QMessageBox.information(self, "Saved", f"Heatmap saved to:\n{filepath}")
    
    def _clear_data(self):
        """Clear all data"""
        self.detections = []
        self.detection_count_label.setText("Points: 0")
        self.data_path_label.setText("No data loaded")
        self.generate_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        if self.plan_image is not None:
            self._display_image(self.plan_image)
    
    def add_detection(self, plan_point: tuple):
        """Add a detection point (from live detection)"""
        self.detections.append(plan_point)
        self.detection_count_label.setText(f"Points: {len(self.detections)}")
        self._check_ready()
    
    def set_plan(self, plan_path: str):
        """Set floor plan from path"""
        if os.path.exists(plan_path):
            self.plan_image = cv2.imread(plan_path)
            if self.plan_image is not None:
                self.plan_path = plan_path
                self.plan_path_label.setText(os.path.basename(plan_path))
                self.generator = HeatmapGenerator(self.plan_image)
                self._display_image(self.plan_image)
                self._check_ready()


# =============================================================================
# Live Detection Tab Widget
# =============================================================================

class LiveDetectionTab(QWidget):
    """Tab for live multi-camera detection"""
    
    detection_recorded = pyqtSignal(tuple)  # plan_point
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cameras = {}
        self.calibration_data = None
        self.detector = None
        self.worker = None
        self.worker_thread = None
        self.all_detections = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Calibration loading
        cal_group = QGroupBox("Calibration Data")
        cal_layout = QVBoxLayout()
        
        self.cal_path_label = QLabel("No calibration loaded")
        self.cal_path_label.setWordWrap(True)
        cal_layout.addWidget(self.cal_path_label)
        
        self.load_cal_btn = QPushButton("Load Calibration")
        self.load_cal_btn.clicked.connect(self._load_calibration)
        cal_layout.addWidget(self.load_cal_btn)
        
        cal_group.setLayout(cal_layout)
        left_layout.addWidget(cal_group)
        
        # Camera list
        cam_group = QGroupBox("Cameras")
        cam_layout = QVBoxLayout()
        
        self.camera_list = QListWidget()
        self.camera_list.setMaximumHeight(150)
        cam_layout.addWidget(self.camera_list)
        
        self.connect_btn = QPushButton("Connect All Cameras")
        self.connect_btn.clicked.connect(self._connect_cameras)
        self.connect_btn.setEnabled(False)
        cam_layout.addWidget(self.connect_btn)
        
        cam_group.setLayout(cam_layout)
        left_layout.addWidget(cam_group)
        
        # Detection controls
        detect_group = QGroupBox("Detection")
        detect_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self._toggle_detection)
        self.start_btn.setEnabled(False)
        detect_layout.addWidget(self.start_btn)
        
        self.detection_count_label = QLabel("Detections: 0")
        detect_layout.addWidget(self.detection_count_label)
        
        self.save_detections_btn = QPushButton("Save Detections")
        self.save_detections_btn.clicked.connect(self._save_detections)
        self.save_detections_btn.setEnabled(False)
        detect_layout.addWidget(self.save_detections_btn)
        
        self.clear_btn = QPushButton("Clear Detections")
        self.clear_btn.clicked.connect(self._clear_detections)
        detect_layout.addWidget(self.clear_btn)
        
        detect_group.setLayout(detect_layout)
        left_layout.addWidget(detect_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)
        
        left_layout.addStretch()
        layout.addWidget(left_panel)
        
        # Right panel - Camera preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.preview_label = QLabel("Load calibration and connect cameras to start")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self.preview_label.setMinimumSize(800, 600)
        right_layout.addWidget(self.preview_label)
        
        # Camera selector
        cam_select_layout = QHBoxLayout()
        cam_select_layout.addWidget(QLabel("Preview Camera:"))
        self.preview_camera_combo = QComboBox()
        self.preview_camera_combo.currentIndexChanged.connect(self._change_preview_camera)
        cam_select_layout.addWidget(self.preview_camera_combo)
        cam_select_layout.addStretch()
        right_layout.addLayout(cam_select_layout)
        
        layout.addWidget(right_panel, 1)
        
        self.current_preview_camera = None
    
    def _load_calibration(self):
        """Load calibration data"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration File",
            "calibration_data",
            "JSON Files (*.json)"
        )
        if not filepath:
            return
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.calibration_data = data
            self.cameras = {}
            
            # Check if it's a full calibration file or matrices only
            if 'cameras' in data:
                # Full calibration file
                for cam in data['cameras']:
                    cam_id = cam['camera_id']
                    matrix = cam.get('calibration', {}).get('homography_matrix')
                    if matrix:
                        self.cameras[cam_id] = {
                            'name': cam.get('name', f'Camera {cam_id}'),
                            'url': cam.get('rtsp_url', ''),
                            'matrix': matrix,
                            'capture': None
                        }
                self.plan_path = data.get('store', {}).get('plan_path', '')
            elif 'matrices' in data:
                # Matrices only file - need to find URLs
                for key, cam_data in data['matrices'].items():
                    try:
                        cam_id = int(key.split('_')[1])
                        self.cameras[cam_id] = {
                            'name': cam_data.get('name', f'Camera {cam_id}'),
                            'url': '',  # Will need to be set manually
                            'matrix': cam_data['matrix'],
                            'capture': None
                        }
                    except:
                        continue
            
            # Update UI
            self.cal_path_label.setText(os.path.basename(filepath))
            self.camera_list.clear()
            for cam_id, cam_data in self.cameras.items():
                item = QListWidgetItem(f"{cam_data['name']} (ID: {cam_id})")
                self.camera_list.addItem(item)
            
            self.connect_btn.setEnabled(len(self.cameras) > 0)
            self.status_label.setText(f"Loaded {len(self.cameras)} cameras")
            
            # Update preview camera combo
            self.preview_camera_combo.clear()
            for cam_id, cam_data in self.cameras.items():
                self.preview_camera_combo.addItem(cam_data['name'], cam_id)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load calibration: {e}")
    
    def _connect_cameras(self):
        """Connect to all cameras in background thread"""
        if not self.cameras:
            return
        
        self.status_label.setText("Connecting to cameras...")
        self.connect_btn.setEnabled(False)
        self.connected_count = 0
        
        # Create worker thread for camera connections
        self.camera_worker = CameraConnectionWorker(self.cameras)
        self.camera_worker_thread = QThread()
        self.camera_worker.moveToThread(self.camera_worker_thread)
        
        self.camera_worker_thread.started.connect(self.camera_worker.connect_all)
        self.camera_worker.camera_connected.connect(self._on_camera_connected)
        self.camera_worker.connection_failed.connect(self._on_camera_failed)
        self.camera_worker.finished.connect(self._on_cameras_connected_finished)
        
        self.camera_worker_thread.start()
    
    def _on_camera_connected(self, cam_id: int, capture):
        """Handle successful camera connection"""
        if cam_id in self.cameras:
            self.cameras[cam_id]['capture'] = capture
            self.connected_count += 1
            self.status_label.setText(f"Connecting... ({self.connected_count}/{len(self.cameras)})")
    
    def _on_camera_failed(self, cam_id: int, error: str):
        """Handle camera connection failure"""
        print(f"[ERROR] Failed to connect to camera {cam_id}: {error}")
    
    def _on_cameras_connected_finished(self):
        """Handle camera connection process finished"""
        self.camera_worker_thread.quit()
        self.camera_worker_thread.wait(1000)
        
        self.status_label.setText(f"Connected: {self.connected_count}/{len(self.cameras)}")
        self.connect_btn.setEnabled(True)
        
        if self.connected_count > 0:
            self.start_btn.setEnabled(True)
            # Load YOLO model in background
            if self.detector is None:
                self._load_yolo_model_async()
            else:
                self.status_label.setText(f"Ready - {self.connected_count} cameras connected")
    
    def _load_yolo_model_async(self):
        """Load YOLO model in background thread"""
        self.status_label.setText("Loading YOLO model...")
        self.start_btn.setEnabled(False)
        
        self.model_worker = ModelLoaderWorker()
        self.model_worker_thread = QThread()
        self.model_worker.moveToThread(self.model_worker_thread)
        
        self.model_worker_thread.started.connect(self.model_worker.load_model)
        self.model_worker.model_loaded.connect(self._on_model_loaded)
        self.model_worker.error.connect(self._on_model_error)
        
        self.model_worker_thread.start()
    
    def _on_model_loaded(self, detector):
        """Handle YOLO model loaded"""
        self.detector = detector
        self.model_worker_thread.quit()
        self.model_worker_thread.wait(1000)
        self.status_label.setText(f"Ready - {self.connected_count} cameras connected")
        self.start_btn.setEnabled(True)
    
    def _on_model_error(self, error: str):
        """Handle YOLO model loading error"""
        self.model_worker_thread.quit()
        self.model_worker_thread.wait(1000)
        QMessageBox.warning(self, "Error", f"Failed to load YOLO model: {error}")
        self.status_label.setText("Model loading failed")
        self.start_btn.setEnabled(False)
    
    def _toggle_detection(self):
        """Start/stop detection"""
        if self.worker is not None and self.worker.is_running:
            self._stop_detection()
        else:
            self._start_detection()
    
    def _start_detection(self):
        """Start detection worker"""
        if self.detector is None:
            return
        
        self.worker = DetectionWorker(self.cameras, self.detector)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.detection_added.connect(self._on_detection)
        self.worker.finished.connect(self._on_detection_finished)
        self.worker.error.connect(self._on_detection_error)
        
        self.worker_thread.start()
        
        self.start_btn.setText("Stop Detection")
        self.status_label.setText("Detection running...")
    
    def _stop_detection(self):
        """Stop detection worker"""
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            # Use non-blocking wait with timeout
            if not self.worker_thread.wait(1000):
                # Force terminate if still running after 1 second
                self.worker_thread.terminate()
                self.worker_thread.wait(500)
        
        self.start_btn.setText("Start Detection")
        self.status_label.setText("Detection stopped")
    
    def _on_frame_processed(self, camera_id: int, frame: np.ndarray, detections: list):
        """Handle processed frame"""
        # Update preview if this is the selected camera
        current_cam = self.preview_camera_combo.currentData()
        if current_cam == camera_id:
            self._display_frame(frame, detections)
    
    def _on_detection(self, detection: dict):
        """Handle new detection"""
        self.all_detections.append(detection)
        self.detection_count_label.setText(f"Detections: {len(self.all_detections)}")
        self.save_detections_btn.setEnabled(True)
        
        # Emit for heatmap tab
        plan_point = detection.get('plan_point')
        if plan_point:
            self.detection_recorded.emit(tuple(plan_point))
    
    def _on_detection_finished(self):
        """Handle detection finished"""
        self.start_btn.setText("Start Detection")
        self.status_label.setText("Detection finished")
    
    def _on_detection_error(self, error: str):
        """Handle detection error"""
        QMessageBox.warning(self, "Detection Error", error)
        self._stop_detection()
    
    def _display_frame(self, frame: np.ndarray, detections: list):
        """Display frame with detections"""
        annotated = frame.copy()
        
        for det in detections:
            bbox = det.get('bbox')
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw feet point
                cam_pt = det.get('camera_point')
                if cam_pt:
                    cv2.circle(annotated, (int(cam_pt[0]), int(cam_pt[1])), 5, (0, 0, 255), -1)
        
        # Convert and display
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        
        # Scale to fit
        label_size = self.preview_label.size()
        scale = min(label_size.width() / w, label_size.height() / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        
        if scale < 1.0:
            rgb = cv2.resize(rgb, (new_w, new_h))
            h, w = new_h, new_w
        
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.preview_label.setPixmap(pixmap)
    
    def _change_preview_camera(self, index):
        """Change preview camera"""
        pass  # Preview updates automatically on frame_processed
    
    def _save_detections(self):
        """Save detections to file"""
        if not self.all_detections:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        store_name = self.calibration_data.get('metadata', {}).get('store_name', 'unknown') if self.calibration_data else 'unknown'
        default_name = f"detections_{store_name}_{timestamp}.json"
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Detections",
            f"output/{default_name}",
            "JSON Files (*.json)"
        )
        
        if filepath:
            data = {
                'metadata': {
                    'store_name': store_name,
                    'created_at': datetime.now().isoformat(),
                    'total_detections': len(self.all_detections)
                },
                'plan_path': getattr(self, 'plan_path', ''),
                'detections': self.all_detections
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            QMessageBox.information(self, "Saved", f"Detections saved to:\n{filepath}")
    
    def _clear_detections(self):
        """Clear all detections"""
        self.all_detections = []
        self.detection_count_label.setText("Detections: 0")
        self.save_detections_btn.setEnabled(False)
    
    def cleanup(self):
        """Clean up resources"""
        self._stop_detection()
        for cam_data in self.cameras.values():
            cap = cam_data.get('capture')
            if cap:
                cap.release()


# =============================================================================
# Main Application Window
# =============================================================================

class TrafficHeatmapApp(QMainWindow):
    """Main application window with tabs for all features"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Heatmap - Multi-Camera Analysis System")
        self.setMinimumSize(1400, 900)
        self.showMaximized()
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
    
    def _setup_ui(self):
        """Setup main UI with tabs"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Tab 1: Calibration (import existing)
        from calibration_system.calibration_app import CalibrationMainWindow
        self.calibration_widget = CalibrationMainWindow()
        # Remove menu bar from calibration window since we have our own
        self.calibration_widget.setMenuBar(None)
        self.tabs.addTab(self.calibration_widget, "Calibration")
        
        # Tab 2: Live Detection
        self.detection_tab = LiveDetectionTab()
        self.tabs.addTab(self.detection_tab, "Live Detection")
        
        # Tab 3: Heatmap
        self.heatmap_tab = HeatmapTab()
        self.tabs.addTab(self.heatmap_tab, "Heatmap Visualization")
        
        # Connect detection to heatmap
        self.detection_tab.detection_recorded.connect(self.heatmap_tab.add_detection)
        
        layout.addWidget(self.tabs)
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_cal_action = QAction("New Calibration", self)
        new_cal_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        file_menu.addAction(new_cal_action)
        
        load_cal_action = QAction("Load Calibration", self)
        load_cal_action.triggered.connect(self._load_calibration_for_detection)
        file_menu.addAction(load_cal_action)
        
        file_menu.addSeparator()
        
        load_detections_action = QAction("Load Detections", self)
        load_detections_action.triggered.connect(self._load_detections_for_heatmap)
        file_menu.addAction(load_detections_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        cal_tab_action = QAction("Calibration Tab", self)
        cal_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        view_menu.addAction(cal_tab_action)
        
        detect_tab_action = QAction("Detection Tab", self)
        detect_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        view_menu.addAction(detect_tab_action)
        
        heatmap_tab_action = QAction("Heatmap Tab", self)
        heatmap_tab_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        view_menu.addAction(heatmap_tab_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = self.statusBar()
        self.statusbar.showMessage("Ready")
    
    def _load_calibration_for_detection(self):
        """Load calibration and switch to detection tab"""
        self.detection_tab._load_calibration()
        self.tabs.setCurrentIndex(1)
    
    def _load_detections_for_heatmap(self):
        """Load detections and switch to heatmap tab"""
        self.heatmap_tab._load_detections()
        self.tabs.setCurrentIndex(2)
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Traffic Heatmap",
            "Traffic Heatmap - Multi-Camera Analysis System\n\n"
            "Features:\n"
            "- Multi-camera calibration with homography\n"
            "- Real-time person detection\n"
            "- Heatmap visualization\n\n"
            "Version 1.0"
        )
    
    def closeEvent(self, event):
        """Handle window close"""
        self.detection_tab.cleanup()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set dark theme
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = TrafficHeatmapApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
