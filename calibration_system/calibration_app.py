"""
Main Calibration Application
Qt-based GUI for multi-camera homography calibration
"""

import sys
import os
import cv2
import numpy as np
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QPushButton, QLabel, QLineEdit, QSpinBox,
    QTabWidget, QFileDialog, QMessageBox, QGroupBox, QScrollArea,
    QComboBox, QProgressBar, QStatusBar, QMenuBar, QMenu,
    QDialog, QFormLayout, QDialogButtonBox, QTextEdit, QSplitter,
    QFrame, QListWidget, QListWidgetItem, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QFont, QPixmap, QColor

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration_system.config import CalibrationConfig, CameraConfig
from calibration_system.camera_manager import CameraManager
from calibration_system.homography import HomographyCalculator
from calibration_system.storage import CalibrationStorage
from calibration_system.widgets import PointSelectorWidget, CameraPreviewWidget


class SetupDialog(QDialog):
    """Initial setup dialog for store and cameras configuration"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Setup")
        self.setMinimumWidth(600)
        self.config = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Store info
        store_group = QGroupBox("Store Information")
        store_layout = QFormLayout()
        
        self.store_name_edit = QLineEdit()
        self.store_name_edit.setPlaceholderText("Enter store name")
        store_layout.addRow("Store Name:", self.store_name_edit)
        
        plan_layout = QHBoxLayout()
        self.plan_path_edit = QLineEdit()
        self.plan_path_edit.setPlaceholderText("Select floor plan image...")
        self.plan_browse_btn = QPushButton("Browse...")
        self.plan_browse_btn.clicked.connect(self._browse_plan)
        plan_layout.addWidget(self.plan_path_edit)
        plan_layout.addWidget(self.plan_browse_btn)
        store_layout.addRow("Floor Plan:", plan_layout)
        
        store_group.setLayout(store_layout)
        layout.addWidget(store_group)
        
        # Camera setup - NEW TEMPLATE APPROACH
        camera_group = QGroupBox("Camera Configuration")
        camera_layout = QVBoxLayout()
        
        # RTSP URL Template
        template_label = QLabel("RTSP URL Template (use {camera_id} as placeholder):")
        camera_layout.addWidget(template_label)
        
        self.url_template_edit = QLineEdit()
        self.url_template_edit.setPlaceholderText("rtsp://user:pass@ip:port/cam/realmonitor?channel={camera_id}&subtype=1")
        self.url_template_edit.setText("rtsp://user:pass@ip:port/cam/realmonitor?channel={camera_id}&subtype=1")
        camera_layout.addWidget(self.url_template_edit)
        
        camera_layout.addSpacing(10)
        
        # Camera IDs to calibrate
        ids_label = QLabel("Camera IDs to calibrate (comma-separated):")
        camera_layout.addWidget(ids_label)
        
        self.camera_ids_edit = QLineEdit()
        self.camera_ids_edit.setPlaceholderText("e.g., 6,7,10,11,12,22")
        camera_layout.addWidget(self.camera_ids_edit)
        
        camera_layout.addSpacing(10)
        
        # Calibration points per camera
        points_layout = QHBoxLayout()
        points_label = QLabel("Calibration points per camera:")
        points_layout.addWidget(points_label)
        
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setMinimum(4)
        self.points_spinbox.setMaximum(20)
        self.points_spinbox.setValue(6)
        self.points_spinbox.setToolTip("Minimum 4 points required. More points = better accuracy (recommended: 6-10)")
        points_layout.addWidget(self.points_spinbox)
        
        points_layout.addStretch()
        camera_layout.addLayout(points_layout)
        
        # Points recommendation label
        points_info = QLabel(
            "Tip: 4 points minimum, 6-10 recommended for better accuracy.\n"
            "Points must be identifiable in both camera view and floor plan."
        )
        points_info.setStyleSheet("color: #888; font-size: 11px;")
        points_info.setWordWrap(True)
        camera_layout.addWidget(points_info)
        
        camera_layout.addSpacing(10)
        
        # Preview of generated URLs
        preview_btn = QPushButton("Preview URLs")
        preview_btn.clicked.connect(self._preview_urls)
        camera_layout.addWidget(preview_btn)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setPlaceholderText("Click 'Preview URLs' to see generated camera URLs...")
        camera_layout.addWidget(self.preview_text)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _browse_plan(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Floor Plan Image",
            "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filepath:
            self.plan_path_edit.setText(filepath)
    
    def _get_camera_ids(self):
        """Parse camera IDs from input"""
        ids_text = self.camera_ids_edit.text().strip()
        if not ids_text:
            return []
        
        try:
            ids = [int(x.strip()) for x in ids_text.split(",") if x.strip()]
            return ids
        except ValueError:
            return []
    
    def _preview_urls(self):
        """Preview generated URLs"""
        template = self.url_template_edit.text().strip()
        camera_ids = self._get_camera_ids()
        
        if not template:
            self.preview_text.setText("Please enter a URL template.")
            return
        
        if not camera_ids:
            self.preview_text.setText("Please enter camera IDs (e.g., 6,7,10,11,12).")
            return
        
        if "{camera_id}" not in template:
            self.preview_text.setText("Warning: Template does not contain {camera_id} placeholder.\n\nAll cameras will use the same URL.")
        
        preview = ""
        for cam_id in camera_ids:
            url = template.replace("{camera_id}", str(cam_id))
            preview += f"Camera {cam_id}: {url}\n"
        
        self.preview_text.setText(preview)
    
    def _on_accept(self):
        """Validate and create configuration"""
        store_name = self.store_name_edit.text().strip()
        plan_path = self.plan_path_edit.text().strip()
        template = self.url_template_edit.text().strip()
        camera_ids = self._get_camera_ids()
        required_points = self.points_spinbox.value()
        
        if not store_name:
            QMessageBox.warning(self, "Validation Error", "Please enter a store name.")
            return
        
        if not plan_path or not os.path.exists(plan_path):
            QMessageBox.warning(self, "Validation Error", "Please select a valid floor plan image.")
            return
        
        if not template:
            QMessageBox.warning(self, "Validation Error", "Please enter an RTSP URL template.")
            return
        
        if not camera_ids:
            QMessageBox.warning(self, "Validation Error", "Please enter camera IDs to calibrate.")
            return
        
        # Create config
        self.config = CalibrationConfig(
            store_name=store_name,
            store_plan_path=plan_path,
            num_cameras=0,
            min_calibration_points=required_points
        )
        self.config.cameras = []
        
        for cam_id in camera_ids:
            url = template.replace("{camera_id}", str(cam_id))
            cam = CameraConfig(
                camera_id=cam_id,
                name=f"Camera {cam_id}",
                rtsp_url=url,
                required_points=required_points
            )
            self.config.cameras.append(cam)
        
        self.config.num_cameras = len(self.config.cameras)
        self.accept()
    
    def get_config(self) -> Optional[CalibrationConfig]:
        return self.config


class CalibrationMainWindow(QMainWindow):
    """Main calibration application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Heatmap - Camera Calibration System")
        self.setMinimumSize(1600, 1000)
        self.showMaximized()  # Start maximized for bigger display
        
        self.config: Optional[CalibrationConfig] = None
        self.camera_manager = CameraManager(preview_width=800, preview_height=600)  # Bigger preview
        self.homography_calc = HomographyCalculator()
        self.storage = CalibrationStorage()
        
        self.current_camera_index = 0  # Index in config.cameras list
        self.current_camera_id = 0     # Actual camera ID for camera_manager
        self.camera_widgets = {}
        self.plan_widget = None
        
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        
        # Timer for camera preview
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self._update_preview)
    
    def _setup_ui(self):
        """Setup main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Welcome screen (shown initially)
        self.welcome_widget = self._create_welcome_widget()
        main_layout.addWidget(self.welcome_widget)
        
        # Main calibration interface (hidden initially)
        self.calibration_widget = QWidget()
        self.calibration_widget.hide()
        main_layout.addWidget(self.calibration_widget)
        
        self._setup_calibration_ui()
    
    def _create_welcome_widget(self) -> QWidget:
        """Create welcome/start screen"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Title
        title = QLabel("Camera Calibration System")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Multi-camera homography calibration for traffic heatmap generation")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(40)
        
        # Buttons
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        new_btn = QPushButton("New Calibration")
        new_btn.setMinimumSize(200, 50)
        new_btn.setFont(QFont("Arial", 12))
        new_btn.clicked.connect(self._new_calibration)
        btn_layout.addWidget(new_btn)
        
        load_btn = QPushButton("Load Existing Calibration")
        load_btn.setMinimumSize(200, 50)
        load_btn.setFont(QFont("Arial", 12))
        load_btn.clicked.connect(self._load_calibration)
        btn_layout.addWidget(load_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _setup_calibration_ui(self):
        """Setup the main calibration interface"""
        layout = QHBoxLayout(self.calibration_widget)
        
        # Left panel - Camera list and controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Store info
        info_group = QGroupBox("Project Info")
        info_layout = QVBoxLayout()
        self.store_label = QLabel("Store: -")
        self.cameras_count_label = QLabel("Cameras: -")
        self.status_label = QLabel("Status: Not started")
        info_layout.addWidget(self.store_label)
        info_layout.addWidget(self.cameras_count_label)
        info_layout.addWidget(self.status_label)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        
        # Camera list
        cameras_group = QGroupBox("Cameras")
        cameras_layout = QVBoxLayout()
        self.camera_list = QListWidget()
        self.camera_list.currentRowChanged.connect(self._on_camera_selected)
        cameras_layout.addWidget(self.camera_list)
        
        cam_btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_current_camera)
        cam_btn_layout.addWidget(self.connect_btn)
        
        self.connect_all_btn = QPushButton("Connect All")
        self.connect_all_btn.clicked.connect(self._connect_all_cameras)
        cam_btn_layout.addWidget(self.connect_all_btn)
        cameras_layout.addLayout(cam_btn_layout)
        
        cameras_group.setLayout(cameras_layout)
        left_layout.addWidget(cameras_group)
        
        # Calibration controls
        calib_group = QGroupBox("Calibration")
        calib_layout = QVBoxLayout()
        
        self.calc_homography_btn = QPushButton("Calculate Homography")
        self.calc_homography_btn.clicked.connect(self._calculate_homography)
        calib_layout.addWidget(self.calc_homography_btn)
        
        self.calc_all_btn = QPushButton("Calculate All")
        self.calc_all_btn.clicked.connect(self._calculate_all_homographies)
        calib_layout.addWidget(self.calc_all_btn)
        
        self.save_btn = QPushButton("Save Calibration")
        self.save_btn.clicked.connect(self._save_calibration)
        calib_layout.addWidget(self.save_btn)
        
        self.export_btn = QPushButton("Export Matrices")
        self.export_btn.clicked.connect(self._export_matrices)
        calib_layout.addWidget(self.export_btn)
        
        calib_group.setLayout(calib_layout)
        left_layout.addWidget(calib_group)
        
        left_layout.addStretch()
        layout.addWidget(left_panel)
        
        # Center panel - Camera view and Floor plan side by side
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Camera view
        camera_container = QWidget()
        camera_layout = QVBoxLayout(camera_container)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        
        self.camera_title = QLabel("Camera View")
        self.camera_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        camera_layout.addWidget(self.camera_title)
        
        self.camera_widget = CameraPreviewWidget(
            title="",
            point_color="#00FF00",
            point_radius=12
        )
        self.camera_widget.setMinimumSize(700, 500)  # Bigger minimum size
        self.camera_widget.points_changed.connect(self._on_camera_points_changed)
        self.camera_widget.capture_requested.connect(self._capture_camera_frame)
        camera_layout.addWidget(self.camera_widget)
        
        splitter.addWidget(camera_container)
        
        # Floor plan view
        plan_container = QWidget()
        plan_layout = QVBoxLayout(plan_container)
        plan_layout.setContentsMargins(0, 0, 0, 0)
        
        plan_title = QLabel("Floor Plan")
        plan_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        plan_layout.addWidget(plan_title)
        
        self.plan_widget = PointSelectorWidget(
            title="",
            point_color="#FF0000",
            point_radius=12
        )
        self.plan_widget.setMinimumSize(700, 500)  # Bigger minimum size
        self.plan_widget.points_changed.connect(self._on_plan_points_changed)
        plan_layout.addWidget(self.plan_widget)
        
        splitter.addWidget(plan_container)
        
        # Set initial splitter sizes (50-50 split)
        splitter.setSizes([800, 800])
        
        layout.addWidget(splitter, 1)
        
        # Right panel - Results and info
        right_panel = QWidget()
        right_panel.setMaximumWidth(300)
        right_panel.setMinimumWidth(250)
        right_layout = QVBoxLayout(right_panel)
        
        # Points info
        points_group = QGroupBox("Calibration Points")
        points_layout = QVBoxLayout()
        self.points_text = QTextEdit()
        self.points_text.setReadOnly(True)
        self.points_text.setMaximumHeight(200)
        points_layout.addWidget(self.points_text)
        points_group.setLayout(points_layout)
        right_layout.addWidget(points_group)
        
        # Homography info
        matrix_group = QGroupBox("Homography Matrix")
        matrix_layout = QVBoxLayout()
        self.matrix_text = QTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setFont(QFont("Courier", 10))
        self.matrix_text.setMaximumHeight(150)
        matrix_layout.addWidget(self.matrix_text)
        
        self.error_label = QLabel("Reprojection Error: -")
        matrix_layout.addWidget(self.error_label)
        
        matrix_group.setLayout(matrix_layout)
        right_layout.addWidget(matrix_group)
        
        # Instructions
        help_group = QGroupBox("Instructions")
        help_layout = QVBoxLayout()
        help_text = QLabel(
            "1. Select a camera from the list\n"
            "2. Click 'Connect' to view camera feed\n"
            "3. Click 'Capture Frame' to freeze\n"
            "4. Click 4+ points on camera view\n"
            "5. Click corresponding points on floor plan\n"
            "6. Click 'Calculate Homography'\n"
            "7. Repeat for all cameras\n"
            "8. Save calibration when complete"
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text)
        help_group.setLayout(help_layout)
        right_layout.addWidget(help_group)
        
        right_layout.addStretch()
        layout.addWidget(right_panel)
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Calibration", self)
        new_action.triggered.connect(self._new_calibration)
        file_menu.addAction(new_action)
        
        load_action = QAction("Load Calibration", self)
        load_action.triggered.connect(self._load_calibration)
        file_menu.addAction(load_action)
        
        save_action = QAction("Save Calibration", self)
        save_action.triggered.connect(self._save_calibration)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Homography Matrices", self)
        export_action.triggered.connect(self._export_matrices)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera menu
        camera_menu = menubar.addMenu("Camera")
        
        connect_action = QAction("Connect Current Camera", self)
        connect_action.triggered.connect(self._connect_current_camera)
        camera_menu.addAction(connect_action)
        
        connect_all_action = QAction("Connect All Cameras", self)
        connect_all_action.triggered.connect(self._connect_all_cameras)
        camera_menu.addAction(connect_all_action)
        
        disconnect_action = QAction("Disconnect All", self)
        disconnect_action.triggered.connect(self._disconnect_all_cameras)
        camera_menu.addAction(disconnect_action)
    
    def _setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
    
    def _new_calibration(self):
        """Start new calibration setup"""
        dialog = SetupDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.config = dialog.get_config()
            self._initialize_calibration()
    
    def _initialize_calibration(self):
        """Initialize calibration with current config"""
        if self.config is None:
            return
        
        # Switch to calibration view
        self.welcome_widget.hide()
        self.calibration_widget.show()
        
        # Update info labels
        self.store_label.setText(f"Store: {self.config.store_name}")
        self.cameras_count_label.setText(f"Cameras: {self.config.num_cameras}")
        self._update_status()
        
        # Populate camera list
        self.camera_list.clear()
        for cam in self.config.cameras:
            status = "calibrated" if cam.is_calibrated() else "pending"
            item = QListWidgetItem(f"{cam.name} [{status}]")
            self.camera_list.addItem(item)
            
            # Add to camera manager
            self.camera_manager.add_camera(cam.camera_id, cam.rtsp_url, cam.name)
        
        # Load floor plan
        if self.config.store_plan_path and os.path.exists(self.config.store_plan_path):
            self.plan_widget.load_image_file(self.config.store_plan_path)
        
        # Select first camera
        if self.config.cameras:
            self.camera_list.setCurrentRow(0)
        
        self.statusbar.showMessage("Calibration initialized")
    
    def _load_calibration(self):
        """Load existing calibration"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration",
            "./calibration_data", "JSON Files (*.json)"
        )
        if filepath:
            self.config = self.storage.load_calibration(filepath)
            if self.config:
                self._initialize_calibration()
                # Restore points for current camera
                self._load_camera_points()
                self.statusbar.showMessage(f"Loaded: {filepath}")
    
    def _save_calibration(self):
        """Save current calibration"""
        if self.config is None:
            QMessageBox.warning(self, "Error", "No calibration to save.")
            return
        
        filepath = self.storage.save_calibration(self.config)
        QMessageBox.information(self, "Saved", f"Calibration saved to:\n{filepath}")
    
    def _export_matrices(self):
        """Export homography matrices"""
        if self.config is None:
            QMessageBox.warning(self, "Error", "No calibration data.")
            return
        
        filepath = self.storage.export_homography_matrices(self.config)
        QMessageBox.information(self, "Exported", f"Matrices exported to:\n{filepath}")
    
    def _on_camera_selected(self, row: int):
        """Handle camera selection change"""
        if row < 0 or self.config is None:
            return
        
        # Save current camera points before switching
        self._save_camera_points()
        
        # Stop preview timer when switching cameras
        self.preview_timer.stop()
        
        # Get the actual camera from config (row is the list index)
        self.current_camera_index = row
        cam = self.config.cameras[row]
        self.current_camera_id = cam.camera_id  # Use actual camera_id
        
        self.camera_title.setText(f"Camera: {cam.name} (ID: {cam.camera_id}) - {cam.required_points} points required")
        
        # Set required points for this camera on both widgets
        self.camera_widget.set_required_points(cam.required_points)
        self.plan_widget.set_required_points(cam.required_points)
        
        # Clear the camera widget display (will show new camera when connected)
        self.camera_widget.clear_points()
        
        # Load saved points for this camera
        self._load_camera_points()
        
        # Update matrix display
        self._update_matrix_display()
        
        # Auto-connect to newly selected camera
        self._connect_current_camera()
    
    def _save_camera_points(self):
        """Save current points to config"""
        if self.config is None or not hasattr(self, 'current_camera_index'):
            return
        
        if self.current_camera_index >= len(self.config.cameras):
            return
            
        cam = self.config.cameras[self.current_camera_index]
        cam.calibration_points_camera = self.camera_widget.get_points()
        cam.calibration_points_plan = self.plan_widget.get_points()
    
    def _load_camera_points(self):
        """Load points for current camera"""
        if self.config is None or not hasattr(self, 'current_camera_index'):
            return
        
        if self.current_camera_index >= len(self.config.cameras):
            return
            
        cam = self.config.cameras[self.current_camera_index]
        
        # Load camera points
        if cam.calibration_points_camera:
            self.camera_widget.set_points(cam.calibration_points_camera)
        else:
            self.camera_widget.clear_points()
        
        # Load plan points
        if cam.calibration_points_plan:
            self.plan_widget.set_points(cam.calibration_points_plan)
        else:
            self.plan_widget.clear_points()
        
        self._update_points_display()
    
    def _connect_current_camera(self):
        """Connect to currently selected camera"""
        if self.config is None or not hasattr(self, 'current_camera_index'):
            return
        
        if self.current_camera_index >= len(self.config.cameras):
            return
            
        cam = self.config.cameras[self.current_camera_index]
        self.statusbar.showMessage(f"Connecting to {cam.name}...")
        
        if self.camera_manager.connect_camera(cam.camera_id):
            self.current_camera_id = cam.camera_id  # Track actual camera ID for frame retrieval
            self.statusbar.showMessage(f"Connected to {cam.name}")
            # Start preview
            self.preview_timer.start(33)  # ~30 FPS
        else:
            QMessageBox.warning(self, "Connection Error", f"Failed to connect to {cam.name}")
            self.statusbar.showMessage("Connection failed")
    
    def _connect_all_cameras(self):
        """Connect all cameras"""
        if self.config is None:
            return
        
        self.statusbar.showMessage("Connecting to all cameras...")
        results = self.camera_manager.connect_all()
        
        connected = sum(1 for v in results.values() if v)
        self.statusbar.showMessage(f"Connected: {connected}/{len(results)} cameras")
    
    def _disconnect_all_cameras(self):
        """Disconnect all cameras"""
        self.preview_timer.stop()
        self.camera_manager.disconnect_all()
        self.statusbar.showMessage("All cameras disconnected")
    
    def _update_preview(self):
        """Update camera preview"""
        frame = self.camera_manager.get_frame(self.current_camera_id)
        if frame is not None:
            self.camera_widget.set_image(frame)
    
    def _capture_camera_frame(self):
        """Capture current frame for calibration"""
        self.preview_timer.stop()
        frame = self.camera_manager.capture_frame(self.current_camera_id)
        if frame is not None:
            self.camera_widget.set_image(frame)
            # Save snapshot using camera name for better organization
            if self.current_camera_index < len(self.config.cameras):
                cam_name = self.config.cameras[self.current_camera_index].name
                self.storage.save_camera_snapshot(cam_name, frame)
            else:
                self.storage.save_camera_snapshot(self.current_camera_id, frame)
            self.statusbar.showMessage("Frame captured")
    
    def _on_camera_points_changed(self, points):
        """Handle camera points change"""
        self._update_points_display()
    
    def _on_plan_points_changed(self, points):
        """Handle plan points change"""
        self._update_points_display()
    
    def _update_points_display(self):
        """Update points text display"""
        cam_pts = self.camera_widget.get_points()
        plan_pts = self.plan_widget.get_points()
        
        text = "Camera Points:\n"
        for i, pt in enumerate(cam_pts):
            text += f"  {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})\n"
        
        text += "\nFloor Plan Points:\n"
        for i, pt in enumerate(plan_pts):
            text += f"  {i+1}: ({pt[0]:.1f}, {pt[1]:.1f})\n"
        
        min_pts = min(len(cam_pts), len(plan_pts))
        text += f"\nMatched pairs: {min_pts}"
        if min_pts < 4:
            text += f" (need {4 - min_pts} more)"
        
        self.points_text.setText(text)
    
    def _calculate_homography(self):
        """Calculate homography for current camera"""
        if self.config is None:
            return
        
        cam_pts = self.camera_widget.get_points()
        plan_pts = self.plan_widget.get_points()
        
        if len(cam_pts) < 4 or len(plan_pts) < 4:
            QMessageBox.warning(self, "Error", "Need at least 4 points on both views.")
            return
        
        if len(cam_pts) != len(plan_pts):
            QMessageBox.warning(self, "Error", "Number of points must match on both views.")
            return
        
        # Calculate homography
        result = self.homography_calc.calculate_homography(
            [(p[0], p[1]) for p in cam_pts],
            [(p[0], p[1]) for p in plan_pts]
        )
        
        if not result.is_valid:
            QMessageBox.warning(self, "Error", f"Homography calculation failed:\n{result.error_message}")
            return
        
        # Save to config
        if self.current_camera_index < len(self.config.cameras):
            cam = self.config.cameras[self.current_camera_index]
            cam.calibration_points_camera = cam_pts
            cam.calibration_points_plan = plan_pts
            cam.homography_matrix = self.homography_calc.matrix_to_list(result.matrix)
        
        # Update display
        self._update_matrix_display()
        self._update_camera_list()
        self._update_status()
        
        self.statusbar.showMessage(
            f"Homography calculated (error: {result.reprojection_error:.3f}px)"
        )
    
    def _calculate_all_homographies(self):
        """Calculate homography for all cameras with sufficient points"""
        if self.config is None:
            return
        
        calculated = 0
        for cam in self.config.cameras:
            if (len(cam.calibration_points_camera) >= 4 and 
                len(cam.calibration_points_plan) >= 4):
                
                result = self.homography_calc.calculate_homography(
                    [(p[0], p[1]) for p in cam.calibration_points_camera],
                    [(p[0], p[1]) for p in cam.calibration_points_plan]
                )
                
                if result.is_valid:
                    cam.homography_matrix = self.homography_calc.matrix_to_list(result.matrix)
                    calculated += 1
        
        self._update_camera_list()
        self._update_status()
        self.statusbar.showMessage(f"Calculated homography for {calculated} cameras")
    
    def _update_matrix_display(self):
        """Update homography matrix display"""
        if self.config is None or not hasattr(self, 'current_camera_index'):
            return
        
        if self.current_camera_index >= len(self.config.cameras):
            return
            
        cam = self.config.cameras[self.current_camera_index]
        
        if cam.homography_matrix:
            H = np.array(cam.homography_matrix)
            text = ""
            for row in H:
                text += "  ".join(f"{v:10.4f}" for v in row) + "\n"
            self.matrix_text.setText(text)
            
            # Calculate and show error
            if cam.calibration_points_camera and cam.calibration_points_plan:
                result = self.homography_calc.calculate_homography(
                    [(p[0], p[1]) for p in cam.calibration_points_camera],
                    [(p[0], p[1]) for p in cam.calibration_points_plan]
                )
                self.error_label.setText(f"Reprojection Error: {result.reprojection_error:.3f} px")
        else:
            self.matrix_text.setText("Not calculated yet")
            self.error_label.setText("Reprojection Error: -")
    
    def _update_camera_list(self):
        """Update camera list status"""
        if self.config is None:
            return
        
        for i, cam in enumerate(self.config.cameras):
            current, required = cam.calibration_progress()
            if cam.is_calibrated():
                status = "calibrated"
            elif current > 0:
                status = f"{current}/{required} pts"
            else:
                status = "pending"
            self.camera_list.item(i).setText(f"{cam.name} [{status}]")
    
    def _update_status(self):
        """Update overall calibration status"""
        if self.config is None:
            return
        
        summary = self.config.get_calibration_summary()
        status = f"Status: {summary['calibrated']}/{summary['total_cameras']} calibrated"
        if summary['complete']:
            status += " (Complete)"
        self.status_label.setText(status)
    
    def closeEvent(self, event):
        """Handle window close"""
        self.preview_timer.stop()
        self.camera_manager.disconnect_all()
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
    
    window = CalibrationMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
