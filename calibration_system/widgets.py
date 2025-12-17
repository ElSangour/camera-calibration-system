"""
Qt Widgets for Calibration System
Interactive point selection on images
"""

from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt6.QtGui import (
    QPixmap, QPainter, QPen, QColor, QBrush, 
    QImage, QMouseEvent, QPaintEvent
)
import numpy as np
from typing import List, Tuple, Optional


class ClickableImageLabel(QLabel):
    """
    QLabel that emits click positions for point selection
    """
    clicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(event)


class PointSelectorWidget(QWidget):
    """
    Widget for selecting and displaying calibration points on an image
    Supports both camera frames and floor plan images
    """
    
    points_changed = pyqtSignal(list)  # Emits list of points when changed
    point_selected = pyqtSignal(int, int)  # Emits when a new point is selected
    points_complete = pyqtSignal()  # Emits when required points are reached
    
    def __init__(
        self,
        title: str = "Image",
        point_color: str = "#00FF00",
        point_radius: int = 8,
        max_points: int = 20,
        required_points: int = 4,
        parent=None
    ):
        super().__init__(parent)
        
        self.title = title
        self.point_color = QColor(point_color)
        self.point_radius = point_radius
        self.max_points = max_points
        self.required_points = required_points
        
        self.points: List[Tuple[int, int]] = []
        self.original_image: Optional[QImage] = None
        self.display_scale = 1.0
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)
        
        # Image display - BIGGER SIZE
        self.image_label = ClickableImageLabel()
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #444; background-color: #222;")
        self.image_label.clicked.connect(self._on_click)
        layout.addWidget(self.image_label, 1)  # Stretch factor 1
        
        # Info and controls
        controls_layout = QHBoxLayout()
        
        self.info_label = QLabel("Points: 0")
        controls_layout.addWidget(self.info_label)
        
        controls_layout.addStretch()
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.remove_last_point)
        controls_layout.addWidget(self.undo_btn)
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_points)
        controls_layout.addWidget(self.clear_btn)
        
        layout.addLayout(controls_layout)
    
    def set_image(self, image: np.ndarray):
        """
        Set image from numpy array (OpenCV format BGR)
        """
        if image is None:
            return
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = np.ascontiguousarray(image[:, :, ::-1])
        
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        
        self.original_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        ).copy()
        
        self._update_display()
    
    def load_image_file(self, filepath: str) -> bool:
        """Load image from file path"""
        self.original_image = QImage(filepath)
        if self.original_image.isNull():
            print(f"[ERROR] Failed to load image: {filepath}")
            return False
        
        self._update_display()
        return True
    
    def _update_display(self):
        """Update the displayed image with points"""
        if self.original_image is None:
            return
        
        # Create a copy to draw on
        display_image = self.original_image.copy()
        
        # Draw points
        painter = QPainter(display_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Point style
        pen = QPen(self.point_color)
        pen.setWidth(2)
        painter.setPen(pen)
        brush = QBrush(self.point_color)
        painter.setBrush(brush)
        
        # Draw each point with number
        for i, (x, y) in enumerate(self.points):
            # Draw filled circle
            painter.setBrush(brush)
            painter.drawEllipse(
                QPoint(x, y),
                self.point_radius,
                self.point_radius
            )
            
            # Draw point number
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(
                x + self.point_radius + 2,
                y + self.point_radius // 2,
                str(i + 1)
            )
            painter.setPen(pen)
        
        # Draw lines connecting points
        if len(self.points) > 1:
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]
                painter.drawLine(x1, y1, x2, y2)
            
            # Connect last to first if 4+ points
            if len(self.points) >= 4:
                x1, y1 = self.points[-1]
                x2, y2 = self.points[0]
                painter.drawLine(x1, y1, x2, y2)
        
        painter.end()
        
        # Scale to fit widget
        label_size = self.image_label.size()
        pixmap = QPixmap.fromImage(display_image)
        scaled_pixmap = pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Calculate scale factor for coordinate conversion
        self.display_scale = min(
            label_size.width() / self.original_image.width(),
            label_size.height() / self.original_image.height()
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self._update_info()
    
    def _on_click(self, x: int, y: int):
        """Handle click on image"""
        if self.original_image is None:
            return
        
        if len(self.points) >= self.max_points:
            print(f"[WARN] Maximum points ({self.max_points}) reached")
            return
        
        # Convert display coordinates to image coordinates
        # Account for centering in label
        label_size = self.image_label.size()
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
        
        # Calculate offset due to centering
        offset_x = (label_size.width() - pixmap.width()) // 2
        offset_y = (label_size.height() - pixmap.height()) // 2
        
        # Adjust coordinates
        img_x = int((x - offset_x) / self.display_scale)
        img_y = int((y - offset_y) / self.display_scale)
        
        # Bounds check
        if img_x < 0 or img_x >= self.original_image.width():
            return
        if img_y < 0 or img_y >= self.original_image.height():
            return
        
        self.points.append((img_x, img_y))
        self._update_display()
        self.points_changed.emit(self.get_points())
        self.point_selected.emit(img_x, img_y)
        
        # Check if required points reached
        if len(self.points) == self.required_points:
            self.points_complete.emit()
    
    def add_point(self, x: int, y: int):
        """Programmatically add a point"""
        if len(self.points) < self.max_points:
            self.points.append((x, y))
            self._update_display()
            self.points_changed.emit(self.get_points())
    
    def remove_last_point(self):
        """Remove the last added point"""
        if self.points:
            self.points.pop()
            self._update_display()
            self.points_changed.emit(self.get_points())
    
    def clear_points(self):
        """Clear all points"""
        self.points = []
        self._update_display()
        self.points_changed.emit(self.get_points())
    
    def get_points(self) -> List[List[float]]:
        """Get points as list of [x, y] lists"""
        return [[float(x), float(y)] for x, y in self.points]
    
    def set_points(self, points: List[List[float]]):
        """Set points from list"""
        self.points = [(int(p[0]), int(p[1])) for p in points]
        self._update_display()
    
    def set_required_points(self, count: int):
        """Set the number of required calibration points"""
        self.required_points = max(4, min(count, self.max_points))
        self._update_info()
    
    def get_progress(self) -> tuple:
        """Return (current_points, required_points)"""
        return (len(self.points), self.required_points)
    
    def is_complete(self) -> bool:
        """Check if required points have been selected"""
        return len(self.points) >= self.required_points
    
    def _update_info(self):
        """Update info label with progress"""
        count = len(self.points)
        if count >= self.required_points:
            status = "Complete"
            self.info_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        else:
            remaining = self.required_points - count
            status = f"Need {remaining} more"
            self.info_label.setStyleSheet("color: #FFAA00;")
        self.info_label.setText(f"Points: {count}/{self.required_points} ({status})")
    
    def set_point_color(self, color: str):
        """Set point color"""
        self.point_color = QColor(color)
        self._update_display()
    
    def resizeEvent(self, event):
        """Handle resize"""
        super().resizeEvent(event)
        self._update_display()


class CameraPreviewWidget(PointSelectorWidget):
    """
    Extended point selector with live camera preview capability
    """
    
    capture_requested = pyqtSignal()  # Request frame capture
    
    def __init__(self, camera_id: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.camera_id = camera_id
        self.is_live = False
        
        # Add capture button
        self.capture_btn = QPushButton("Capture Frame")
        self.capture_btn.clicked.connect(self._on_capture)
        self.layout().insertWidget(2, self.capture_btn)
    
    def _on_capture(self):
        """Handle capture button click"""
        self.capture_requested.emit()
    
    def update_frame(self, frame: np.ndarray):
        """Update with new frame (for live preview)"""
        if self.is_live:
            self.set_image(frame)
    
    def set_live_mode(self, enabled: bool):
        """Enable/disable live preview mode"""
        self.is_live = enabled
        self.capture_btn.setEnabled(not enabled)
        if enabled:
            self.clear_points()


if __name__ == "__main__":
    # Test widget
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    widget = PointSelectorWidget(title="Test Image", point_color="#FF0000")
    widget.resize(800, 600)
    
    # Create test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (50, 50, 50)  # Gray background
    widget.set_image(test_img)
    
    widget.show()
    sys.exit(app.exec())
