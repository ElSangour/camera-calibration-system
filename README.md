# Traffic Heatmap - Multi-Camera Vision System

Real-time customer traffic analysis system that processes multiple RTSP camera streams to generate unified heatmaps on a 2D floor plan using computer vision and homography transformation.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Usage](#usage)
8. [Integrated Application](#integrated-application)
9. [Calibration System](#calibration-system)
10. [Multi-Camera Detection](#multi-camera-detection)
11. [Heatmap Generation](#heatmap-generation)
12. [API Reference](#api-reference)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Overview

This system transforms multiple camera feeds into actionable analytics by:

- Detecting people in multiple simultaneous RTSP camera streams using YOLOv8
- Transforming camera coordinates to a unified 2D floor plan using homography matrices
- Generating density heatmaps showing traffic patterns
- Providing an integrated Qt application for complete workflow management
- Exporting JSON data for visualization and analysis

## Features

### Core Capabilities

- [x] RTSP camera stream connection and management
- [x] YOLOv8 person detection with configurable models
- [x] Multi-camera calibration system with dynamic point selection
- [x] Homography matrix calculation and transformation
- [x] Multi-threaded concurrent camera processing
- [x] Real-time coordinate transformation from camera to floor plan
- [x] Density heatmap generation with Gaussian smoothing
- [x] Integrated PyQt6 application with unified interface

### Application Modules

| Module | Description |
|--------|-------------|
| Calibration System | Qt-based GUI for camera-to-floor-plan point mapping |
| Multi-Camera Detector | Concurrent YOLO detection across all calibrated cameras |
| Heatmap Generator | Density visualization from aggregated detection data |
| Main Application | Integrated interface combining all modules |

## Use Cases

- Retail store traffic analysis
- Queue management optimization
- Facility layout effectiveness studies
- Peak hours identification
- Pedestrian flow analysis
- Space utilization monitoring

---

## Project Structure

\`\`\`
Traffic-Heatmap/
|-- main_app.py                 # Integrated PyQt6 application
|
|-- calibration_data/           # Stored calibration files and snapshots
|   |-- snapshots/              # Camera frame snapshots
|   |-- *.json                  # Calibration and homography matrix files
|
|-- calibration_system/         # Qt-based calibration module
|   |-- __init__.py
|   |-- calibration_app.py      # Standalone calibration GUI
|   |-- camera_manager.py       # RTSP camera connection management
|   |-- config.py               # Configuration data classes
|   |-- homography.py           # Homography matrix calculations
|   |-- storage.py              # JSON storage for calibration data
|   |-- widgets.py              # Custom Qt widgets for point selection
|
|-- models/                     # YOLO model management
|   |-- __init__.py
|   |-- scripts/
|   |   |-- __init__.py
|   |   |-- config.py           # Model configuration
|   |   |-- yolo_loader.py      # YOLO model loading utilities
|   |-- weights/                # Model weight files
|
|-- src/                        # Source scripts
|   |-- multi_camera_detector.py    # Multi-threaded detection system
|   |-- heatmap_generator.py        # Density heatmap generation
|   |-- test_rtsp_connection.py     # RTSP connection testing
|   |-- yolo_person_detection.py    # Single camera detection
|   |-- output/                     # Detection output files
|
|-- output/                     # Generated output files
|   |-- detections_*.json       # Detection data exports
|   |-- heatmap_*.png           # Generated heatmap images
|
|-- .env                        # Environment configuration (not in repo)
|-- .env.example                # Environment template
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
|-- LICENSE                     # Project license
\`\`\`

---

## Requirements

### System Requirements

- Python 3.8 or higher
- Linux/Windows/macOS
- Network access to RTSP cameras
- GPU recommended for real-time processing (optional)

### Python Dependencies

\`\`\`
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
python-dotenv>=1.0.0
PyQt6>=6.5.0
scipy>=1.10.0
\`\`\`

---

## Installation

### 1. Clone the Repository

\`\`\`bash
git clone https://github.com/ElSangour/Traffic-Heatmap.git
cd Traffic-Heatmap
\`\`\`

### 2. Create Virtual Environment

\`\`\`bash
python -m venv .heatmap_env
source .heatmap_env/bin/activate  # Linux/macOS
# or
.heatmap_env\Scripts\activate     # Windows
\`\`\`

### 3. Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Download YOLO Model

The YOLOv8 model will be downloaded automatically on first run, or manually place it in \`models/weights/\`.

---

## Configuration

### Environment Variables

Copy the example environment file and configure it:

\`\`\`bash
cp .env.example .env
\`\`\`

Edit \`.env\` with your settings:

\`\`\`dotenv
# RTSP Camera Configuration
RTSP_URL=rtsp://username:password@camera_ip:554/stream
RTSP_USERNAME=admin
RTSP_PASSWORD=your_password

# YOLOv8 Model Configuration
YOLO_MODEL=yolov8s.pt
CONFIDENCE_THRESHOLD=0.5

# Floor Plan
FLOOR_PLAN_PATH=/path/to/floor_plan.png

# Output Settings
OUTPUT_DIR=./output
\`\`\`

### Available YOLO Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | Nano | Fastest | Good |
| yolov8s.pt | Small | Fast | Better |
| yolov8m.pt | Medium | Medium | High |
| yolov8l.pt | Large | Slow | Higher |
| yolov8x.pt | XLarge | Slowest | Highest |

---

## Usage

### Integrated Application (Recommended)

Launch the complete application with all features:

\`\`\`bash
python main_app.py
\`\`\`

The integrated application provides three tabs:

1. **Calibration** - Set up camera-to-floor-plan point mappings
2. **Live Detection** - Run multi-camera person detection
3. **Heatmap** - Generate and visualize density heatmaps

### Standalone Scripts

#### Test RTSP Connection

\`\`\`bash
python src/test_rtsp_connection.py
\`\`\`

#### Run Single Camera Detection

\`\`\`bash
python src/yolo_person_detection.py
\`\`\`

#### Run Multi-Camera Detection

\`\`\`bash
python src/multi_camera_detector.py \\
    --calibration calibration_data/homography_matrices_store.json \\
    --duration 300 \\
    --output output/detections.json
\`\`\`

#### Generate Heatmap

\`\`\`bash
python src/heatmap_generator.py \\
    -i output/detections.json \\
    -p /path/to/floor_plan.png \\
    -o output/heatmap.png \\
    --preview
\`\`\`

---

## Integrated Application

The main application (\`main_app.py\`) provides a unified interface for the complete workflow.

### Launching

\`\`\`bash
python main_app.py
\`\`\`

### Tab 1: Calibration

- Load existing calibration or create new configuration
- Configure RTSP URL template and camera IDs
- Select calibration points on camera view and floor plan
- Calculate and save homography matrices

### Tab 2: Live Detection

- Load calibration data with homography matrices
- Configure detection parameters (confidence, duration)
- Start/stop multi-camera detection
- Real-time status monitoring
- Export detection results to JSON

### Tab 3: Heatmap

- Load detection data from JSON files
- Import floor plan image
- Configure heatmap parameters (colormap, alpha, sigma)
- Generate and preview heatmap
- Save heatmap image

---

## Calibration System

The calibration system maps camera views to floor plan coordinates using homography transformation.

### Standalone Calibration App

\`\`\`bash
python calibration_system/calibration_app.py
\`\`\`

### Calibration Workflow

1. **Setup Configuration**
   - Enter location/store name
   - Provide RTSP URL template with \`{camera_id}\` placeholder
   - Enter camera IDs (comma-separated)
   - Select number of calibration points per camera (4-20)
   - Load floor plan image

2. **Point Selection**
   - Click corresponding points on camera view and floor plan
   - Progress indicator shows points remaining
   - Points should cover the visible floor area

3. **Calculate Homography**
   - Click "Calculate Homography" after selecting all points
   - Review reprojection error (aim for < 5 pixels)

4. **Save and Export**
   - Save complete calibration data
   - Export homography matrices for detection system

### Calibration Best Practices

- Use clearly identifiable floor markers or corners
- Distribute points across the entire visible floor area
- Avoid selecting points on vertical surfaces or furniture
- Use 6-8 points minimum for better accuracy
- Ensure corresponding points match exactly between views

### Output Files

\`\`\`
calibration_data/
|-- calibration_{name}_{timestamp}.json    # Full calibration data
|-- homography_matrices_{name}.json        # Matrices for detection
|-- snapshots/                             # Camera frame captures
\`\`\`

---

## Multi-Camera Detection

The detection system processes multiple RTSP streams concurrently using YOLO.

### Command Line Usage

\`\`\`bash
python src/multi_camera_detector.py \\
    --calibration calibration_data/homography_matrices_store.json \\
    --duration 600 \\
    --confidence 0.5 \\
    --output output/detections.json
\`\`\`

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| \`--calibration\` | Path to homography matrices JSON | Required |
| \`--duration\` | Detection duration in seconds | 300 |
| \`--confidence\` | YOLO confidence threshold | 0.5 |
| \`--output\` | Output JSON file path | Auto-generated |

### Detection Process

1. Loads calibration data with camera configurations
2. Initializes YOLO model for person detection
3. Spawns threads for each camera stream
4. Extracts foot points from bounding boxes (bottom-center)
5. Transforms coordinates using homography matrices
6. Aggregates detections with timestamps

### Output Format

\`\`\`json
{
  "metadata": {
    "start_time": "2025-12-17T14:00:00",
    "end_time": "2025-12-17T14:05:00",
    "cameras": ["1", "2", "3"],
    "total_detections": 7972
  },
  "detections": [
    {
      "timestamp": "2025-12-17T14:00:01.234",
      "camera_id": "1",
      "camera_point": [320, 450],
      "plan_point": [156.7, 234.2],
      "confidence": 0.87
    }
  ]
}
\`\`\`

---

## Heatmap Generation

Generate density heatmaps from detection data.

### Command Line Usage

\`\`\`bash
python src/heatmap_generator.py \\
    -i output/detections.json \\
    -p /path/to/floor_plan.png \\
    -o output/heatmap.png \\
    --colormap jet \\
    --alpha 0.6 \\
    --sigma 20 \\
    --preview
\`\`\`

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| \`-i, --input\` | Detection JSON file | Required |
| \`-p, --plan\` | Floor plan image path | Auto-detect |
| \`-o, --output\` | Output heatmap path | Auto-generated |
| \`--colormap\` | OpenCV colormap name | jet |
| \`--alpha\` | Heatmap overlay transparency | 0.6 |
| \`--sigma\` | Gaussian smoothing sigma | 20.0 |
| \`--preview\` | Show preview window | False |

### Available Colormaps

- \`jet\` - Blue to red (default)
- \`hot\` - Black to white through red
- \`inferno\` - Purple to yellow
- \`viridis\` - Purple to green to yellow
- \`plasma\` - Purple to orange to yellow

### Heatmap Process

1. Load detection points from JSON
2. Create density map on floor plan dimensions
3. Apply Gaussian smoothing for smooth distribution
4. Normalize and apply colormap
5. Overlay on floor plan with transparency

---

## API Reference

### MultiCameraDetector

\`\`\`python
from src.multi_camera_detector import MultiCameraDetector

# Initialize with calibration file
detector = MultiCameraDetector(
    calibration_path="calibration_data/homography_matrices.json",
    confidence=0.5
)

# Run detection for specified duration
detector.run(duration_seconds=300)

# Access results
detections = detector.get_detections()

# Save to file
detector.save_detections("output/detections.json")
\`\`\`

### HeatmapGenerator

\`\`\`python
from src.heatmap_generator import HeatmapGenerator

# Initialize with floor plan
generator = HeatmapGenerator(
    plan_path="/path/to/floor_plan.png",
    colormap="jet",
    alpha=0.6,
    sigma=20.0
)

# Load detection data
generator.load_detections("output/detections.json")

# Generate heatmap
heatmap = generator.generate()

# Save result
generator.save("output/heatmap.png")
\`\`\`

### HomographyCalculator

\`\`\`python
from calibration_system.homography import HomographyCalculator

calc = HomographyCalculator()

# Calculate homography from point pairs
result = calc.calculate_homography(camera_points, plan_points)

if result.is_valid:
    # Transform point from camera to plan
    plan_point = calc.transform_point(camera_point, result.matrix)
    print(f"Reprojection error: {result.reprojection_error:.2f}px")
\`\`\`

### YOLOPersonDetector

\`\`\`python
from models import YOLOPersonDetector

# Initialize detector
detector = YOLOPersonDetector(
    model_name="yolov8s.pt",
    confidence=0.5
)

# Detect persons in frame
detections = detector.detect_persons(frame)

# Each detection contains:
# - bbox: (x1, y1, x2, y2)
# - confidence: float
# - class_id: int (0 for person)
\`\`\`

---

## Troubleshooting

### RTSP Connection Issues

\`\`\`
[ERROR] Cannot connect to RTSP stream
\`\`\`

Solutions:
- Verify camera IP address and credentials
- Check network connectivity and firewall rules
- Ensure RTSP port (usually 554) is accessible
- Test the URL in VLC player first
- Verify the channel and subtype parameters

### YOLO Model Not Found

\`\`\`
[ERROR] Model not found
\`\`\`

Solutions:
- Run detection script once to auto-download
- Manually download model to \`models/weights/\`
- Check internet connectivity for auto-download

### Qt Display Issues

\`\`\`
qt.qpa.xcb: could not connect to display
\`\`\`

Solutions:
- Ensure X11 forwarding if using SSH (\`ssh -X\`)
- Set \`DISPLAY\` environment variable
- Use \`export QT_QPA_PLATFORM=offscreen\` for headless operation

### Homography Calculation Failed

\`\`\`
[ERROR] Could not calculate homography matrix
\`\`\`

Solutions:
- Ensure at least 4 point pairs are selected
- Check that points are not collinear
- Verify corresponding points match correctly
- Spread points across the visible floor area

### Detection Performance Issues

Solutions:
- Use a smaller YOLO model (yolov8n.pt)
- Reduce frame resolution in camera settings
- Enable GPU acceleration if available
- Reduce number of concurrent cameras

---

## Contributing

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/new-feature\`)
3. Commit changes (\`git commit -am 'Add new feature'\`)
4. Push to branch (\`git push origin feature/new-feature\`)
5. Create Pull Request

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Contact

For questions or support, please open an issue on the GitHub repository.
