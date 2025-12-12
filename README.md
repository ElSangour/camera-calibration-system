# Store Traffic Heatmap - Multi-Camera Vision System

Real-time customer traffic analysis system that processes multiple RTSP camera streams to generate unified heatmaps on a 2D store floor plan using computer vision and homography transformation.

## Overview
This system transforms multiple camera feeds into actionable retail analytics by:

Detecting people in 7 simultaneous RTSP camera streams using YOLOv8
Transforming camera coordinates to a unified 2D store map using homography matrices
Generating real-time heatmaps showing customer traffic patterns
Exporting JSON data for visualization in web/mobile interfaces

## Use Cases:

- Retail store traffic analysis
- Queue management optimization
- Store layout effectiveness
- Peak hours identification
- Customer behavior insights