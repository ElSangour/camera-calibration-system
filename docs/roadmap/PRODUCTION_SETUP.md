# Production Deployment Guide

Complete guide for deploying Traffic Heatmap to production with automated scheduling, detection, and API integration.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running Services](#running-services)
6. [Systemd Integration](#systemd-integration)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [API Specifications](#api-specifications)

---

## System Overview

The production system consists of three independent services:

| Service | Purpose | Responsibility |
|---------|---------|-----------------|
| **Scheduler** | Monitors schedules | Enqueues jobs when conditions met |
| **Detector** | Runs detection | Processes jobs, stores results to Redis |
| **Exporter** | Exports results | Retrieves results, POSTs to API |

### Architecture

```
┌─────────────────────────────────────────────┐
│   Production Configuration (JSON)            │
│   - Detection schedules per store           │
│   - Time windows, parameters                │
│   - API endpoints, authentication           │
└──────────────────┬──────────────────────────┘
                   |
    ┌──────────────┼──────────────┐
    |              |              |
    v              v              v
┌────────┐  ┌────────────┐  ┌──────────┐
│Scheduler│  │ Detection  │  │ Exporter │
│Service  │  │ Service    │  │ Service  │
└────┬───┘  └──────┬─────┘  └──────┬───┘
     |             |              |
     └─────────────┼──────────────┘
                   |
              ┌────v─────┐
              │  Redis   │
              │ Queue    │
              └──────────┘
                   |
     ┌─────────────┴─────────────┐
     |                           |
     v                           v
  Backend API            Detection Database
```

---

## Prerequisites

### System Requirements

- Linux server (Ubuntu 20.04+ recommended)
- Python 3.8+
- Redis server 5.0+
- Network access to RTSP cameras
- Network access to backend API
- GPU recommended (optional, for faster detection)

### Software Dependencies

All Python dependencies are in `requirements.txt`:
- OpenCV 4.12.0
- ultralytics 8.3.236 (YOLO)
- PyQt6 6.10.1
- redis 7.1.0
- requests 2.31.0

---

## Installation

### 1. Prepare System

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y python3-pip python3-venv redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Verify Redis
redis-cli ping
# Should return: PONG
```

### 2. Clone and Setup Project

```bash
# Clone repository
git clone https://github.com/ElSangour/Traffic-Heatmap.git
cd Traffic-Heatmap

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create Service User

```bash
# Create dedicated user for service
sudo useradd -m -s /bin/bash heatmap

# Create directories with proper permissions
sudo mkdir -p /opt/traffic-heatmap
sudo mkdir -p /var/log/traffic-heatmap
sudo mkdir -p /var/lib/traffic-heatmap
sudo chown -R heatmap:heatmap /opt/traffic-heatmap
sudo chown -R heatmap:heatmap /var/log/traffic-heatmap
sudo chown -R heatmap:heatmap /var/lib/traffic-heatmap

# Copy project to /opt
sudo cp -r . /opt/traffic-heatmap
sudo chown -R heatmap:heatmap /opt/traffic-heatmap
```

---

## Configuration

### 1. Create Production Configuration

Edit `production_config.json`:

```json
{
  "schedules": {
    "store_1": {
      "store_name": "store_1",
      "enabled": true,
      "start_time": "09:00",
      "end_time": "22:00",
      "days_of_week": [0, 1, 2, 3, 4, 5, 6],
      "duration_minutes": 60,
      "frame_skip": 1,
      "confidence_threshold": 0.5,
      "export_enabled": true,
      "output_format": "json",
      "webhook_url": "https://api.backend.com/analytics/heatmap",
      "webhook_enabled": true,
      "retry_count": 3,
      "timeout_seconds": 1800,
      "api_token": "your_api_token_here"
    }
  },
  "redis_host": "localhost",
  "redis_port": 6379,
  "redis_db": 0,
  "log_level": "INFO",
  "log_file": "/var/log/traffic-heatmap/production.log",
  "enable_monitoring": true,
  "alert_email": "admin@example.com"
}
```

### 2. Validate Configuration

```bash
# Test configuration
source .venv/bin/activate
python3 test_production_system.py

# You should see tests pass for:
# - Production Configuration
# - Redis Manager
# - Scheduler Service
# - Monitoring Service
```

### 3. Prepare Calibration Data

Ensure calibration files exist:

```bash
# Should contain calibration_*.json and homography_matrices_*.json
ls -la calibration_data/

# Copy calibration to accessible location if needed
cp calibration_data/* /var/lib/traffic-heatmap/
```

---

## Running Services

### Manual Testing

```bash
# Terminal 1: Start Scheduler
python3 src/scheduler.py --config production_config.json

# Terminal 2: Start Detector
python3 src/production_detector.py --config production_config.json

# Terminal 3: Start Exporter
python3 src/api_exporter.py --config production_config.json

# Terminal 4: Monitor via Redis
redis-cli MONITOR
```

### Stopping Services

```bash
# Press Ctrl+C in each terminal
# Or kill the processes
pkill -f production_detector.py
pkill -f api_exporter.py
pkill -f scheduler.py
```

---

## Systemd Integration

### 1. Create Service Files

Create `/etc/systemd/system/traffic-heatmap-scheduler.service`:

```ini
[Unit]
Description=Traffic Heatmap Scheduler Service
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=heatmap
WorkingDirectory=/opt/traffic-heatmap
Environment="PATH=/opt/traffic-heatmap/.venv/bin"
ExecStart=/opt/traffic-heatmap/.venv/bin/python src/scheduler.py --config production_config.json
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/traffic-heatmap-detector.service`:

```ini
[Unit]
Description=Traffic Heatmap Detection Service
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=heatmap
WorkingDirectory=/opt/traffic-heatmap
Environment="PATH=/opt/traffic-heatmap/.venv/bin"
ExecStart=/opt/traffic-heatmap/.venv/bin/python src/production_detector.py --config production_config.json
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/traffic-heatmap-exporter.service`:

```ini
[Unit]
Description=Traffic Heatmap API Exporter Service
After=network.target redis-server.service
Wants=redis-server.service

[Service]
Type=simple
User=heatmap
WorkingDirectory=/opt/traffic-heatmap
Environment="PATH=/opt/traffic-heatmap/.venv/bin"
ExecStart=/opt/traffic-heatmap/.venv/bin/python src/api_exporter.py --config production_config.json
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable traffic-heatmap-scheduler
sudo systemctl enable traffic-heatmap-detector
sudo systemctl enable traffic-heatmap-exporter

# Start services
sudo systemctl start traffic-heatmap-scheduler
sudo systemctl start traffic-heatmap-detector
sudo systemctl start traffic-heatmap-exporter

# Check status
sudo systemctl status traffic-heatmap-*
```

### 3. Manage Services

```bash
# Start individual service
sudo systemctl start traffic-heatmap-scheduler

# Stop service
sudo systemctl stop traffic-heatmap-detector

# Restart service
sudo systemctl restart traffic-heatmap-exporter

# View logs
sudo journalctl -u traffic-heatmap-scheduler -f
sudo journalctl -u traffic-heatmap-detector -f
sudo journalctl -u traffic-heatmap-exporter -f

# View all Traffic Heatmap logs
sudo journalctl -u traffic-heatmap-* -f
```

---

## Monitoring

### Check Service Status

```bash
# View all services
systemctl list-units traffic-heatmap-*

# Check specific service
sudo systemctl status traffic-heatmap-detector

# View recent logs
sudo journalctl -u traffic-heatmap-detector -n 50

# Live log monitoring
sudo journalctl -u traffic-heatmap-detector -f
```

### Redis Monitoring

```bash
# Check Redis connection
redis-cli ping

# View job queue
redis-cli LLEN detection:queue

# List job IDs
redis-cli KEYS "job:*"

# Check job status
redis-cli HGETALL "job:job_1702816800_store_1"

# View queue statistics
redis-cli INFO stats
```

### Metrics

```bash
# View metrics
cat logs/production.log | grep "metrics"

# Export metrics to file
tail -f logs/production.log > metrics.json
```

---

## Troubleshooting

### Redis Connection Failed

```bash
# Check if Redis is running
redis-cli ping

# Start Redis
sudo systemctl start redis-server

# Check Redis logs
sudo journalctl -u redis-server -f

# Verify Redis listening
netstat -tuln | grep 6379
```

### Services Not Starting

```bash
# Check service status
sudo systemctl status traffic-heatmap-detector

# View error logs
sudo journalctl -u traffic-heatmap-detector -n 100

# Try running manually for debugging
cd /opt/traffic-heatmap
source .venv/bin/activate
python3 src/production_detector.py --config production_config.json
```

### Calibration Not Found

```bash
# Verify calibration files exist
ls -la calibration_data/

# Check calibration naming
cat calibration_data/homography_matrices_*.json

# Ensure store_name matches configuration
grep "store_name" production_config.json
```

### API POST Failures

```bash
# Test API endpoint manually
curl -X POST https://api.backend.com/analytics/heatmap \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d @test_payload.json

# Check for network issues
ping api.backend.com
curl -v https://api.backend.com/

# Review error logs
grep "api_post" logs/production.log
```

### High Memory Usage

```bash
# Check Redis memory
redis-cli INFO memory

# Clean old job records
redis-cli EVAL "
  local keys = redis.call('KEYS', 'job:*')
  for i,k in ipairs(keys) do redis.call('DEL', k) end
  return #keys
" 0

# Check log file size
du -sh logs/production.log

# Rotate logs
sudo logrotate -f /etc/logrotate.d/traffic-heatmap
```

---

## API Specifications

### Webhook Endpoint

**URL**: Configured in `webhook_url`  
**Method**: POST  
**Authentication**: Bearer token in Authorization header

### Request Headers

```
Content-Type: application/json
Authorization: Bearer {api_token}
```

### Request Payload

```json
{
  "job_id": "job_1702816800_store_1",
  "store_name": "store_1",
  "timestamp": "2025-12-21T14:00:00.123456",
  "duration_minutes": 60,
  
  "metadata": {
    "total_detections": 7972,
    "cameras_count": 7,
    "average_confidence": 0.84,
    "confidence_threshold": 0.5,
    "detection_service_version": "1.0.0",
    "export_timestamp": "2025-12-21T14:01:00.123456"
  },
  
  "camera_statistics": {
    "1": {
      "count": 1145,
      "avg_confidence": 0.85
    }
  },
  
  "detections": [
    {
      "timestamp": "2025-12-21T14:00:06.123456",
      "camera_id": 1,
      "camera_point": {"x": 320, "y": 480},
      "plan_point": {"x": 150, "y": 280},
      "confidence": 0.87
    }
  ],
  
  "execution": {
    "started_at": "2025-12-21T14:00:00.000000",
    "completed_at": "2025-12-21T14:01:05.000000",
    "elapsed_seconds": 65
  }
}
```

### Response (200 OK)

```json
{
  "success": true,
  "message": "Data received and stored",
  "record_id": "record_uuid_123"
}
```

---

## Performance Tuning

For optimal performance, adjust parameters in `production_config.json`:

- **High Throughput**: Reduce `frame_skip`, enable GPU, lower `confidence_threshold`
- **Low Latency**: Reduce `retry_count` and `timeout_seconds`
- **Cost Optimization**: Use smaller YOLO model (yolov8n.pt)

---

## Support

For issues, check logs with:

```bash
sudo journalctl -u traffic-heatmap-* -f
```
