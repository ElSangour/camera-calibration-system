#!/usr/bin/env python
"""
Production System Testing and Example Usage
Demonstrates how to use all production modules.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.production_config import ProductionConfig, DetectionSchedule
from src.redis_manager import RedisManager
from src.scheduler import SchedulerService
from src.monitoring import get_monitoring_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_production_config():
    """Test production configuration"""
    print("\n" + "="*60)
    print("TEST 1: Production Configuration")
    print("="*60)
    
    # Create config
    config = ProductionConfig()
    
    # Create schedule
    schedule = DetectionSchedule(
        store_name="test_store",
        enabled=True,
        start_time="09:00",
        end_time="22:00",
        days_of_week=[0, 1, 2, 3, 4, 5],
        duration_minutes=60,
        webhook_url="https://api.example.com/heatmap",
        webhook_enabled=True,
        api_token="test_token"
    )
    
    config.add_schedule(schedule)
    
    # Validate
    valid, errors = config.validate_all()
    print(f"[INFO] Config valid: {valid}")
    
    # Save
    config.save_to_json("test_config.json")
    print("[OK] Config saved")
    
    # Load
    loaded = ProductionConfig.load_from_json("test_config.json")
    print(f"[OK] Config loaded with {len(loaded.schedules)} schedules")
    
    return loaded


def test_redis_manager():
    """Test Redis manager"""
    print("\n" + "="*60)
    print("TEST 2: Redis Manager")
    print("="*60)
    
    try:
        manager = RedisManager()
        
        # Test job queueing
        job_id = manager.enqueue_detection_job(
            store_name="test_store",
            schedule_data={"duration_minutes": 60}
        )
        print(f"[OK] Job enqueued: {job_id}")
        
        # Test status update
        manager.update_job_status(job_id, "running")
        print("[OK] Job status updated to running")
        
        # Test results storage
        detections = [
            {
                "timestamp": "2025-12-21T10:00:00",
                "camera_id": 1,
                "x": 100,
                "y": 200,
                "confidence": 0.9
            }
        ]
        manager.store_detection_result(job_id, detections)
        print(f"[OK] Stored {len(detections)} detection results")
        
        # Test retrieval
        result = manager.get_detection_result(job_id)
        print(f"[OK] Retrieved {len(result)} detections")
        
        # Test stats
        stats = manager.get_queue_stats()
        print(f"[OK] Queue stats: {stats}")
        
        manager.close()
        
    except Exception as e:
        print(f"[ERROR] Redis test failed: {e}")
        print("[INFO] Make sure Redis is running: redis-server")


def test_scheduler_service():
    """Test scheduler service"""
    print("\n" + "="*60)
    print("TEST 3: Scheduler Service")
    print("="*60)
    
    try:
        # Create test config
        config = ProductionConfig()
        schedule = DetectionSchedule(
            store_name="test_store",
            enabled=True,
            start_time="09:00",
            end_time="22:00",
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            duration_minutes=60
        )
        config.add_schedule(schedule)
        config.save_to_json("test_scheduler_config.json")
        
        # Create scheduler
        scheduler = SchedulerService("test_scheduler_config.json")
        
        # Get status
        status = scheduler.get_scheduler_status()
        print(f"[OK] Scheduler status: {status}")
        
        # List schedules
        schedules = scheduler.list_schedules()
        print(f"[OK] Active schedules: {list(schedules.keys())}")
        
        scheduler.shutdown()
        
    except Exception as e:
        print(f"[ERROR] Scheduler test failed: {e}")


def test_monitoring_service():
    """Test monitoring service"""
    print("\n" + "="*60)
    print("TEST 4: Monitoring Service")
    print("="*60)
    
    monitoring = get_monitoring_service()
    
    # Log events
    monitoring.log_detection_job(
        "job_test_001",
        "completed",
        {"detections_count": 150, "elapsed_seconds": 65}
    )
    print("[OK] Logged detection job")
    
    monitoring.log_api_post("test_store", True, 200)
    print("[OK] Logged API post")
    
    monitoring.log_error("test_detector", "Test error message")
    print("[OK] Logged error")
    
    # Get metrics
    metrics = monitoring.get_metrics()
    print(f"[OK] Metrics: success_rate={metrics['success_rate']}%, " 
          f"total_jobs={metrics['total_jobs_processed']}")
    
    # Save metrics
    monitoring.save_metrics("test_metrics.json")
    print("[OK] Metrics saved")


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*60)
    print("PRODUCTION SYSTEM USAGE")
    print("="*60)
    
    print("""
1. Start Redis:
   redis-server

2. Configure production_config.json with your schedules

3. Start Scheduler Service:
   python src/scheduler.py --config production_config.json

4. Start Detection Service:
   python src/production_detector.py --config production_config.json

5. Start API Exporter Service:
   python src/api_exporter.py --config production_config.json

6. Monitor via Systemd (on Linux):
   sudo systemctl start traffic-heatmap-scheduler
   sudo systemctl start traffic-heatmap-detector
   sudo systemctl start traffic-heatmap-exporter
   sudo systemctl status traffic-heatmap-*

7. View logs:
   tail -f logs/production.log

8. Check Redis queue:
   redis-cli
   > LLEN detection:queue
   > KEYS job:*
    """)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAFFIC HEATMAP - PRODUCTION SYSTEM TESTS")
    print("="*60)
    
    # Run tests
    try:
        test_production_config()
        test_redis_manager()
        test_scheduler_service()
        test_monitoring_service()
        print_usage_examples()
        
        print("\n[OK] All tests passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Tests failed: {e}")
        import traceback
        traceback.print_exc()
