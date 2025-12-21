"""
Production Detection Service
Runs detection jobs from Redis queue and stores results back.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.redis_manager import RedisManager
from src.production_config import ProductionConfig
from src.multi_camera_detector import MultiCameraDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDetectionService:
    """Main service for running scheduled detection jobs"""
    
    def __init__(self, config_path: str = "production_config.json", 
                 redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize production detection service"""
        
        logger.info("[INFO] Initializing Production Detection Service")
        
        # Load configuration
        self.config = ProductionConfig.load_from_json(config_path)
        if not self.config:
            raise RuntimeError(f"Failed to load config from {config_path}")
        
        # Initialize Redis
        try:
            self.redis_manager = RedisManager(
                host=redis_host,
                port=redis_port,
                db=self.config.redis_db
            )
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Redis: {e}")
            raise
        
        # Cache configuration in Redis
        self.redis_manager.cache_production_config(self.config.to_dict())
        
        self.running = False
        self.detector_threads: Dict[str, threading.Thread] = {}
    
    def run(self, poll_interval: int = 5):
        """
        Main service loop
        
        Args:
            poll_interval: How often to check for new jobs (seconds)
        """
        self.running = True
        logger.info("[INFO] Production Detection Service started")
        
        try:
            while self.running:
                try:
                    # Get next job from queue
                    job_id = self.redis_manager.dequeue_detection_job()
                    
                    if job_id:
                        logger.info(f"[INFO] Processing job {job_id}")
                        self.process_job(job_id)
                    else:
                        # No jobs in queue, wait before checking again
                        time.sleep(poll_interval)
                
                except KeyboardInterrupt:
                    logger.info("[INFO] Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"[ERROR] Service error: {e}")
                    time.sleep(poll_interval)
        
        finally:
            self.running = False
            self.shutdown()
    
    def process_job(self, job_id: str):
        """
        Process a single detection job
        
        Args:
            job_id: Job ID from Redis
        """
        start_time = time.time()
        
        try:
            # Get job details
            job_status = self.redis_manager.get_job_status(job_id)
            if not job_status:
                logger.error(f"[ERROR] Job {job_id} not found")
                return
            
            store_name = job_status.get("store_name")
            config_data = job_status.get("config")
            
            if not config_data:
                logger.error(f"[ERROR] No config in job {job_id}")
                self.redis_manager.update_job_status(job_id, "failed", 
                                                    {"error": "Missing config"})
                return
            
            # Update status to running
            self.redis_manager.update_job_status(job_id, "running")
            
            logger.info(f"[INFO] Running detection for {store_name}")
            
            # Check if calibration exists
            calibration_path = Path("calibration_data") / f"homography_matrices_{store_name}.json"
            if not calibration_path.exists():
                error_msg = f"Calibration not found: {calibration_path}"
                logger.error(f"[ERROR] {error_msg}")
                self.redis_manager.update_job_status(job_id, "failed", 
                                                    {"error": error_msg})
                return
            
            # Run detection
            try:
                detector = MultiCameraDetector(
                    calibration_path=str(calibration_path),
                    confidence=config_data.get("confidence_threshold", 0.5)
                )
                
                # Run for specified duration
                duration = config_data.get("duration_minutes", 60)
                detections = detector.run(duration_seconds=duration * 60)
                
                logger.info(f"[OK] Detection complete: {len(detections)} detections")
                
                # Store results in Redis
                self.redis_manager.store_detection_result(job_id, detections)
                
                # Update job status
                elapsed_time = time.time() - start_time
                self.redis_manager.update_job_status(
                    job_id, 
                    "completed",
                    {
                        "detections_count": len(detections),
                        "elapsed_seconds": int(elapsed_time)
                    }
                )
                
                logger.info(f"[OK] Job {job_id} completed in {elapsed_time:.1f}s")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"[ERROR] Detection failed: {error_msg}")
                self.redis_manager.update_job_status(job_id, "failed", 
                                                    {"error": error_msg})
        
        except Exception as e:
            logger.error(f"[ERROR] Failed to process job {job_id}: {e}")
            self.redis_manager.update_job_status(job_id, "failed", 
                                                {"error": str(e)})
    
    def handle_job_failure(self, job_id: str, error: Exception, retry_count: int = 3):
        """
        Handle job failure with retry logic
        
        Args:
            job_id: Job ID
            error: Exception that occurred
            retry_count: Number of retries remaining
        """
        logger.error(f"[ERROR] Job {job_id} failed: {error}")
        
        if retry_count > 0:
            logger.info(f"[INFO] Retrying job {job_id} ({retry_count} attempts left)")
            # Re-enqueue the job
            self.redis_manager.redis_client.rpush("detection:queue", job_id)
        else:
            logger.error(f"[ERROR] Job {job_id} failed permanently after retries")
            self.redis_manager.update_job_status(job_id, "failed",
                                                {"error": str(error)})
    
    def get_service_status(self) -> Dict:
        """Get service status and statistics"""
        try:
            stats = self.redis_manager.get_queue_stats()
            redis_info = self.redis_manager.get_redis_info()
            
            return {
                "status": "running" if self.running else "stopped",
                "timestamp": datetime.now().isoformat(),
                "queue_stats": stats,
                "redis_info": redis_info
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get status: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown service gracefully"""
        logger.info("[INFO] Shutting down Production Detection Service")
        self.running = False
        
        if self.redis_manager:
            self.redis_manager.close()
        
        logger.info("[INFO] Service shutdown complete")


def main():
    """Main entry point for service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Detection Service")
    parser.add_argument("--config", default="production_config.json",
                       help="Production config file path")
    parser.add_argument("--redis-host", default="localhost",
                       help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379,
                       help="Redis port")
    parser.add_argument("--poll-interval", type=int, default=5,
                       help="Job poll interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run service
    try:
        service = ProductionDetectionService(
            config_path=args.config,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        service.run(poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        logger.info("[INFO] Service interrupted")
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
