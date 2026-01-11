"""
API Exporter Service
Exports detection results from Redis and sends to backend API with retry logic.
"""

import os
import sys
import json
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.redis_manager import RedisManager
from src.production_config import ProductionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIExporter:
    """Exports detection results to backend API"""
    
    def __init__(self, config_path: str = "production_config.json",
                 redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize API exporter"""
        
        logger.info("[INFO] Initializing API Exporter")
        
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
        
        # Setup HTTP session with retry logic
        self.session = self._create_session()
        
        self.running = False
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def run(self, poll_interval: int = 10):
        """
        Main service loop
        
        Args:
            poll_interval: How often to check for completed jobs (seconds)
        """
        self.running = True
        logger.info("[INFO] API Exporter started")
        
        try:
            while self.running:
                try:
                    # Check for completed jobs
                    self._process_completed_jobs()
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
    
    def _process_completed_jobs(self):
        """Process all completed jobs in Redis"""
        try:
            # Get all job keys
            job_keys = self.redis_manager.redis_client.keys("job:*")
            
            for key in job_keys:
                job_id = key.split(":")[1]
                job_status = self.redis_manager.get_job_status(job_id)
                
                if job_status and job_status.get("status") == "completed":
                    # Check if already exported
                    if not job_status.get("exported"):
                        logger.info(f"[INFO] Exporting job {job_id}")
                        self.export_and_send(job_id)
        
        except Exception as e:
            logger.error(f"[ERROR] Error processing jobs: {e}")
    
    def export_and_send(self, job_id: str) -> bool:
        """
        Export job results and send to API
        
        Args:
            job_id: Job ID
        
        Returns:
            True if successful
        """
        try:
            # Get job info
            job_status = self.redis_manager.get_job_status(job_id)
            if not job_status:
                logger.error(f"[ERROR] Job not found: {job_id}")
                return False
            
            store_name = job_status.get("store_name")
            
            # Get detection results
            detections = self.redis_manager.get_detection_result(job_id)
            if detections is None:
                logger.error(f"[ERROR] No detection results for job {job_id}")
                return False
            
            logger.info(f"[INFO] Exporting {len(detections)} detections")
            
            # Generate export JSON
            export_data = self._generate_export_json(
                job_id, 
                store_name, 
                detections, 
                job_status
            )
            
            # Save to file
            output_file = self._save_export_file(job_id, store_name, export_data)
            logger.info(f"[OK] Export saved to {output_file}")
            
            # Get webhook config
            schedule = self.config.get_schedule(store_name)
            if not schedule or not schedule.webhook_enabled:
                logger.info(f"[INFO] Webhook disabled for {store_name}")
                self.redis_manager.update_job_status(job_id, "completed",
                                                    {"exported": True})
                return True
            
            # Send to API
            success = self.post_to_api(
                api_url=schedule.webhook_url,
                data=export_data,
                api_token=schedule.api_token,
                retries=schedule.retry_count,
                timeout=schedule.timeout_seconds
            )
            
            if success:
                self.redis_manager.update_job_status(job_id, "completed",
                                                    {"exported": True, "api_sent": True})
            
            return success
        
        except Exception as e:
            logger.error(f"[ERROR] Export failed for job {job_id}: {e}")
            return False
    
    def _generate_export_json(self, job_id: str, store_name: str,
                            detections: list, job_status: dict) -> Dict:
        """
        Generate final export JSON
        
        Args:
            job_id: Job ID
            store_name: Store name
            detections: List of detection records
            job_status: Job status dict
        
        Returns:
            Export JSON dictionary
        """
        # Calculate statistics
        total_detections = len(detections)
        avg_confidence = (
            sum(d.get("confidence", 0) for d in detections) / max(total_detections, 1)
        )
        
        # Group by camera
        cameras_data = {}
        for detection in detections:
            cam_id = detection.get("camera_id")
            if cam_id not in cameras_data:
                cameras_data[cam_id] = {"count": 0, "avg_confidence": 0}
            cameras_data[cam_id]["count"] += 1
        
        # Calculate camera average confidences
        for cam_id, detections_for_cam in cameras_data.items():
            cam_detections = [d for d in detections if d.get("camera_id") == cam_id]
            if cam_detections:
                avg_conf = sum(d.get("confidence", 0) for d in cam_detections) / len(cam_detections)
                cameras_data[cam_id]["avg_confidence"] = round(avg_conf, 3)
        
        export_data = {
            "job_id": job_id,
            "store_name": store_name,
            "timestamp": datetime.now().isoformat(),
            "duration_minutes": job_status.get("config", {}).get("duration_minutes"),
            
            # Metadata
            "metadata": {
                "total_detections": total_detections,
                "cameras_count": len(cameras_data),
                "average_confidence": round(avg_confidence, 3),
                "confidence_threshold": job_status.get("config", {}).get("confidence_threshold"),
                "detection_service_version": "1.0.0",
                "export_timestamp": datetime.now().isoformat()
            },
            
            # Statistics by camera
            "camera_statistics": cameras_data,
            
            # Detection data
            "detections": detections,
            
            # Execution info
            "execution": {
                "started_at": job_status.get("started_at"),
                "completed_at": job_status.get("completed_at"),
                "elapsed_seconds": int(job_status.get("elapsed_seconds", 0))
            }
        }
        
        return export_data
    
    def _save_export_file(self, job_id: str, store_name: str, data: Dict) -> str:
        """
        Save export data to JSON file
        
        Args:
            job_id: Job ID
            store_name: Store name
            data: Export data
        
        Returns:
            File path
        """
        try:
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{store_name}_{timestamp}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            return str(filepath)
        except Exception as e:
            logger.error(f"[ERROR] Failed to save export file: {e}")
            return ""
    
    def post_to_api(self, api_url: str, data: Dict, api_token: str,
                   retries: int = 3, timeout: int = 30) -> bool:
        """
        POST data to backend API with retry logic
        
        Args:
            api_url: API endpoint URL
            data: Data to POST
            api_token: API authentication token
            retries: Number of retries
            timeout: Request timeout in seconds
        
        Returns:
            True if successful
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"
        }
        
        for attempt in range(retries):
            try:
                logger.info(f"[INFO] Posting to {api_url} (attempt {attempt + 1}/{retries})")
                
                response = self.session.post(
                    api_url,
                    json=data,
                    headers=headers,
                    timeout=timeout
                )
                
                # Check response
                if response.status_code == 200:
                    logger.info(f"[OK] API POST successful (200 OK)")
                    response_data = response.json()
                    logger.info(f"[INFO] Server response: {response_data}")
                    return True
                
                elif response.status_code == 201:
                    logger.info(f"[OK] API POST successful (201 Created)")
                    return True
                
                elif response.status_code == 401:
                    logger.error(f"[ERROR] Authentication failed (401)")
                    return False
                
                elif response.status_code == 400:
                    logger.error(f"[ERROR] Bad request (400): {response.text}")
                    return False
                
                elif response.status_code >= 500:
                    logger.warning(f"[WARNING] Server error ({response.status_code})")
                    if attempt < retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"[INFO] Retrying in {wait_time}s")
                        time.sleep(wait_time)
                    continue
                
                else:
                    logger.warning(f"[WARNING] Unexpected status code: {response.status_code}")
                    return False
            
            except requests.exceptions.Timeout:
                logger.warning(f"[WARNING] Request timeout")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[WARNING] Connection error: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            except Exception as e:
                logger.error(f"[ERROR] Request failed: {e}")
                return False
        
        logger.error(f"[ERROR] Failed to post to API after {retries} retries")
        return False
    
    def shutdown(self):
        """Shutdown service gracefully"""
        logger.info("[INFO] Shutting down API Exporter")
        self.running = False
        
        if self.session:
            self.session.close()
        
        if self.redis_manager:
            self.redis_manager.close()
        
        logger.info("[INFO] Exporter shutdown complete")


def main():
    """Main entry point for service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Exporter Service")
    parser.add_argument("--config", default="production_config.json",
                       help="Production config file path")
    parser.add_argument("--redis-host", default="localhost",
                       help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379,
                       help="Redis port")
    parser.add_argument("--poll-interval", type=int, default=10,
                       help="Job poll interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run service
    try:
        exporter = APIExporter(
            config_path=args.config,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        exporter.run(poll_interval=args.poll_interval)
    except KeyboardInterrupt:
        logger.info("[INFO] Service interrupted")
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
