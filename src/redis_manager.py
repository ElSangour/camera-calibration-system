"""
Redis Manager for Production Detection System
Handles job queuing, result storage, and configuration caching using Redis.
"""

import redis
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict

logger = logging.getLogger(__name__)


class RedisManager:
    """Manages Redis operations for job queue and result storage"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"[OK] Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"[ERROR] Failed to connect to Redis: {e}")
            raise
    
    # ============================================================================
    # Job Queue Management
    # ============================================================================
    
    def enqueue_detection_job(self, store_name: str, schedule_data: Dict, 
                            job_id: Optional[str] = None) -> str:
        """
        Enqueue a detection job to Redis queue
        
        Args:
            store_name: Name of the store
            schedule_data: Detection schedule configuration
            job_id: Optional custom job ID (auto-generated if not provided)
        
        Returns:
            Job ID
        """
        if not job_id:
            job_id = f"job_{int(datetime.now().timestamp())}_{store_name}"
        
        job_data = {
            "job_id": job_id,
            "store_name": store_name,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "detections_count": 0,
            "error": None,
            "config": schedule_data
        }
        
        # Store job info
        self.redis_client.hset(f"job:{job_id}", mapping=job_data)
        
        # Add to queue
        self.redis_client.rpush("detection:queue", job_id)
        
        # Set expiration (7 days)
        self.redis_client.expire(f"job:{job_id}", 604800)
        
        logger.info(f"[OK] Enqueued job {job_id} for {store_name}")
        return job_id
    
    def dequeue_detection_job(self) -> Optional[str]:
        """
        Get next job from queue (blocking, left pop)
        
        Returns:
            Job ID or None if queue empty
        """
        job_id = self.redis_client.lpop("detection:queue")
        if job_id:
            logger.debug(f"[INFO] Dequeued job {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get job status and metadata
        
        Args:
            job_id: Job ID
        
        Returns:
            Job info dictionary or None
        """
        job_data = self.redis_client.hgetall(f"job:{job_id}")
        if job_data:
            return job_data
        logger.warning(f"[WARNING] Job not found: {job_id}")
        return None
    
    def update_job_status(self, job_id: str, status: str, metadata: Optional[Dict] = None) -> bool:
        """
        Update job status
        
        Args:
            job_id: Job ID
            status: New status ("queued", "running", "completed", "failed")
            metadata: Additional metadata to update
        
        Returns:
            True if successful
        """
        try:
            update_data = {"status": status}
            
            if status == "running":
                update_data["started_at"] = datetime.now().isoformat()
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.now().isoformat()
            
            if metadata:
                update_data.update(metadata)
            
            self.redis_client.hset(f"job:{job_id}", mapping=update_data)
            logger.info(f"[INFO] Job {job_id} status updated to {status}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to update job status: {e}")
            return False
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued job
        
        Args:
            job_id: Job ID
        
        Returns:
            True if successful
        """
        try:
            # Remove from queue
            self.redis_client.lrem("detection:queue", 0, job_id)
            # Update status
            self.update_job_status(job_id, "cancelled")
            logger.info(f"[INFO] Job {job_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to cancel job: {e}")
            return False
    
    # ============================================================================
    # Detection Results Storage
    # ============================================================================
    
    def store_detection_result(self, job_id: str, detections: List[Dict], 
                              ttl_hours: int = 24) -> bool:
        """
        Store detection results
        
        Args:
            job_id: Job ID
            detections: List of detection data
            ttl_hours: Time to live in hours
        
        Returns:
            True if successful
        """
        try:
            result_data = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "detections_count": len(detections),
                "data": json.dumps(detections)
            }
            
            self.redis_client.hset(f"result:{job_id}", mapping=result_data)
            
            # Set expiration
            ttl_seconds = ttl_hours * 3600
            self.redis_client.expire(f"result:{job_id}", ttl_seconds)
            
            logger.info(f"[OK] Stored {len(detections)} detections for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to store detection results: {e}")
            return False
    
    def get_detection_result(self, job_id: str) -> Optional[List[Dict]]:
        """
        Retrieve detection results
        
        Args:
            job_id: Job ID
        
        Returns:
            List of detections or None
        """
        try:
            result_data = self.redis_client.hget(f"result:{job_id}", "data")
            if result_data:
                return json.loads(result_data)
            logger.warning(f"[WARNING] No results found for job {job_id}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Failed to retrieve detection results: {e}")
            return None
    
    def delete_result(self, job_id: str) -> bool:
        """Delete detection results"""
        try:
            self.redis_client.delete(f"result:{job_id}")
            logger.info(f"[INFO] Deleted results for job {job_id}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to delete results: {e}")
            return False
    
    # ============================================================================
    # Configuration Caching
    # ============================================================================
    
    def cache_production_config(self, config_dict: Dict, ttl_hours: int = 24) -> bool:
        """
        Cache production configuration
        
        Args:
            config_dict: Configuration dictionary
            ttl_hours: Time to live in hours
        
        Returns:
            True if successful
        """
        try:
            config_json = json.dumps(config_dict)
            self.redis_client.set("config:production", config_json)
            
            # Set expiration
            ttl_seconds = ttl_hours * 3600
            self.redis_client.expire("config:production", ttl_seconds)
            
            logger.info(f"[OK] Production config cached")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to cache config: {e}")
            return False
    
    def get_cached_config(self) -> Optional[Dict]:
        """
        Retrieve cached production configuration
        
        Returns:
            Configuration dictionary or None
        """
        try:
            config_json = self.redis_client.get("config:production")
            if config_json:
                return json.loads(config_json)
            logger.warning("[WARNING] No cached config found")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Failed to retrieve cached config: {e}")
            return None
    
    def cache_calibration_data(self, store_name: str, calibration_data: Dict) -> bool:
        """
        Cache calibration data for a store
        
        Args:
            store_name: Store name
            calibration_data: Calibration dictionary
        
        Returns:
            True if successful
        """
        try:
            key = f"calibration:{store_name}"
            calib_json = json.dumps(calibration_data)
            self.redis_client.set(key, calib_json)
            self.redis_client.expire(key, 604800)  # 7 days
            logger.info(f"[OK] Calibration data cached for {store_name}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to cache calibration: {e}")
            return False
    
    def get_cached_calibration(self, store_name: str) -> Optional[Dict]:
        """
        Retrieve cached calibration data
        
        Args:
            store_name: Store name
        
        Returns:
            Calibration dictionary or None
        """
        try:
            key = f"calibration:{store_name}"
            calib_json = self.redis_client.get(key)
            if calib_json:
                return json.loads(calib_json)
            logger.warning(f"[WARNING] No cached calibration for {store_name}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] Failed to retrieve cached calibration: {e}")
            return None
    
    # ============================================================================
    # Monitoring and Statistics
    # ============================================================================
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            queue_length = self.redis_client.llen("detection:queue")
            
            # Get recent jobs
            job_keys = self.redis_client.keys("job:*")
            total_jobs = len(job_keys)
            
            completed = 0
            failed = 0
            running = 0
            
            for key in job_keys:
                status = self.redis_client.hget(key, "status")
                if status == "completed":
                    completed += 1
                elif status == "failed":
                    failed += 1
                elif status == "running":
                    running += 1
            
            stats = {
                "queue_length": queue_length,
                "total_jobs": total_jobs,
                "completed_jobs": completed,
                "failed_jobs": failed,
                "running_jobs": running,
                "success_rate": f"{(completed / max(total_jobs, 1) * 100):.1f}%"
            }
            
            logger.debug(f"[INFO] Queue stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"[ERROR] Failed to get queue stats: {e}")
            return {}
    
    def get_job_history(self, store_name: str, limit: int = 100) -> List[Dict]:
        """
        Get job history for a store
        
        Args:
            store_name: Store name
            limit: Maximum number of jobs to return
        
        Returns:
            List of job info dictionaries
        """
        try:
            job_keys = self.redis_client.keys(f"job:*_{store_name}")
            job_keys = job_keys[-limit:]  # Get last N jobs
            
            jobs = []
            for key in sorted(job_keys, reverse=True):
                job_data = self.redis_client.hgetall(key)
                jobs.append(job_data)
            
            logger.debug(f"[INFO] Retrieved {len(jobs)} jobs for {store_name}")
            return jobs
        except Exception as e:
            logger.error(f"[ERROR] Failed to get job history: {e}")
            return []
    
    def get_redis_info(self) -> Dict[str, Any]:
        """
        Get Redis server information
        
        Returns:
            Redis info dictionary
        """
        try:
            info = self.redis_client.info()
            return {
                "used_memory_mb": info.get("used_memory") / (1024 * 1024),
                "connected_clients": info.get("connected_clients"),
                "total_commands": info.get("total_commands_processed"),
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get Redis info: {e}")
            return {}
    
    # ============================================================================
    # Cleanup and Maintenance
    # ============================================================================
    
    def cleanup_expired_jobs(self, days: int = 7) -> int:
        """
        Cleanup old job records
        
        Args:
            days: Delete jobs older than this many days
        
        Returns:
            Number of jobs deleted
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            job_keys = self.redis_client.keys("job:*")
            
            deleted_count = 0
            for key in job_keys:
                created_at = self.redis_client.hget(key, "created_at")
                if created_at and created_at < cutoff_date:
                    self.redis_client.delete(key)
                    deleted_count += 1
            
            logger.info(f"[INFO] Cleaned up {deleted_count} old job records")
            return deleted_count
        except Exception as e:
            logger.error(f"[ERROR] Failed to cleanup jobs: {e}")
            return 0
    
    def flush_all(self) -> bool:
        """
        Flush all data (use with caution!)
        
        Returns:
            True if successful
        """
        try:
            self.redis_client.flushdb()
            logger.warning("[WARNING] Flushed all Redis data")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to flush Redis: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        self.redis_client.close()
        logger.info("[INFO] Redis connection closed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    manager = RedisManager()
    
    # Test job queueing
    job_id = manager.enqueue_detection_job(
        store_name="test_store",
        schedule_data={"duration_minutes": 60}
    )
    
    # Test status update
    manager.update_job_status(job_id, "running")
    
    # Test results storage
    detections = [
        {"timestamp": "2025-12-21T10:00:00", "x": 100, "y": 200, "confidence": 0.9}
    ]
    manager.store_detection_result(job_id, detections)
    
    # Test retrieval
    result = manager.get_detection_result(job_id)
    print(f"[INFO] Retrieved {len(result)} detections")
    
    # Test stats
    stats = manager.get_queue_stats()
    print(f"[INFO] Queue stats: {stats}")
    
    manager.close()
