"""
Production Scheduler Service
Monitors detection schedules and enqueues jobs to Redis.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Set
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.redis_manager import RedisManager
from src.production_config import ProductionConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchedulerService:
    """Monitors and schedules detection jobs"""
    
    def __init__(self, config_path: str = "production_config.json",
                 redis_host: str = "localhost", redis_port: int = 6379):
        """Initialize scheduler service"""
        
        logger.info("[INFO] Initializing Scheduler Service")
        
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
        
        # Cache configuration
        self.redis_manager.cache_production_config(self.config.to_dict())
        
        self.running = False
        self.jobs_enqueued_today: Set[str] = set()
        self.config_path = config_path
    
    def run(self, check_interval: int = 60):
        """
        Main scheduler loop
        
        Args:
            check_interval: How often to check schedules (seconds)
        """
        self.running = True
        logger.info("[INFO] Scheduler Service started")
        
        try:
            while self.running:
                try:
                    # Reload config periodically (every 10 checks)
                    if int(time.time()) % (check_interval * 10) == 0:
                        self._reload_config()
                    
                    # Check and enqueue jobs
                    self._check_and_enqueue_jobs()
                    
                    # Print status
                    self._print_status()
                    
                    time.sleep(check_interval)
                
                except KeyboardInterrupt:
                    logger.info("[INFO] Received interrupt signal")
                    break
                except Exception as e:
                    logger.error(f"[ERROR] Scheduler error: {e}")
                    time.sleep(check_interval)
        
        finally:
            self.running = False
            self.shutdown()
    
    def _check_and_enqueue_jobs(self):
        """Check active schedules and enqueue jobs"""
        try:
            # Get current time
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            current_day = now.strftime("%Y-%m-%d")
            
            # Check each schedule
            for store_name, schedule in self.config.schedules.items():
                if not schedule.enabled:
                    continue
                
                # Check if this schedule should be active
                if not schedule.is_active_now():
                    continue
                
                # Create unique key to avoid duplicate jobs per day
                job_key = f"{store_name}_{current_day}"
                
                # Check if already enqueued today
                if job_key in self.jobs_enqueued_today:
                    continue
                
                logger.info(f"[INFO] Enqueuing job for {store_name}")
                
                # Enqueue job
                job_id = self.redis_manager.enqueue_detection_job(
                    store_name=store_name,
                    schedule_data=schedule.to_dict()
                )
                
                # Track that we enqueued this today
                self.jobs_enqueued_today.add(job_key)
                
                logger.info(f"[OK] Job {job_id} enqueued")
        
        except Exception as e:
            logger.error(f"[ERROR] Error checking schedules: {e}")
    
    def _reload_config(self):
        """Reload configuration from file"""
        try:
            logger.debug("[DEBUG] Reloading configuration")
            new_config = ProductionConfig.load_from_json(self.config_path)
            if new_config:
                self.config = new_config
                self.redis_manager.cache_production_config(self.config.to_dict())
                logger.info("[INFO] Configuration reloaded")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to reload config: {e}")
    
    def _print_status(self):
        """Print scheduler status"""
        try:
            stats = self.redis_manager.get_queue_stats()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            active_count = len(self.config.get_active_schedules())
            
            logger.debug(
                f"[DEBUG] {current_time} - "
                f"Queue: {stats.get('queue_length', 0)}, "
                f"Active: {active_count}, "
                f"Completed: {stats.get('completed_jobs', 0)}, "
                f"Failed: {stats.get('failed_jobs', 0)}"
            )
        except Exception as e:
            logger.debug(f"[DEBUG] Error printing status: {e}")
    
    def get_scheduler_status(self) -> Dict:
        """Get scheduler status"""
        try:
            stats = self.redis_manager.get_queue_stats()
            active_schedules = self.config.get_active_schedules()
            
            return {
                "status": "running" if self.running else "stopped",
                "timestamp": datetime.now().isoformat(),
                "active_schedules": len(active_schedules),
                "total_schedules": len(self.config.schedules),
                "queue_stats": stats,
                "jobs_enqueued_today": len(self.jobs_enqueued_today)
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get status: {e}")
            return {}
    
    def manually_enqueue_job(self, store_name: str) -> Optional[str]:
        """
        Manually enqueue a detection job
        
        Args:
            store_name: Store name
        
        Returns:
            Job ID or None
        """
        try:
            schedule = self.config.get_schedule(store_name)
            if not schedule:
                logger.error(f"[ERROR] Schedule not found for {store_name}")
                return None
            
            logger.info(f"[INFO] Manually enqueuing job for {store_name}")
            
            job_id = self.redis_manager.enqueue_detection_job(
                store_name=store_name,
                schedule_data=schedule.to_dict()
            )
            
            return job_id
        except Exception as e:
            logger.error(f"[ERROR] Failed to manually enqueue job: {e}")
            return None
    
    def list_schedules(self) -> Dict:
        """List all configured schedules"""
        result = {}
        for store_name, schedule in self.config.schedules.items():
            is_active = schedule.is_active_now()
            result[store_name] = {
                "enabled": schedule.enabled,
                "active_now": is_active,
                "start_time": schedule.start_time,
                "end_time": schedule.end_time,
                "days": schedule.days_of_week,
                "duration_minutes": schedule.duration_minutes
            }
        return result
    
    def shutdown(self):
        """Shutdown service gracefully"""
        logger.info("[INFO] Shutting down Scheduler Service")
        self.running = False
        
        if self.redis_manager:
            self.redis_manager.close()
        
        logger.info("[INFO] Scheduler shutdown complete")


def main():
    """Main entry point for service"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Scheduler Service")
    parser.add_argument("--config", default="production_config.json",
                       help="Production config file path")
    parser.add_argument("--redis-host", default="localhost",
                       help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379,
                       help="Redis port")
    parser.add_argument("--check-interval", type=int, default=60,
                       help="Schedule check interval in seconds")
    
    args = parser.parse_args()
    
    # Create and run service
    try:
        scheduler = SchedulerService(
            config_path=args.config,
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        scheduler.run(check_interval=args.check_interval)
    except KeyboardInterrupt:
        logger.info("[INFO] Service interrupted")
    except Exception as e:
        logger.error(f"[FATAL] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
