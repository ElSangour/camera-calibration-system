"""
Monitoring Service
Logging, metrics collection, and alerting for production detection system.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler


class MonitoringService:
    """Handles logging, metrics, and alerts"""
    
    def __init__(self, log_file: str = "logs/production.log", log_level: str = "INFO"):
        """Initialize monitoring service"""
        
        # Create logs directory
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        self.file_handler.setLevel(getattr(logging, log_level))
        
        # Setup formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        
        # Get root logger
        self.logger = logging.getLogger()
        self.logger.addHandler(self.file_handler)
        self.logger.setLevel(getattr(logging, log_level))
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "total_jobs_processed": 0,
            "total_jobs_failed": 0,
            "total_detections": 0,
            "total_api_posts": 0,
            "successful_api_posts": 0,
            "failed_api_posts": 0,
            "last_error": None,
            "last_error_time": None
        }
    
    def log_detection_job(self, job_id: str, status: str, metadata: Dict = None):
        """
        Log detection job event
        
        Args:
            job_id: Job ID
            status: Job status
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "detection_job",
            "job_id": job_id,
            "status": status
        }
        
        if metadata:
            log_entry.update(metadata)
        
        # Log to file
        if status == "failed":
            self.logger.error(json.dumps(log_entry))
            self.metrics["total_jobs_failed"] += 1
        else:
            self.logger.info(json.dumps(log_entry))
        
        if status == "completed":
            self.metrics["total_jobs_processed"] += 1
            if metadata and "detections_count" in metadata:
                self.metrics["total_detections"] += metadata["detections_count"]
    
    def log_api_post(self, store_name: str, success: bool, response_code: int = None,
                    error: str = None):
        """
        Log API POST event
        
        Args:
            store_name: Store name
            success: Whether POST was successful
            response_code: HTTP response code
            error: Error message if failed
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "api_post",
            "store_name": store_name,
            "success": success,
            "response_code": response_code
        }
        
        if error:
            log_entry["error"] = error
        
        if success:
            self.logger.info(json.dumps(log_entry))
            self.metrics["successful_api_posts"] += 1
        else:
            self.logger.error(json.dumps(log_entry))
            self.metrics["failed_api_posts"] += 1
        
        self.metrics["total_api_posts"] += 1
    
    def log_error(self, component: str, error: str):
        """
        Log error event
        
        Args:
            component: Component name (detector, exporter, scheduler)
            error: Error message
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "component": component,
            "error": error
        }
        
        self.logger.error(json.dumps(log_entry))
        self.metrics["last_error"] = error
        self.metrics["last_error_time"] = datetime.now().isoformat()
    
    def send_alert(self, message: str, level: str = "WARNING", alert_email: str = None):
        """
        Send alert (via email if configured)
        
        Args:
            message: Alert message
            level: Alert level (WARNING, ERROR, CRITICAL)
            alert_email: Email address to send alert to
        """
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "alert",
            "level": level,
            "message": message
        }
        
        self.logger.warning(json.dumps(alert_entry))
        
        # TODO: Implement email sending
        if alert_email:
            self._send_email_alert(alert_email, level, message)
    
    def _send_email_alert(self, email: str, level: str, message: str):
        """Send email alert (to be implemented)"""
        # This would use SMTP to send emails
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        
        Returns:
            Metrics dictionary
        """
        uptime_seconds = (
            datetime.now() - 
            datetime.fromisoformat(self.metrics["start_time"])
        ).total_seconds()
        
        metrics = self.metrics.copy()
        metrics["uptime_seconds"] = int(uptime_seconds)
        
        # Calculate rates
        if metrics["total_jobs_processed"] > 0:
            metrics["success_rate"] = (
                (metrics["total_jobs_processed"] - metrics["total_jobs_failed"]) / 
                metrics["total_jobs_processed"] * 100
            )
        else:
            metrics["success_rate"] = 0.0
        
        if metrics["total_api_posts"] > 0:
            metrics["api_success_rate"] = (
                metrics["successful_api_posts"] / 
                metrics["total_api_posts"] * 100
            )
        else:
            metrics["api_success_rate"] = 0.0
        
        return metrics
    
    def save_metrics(self, filepath: str = "metrics.json"):
        """Save metrics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_metrics(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self, filepath: str = "metrics.json"):
        """Load metrics from file"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r') as f:
                    saved_metrics = json.load(f)
                    # Merge with current metrics
                    self.metrics.update(saved_metrics)
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")


# Global monitoring instance
_monitoring_service: MonitoringService = None


def get_monitoring_service(log_file: str = "logs/production.log", 
                          log_level: str = "INFO") -> MonitoringService:
    """Get or create monitoring service"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService(log_file, log_level)
    return _monitoring_service


if __name__ == "__main__":
    # Example usage
    monitoring = MonitoringService()
    
    # Log events
    monitoring.log_detection_job(
        "job_123",
        "completed",
        {"detections_count": 100, "elapsed_seconds": 60}
    )
    
    monitoring.log_api_post("store_1", True, 200)
    monitoring.log_error("detector", "Connection timeout")
    monitoring.send_alert("Redis connection lost", "CRITICAL")
    
    # Get metrics
    metrics = monitoring.get_metrics()
    print(json.dumps(metrics, indent=2))
    
    # Save metrics
    monitoring.save_metrics()
