"""
Production Detection Scheduling Configuration
Defines schedules, parameters, and API integration settings for automated detection jobs.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionSchedule:
    """Configuration for a single store's detection schedule"""
    store_name: str
    enabled: bool = True
    
    # Timing Configuration
    start_time: str = "09:00"           # HH:MM format
    end_time: str = "22:00"             # HH:MM format
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # 0=Monday, 6=Sunday
    
    # Detection Parameters
    duration_minutes: int = 60
    frame_skip: int = 1
    confidence_threshold: float = 0.5
    
    # Output Configuration
    export_enabled: bool = True
    output_format: str = "json"         # "json", "csv", "both"
    
    # API Integration
    webhook_url: str = ""
    webhook_enabled: bool = False
    retry_count: int = 3
    timeout_seconds: int = 1800         # 30 minutes
    api_token: str = ""                 # Stored encrypted in production
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectionSchedule':
        """Create from dictionary"""
        return cls(**data)
    
    def is_active_now(self) -> bool:
        """Check if schedule should be active at current time"""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        current_day = now.weekday()
        
        # Check day of week
        if current_day not in self.days_of_week:
            return False
        
        # Check time range
        if not (self.start_time <= current_time <= self.end_time):
            return False
        
        return self.enabled
    
    def validate(self) -> Tuple[bool, str]:
        """Validate schedule configuration"""
        errors = []
        
        # Validate times
        try:
            start = datetime.strptime(self.start_time, "%H:%M").time()
            end = datetime.strptime(self.end_time, "%H:%M").time()
            if start >= end:
                errors.append("Start time must be before end time")
        except ValueError:
            errors.append("Invalid time format (use HH:MM)")
        
        # Validate days
        if not self.days_of_week or not all(0 <= d <= 6 for d in self.days_of_week):
            errors.append("Days must be between 0-6")
        
        # Validate parameters
        if self.duration_minutes <= 0:
            errors.append("Duration must be positive")
        if not (0 <= self.confidence_threshold <= 1.0):
            errors.append("Confidence threshold must be 0.0-1.0")
        
        # Validate API config if enabled
        if self.webhook_enabled:
            if not self.webhook_url:
                errors.append("Webhook URL required when webhook enabled")
            if not self.api_token:
                errors.append("API token required when webhook enabled")
        
        if errors:
            return False, "; ".join(errors)
        return True, "Valid"


@dataclass
class ProductionConfig:
    """Main production configuration"""
    schedules: Dict[str, DetectionSchedule] = field(default_factory=dict)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"             # DEBUG, INFO, WARNING, ERROR
    log_file: str = "logs/production.log"
    
    # Monitoring
    enable_monitoring: bool = True
    alert_email: str = ""
    
    def add_schedule(self, schedule: DetectionSchedule) -> None:
        """Add a detection schedule"""
        self.schedules[schedule.store_name] = schedule
        logger.info(f"[INFO] Added schedule for {schedule.store_name}")
    
    def remove_schedule(self, store_name: str) -> bool:
        """Remove a detection schedule"""
        if store_name in self.schedules:
            del self.schedules[store_name]
            logger.info(f"[INFO] Removed schedule for {store_name}")
            return True
        return False
    
    def get_schedule(self, store_name: str) -> Optional[DetectionSchedule]:
        """Get schedule for store"""
        return self.schedules.get(store_name)
    
    def get_active_schedules(self) -> List[DetectionSchedule]:
        """Get all active schedules that should run now"""
        return [s for s in self.schedules.values() if s.is_active_now()]
    
    def validate_all(self) -> Tuple[bool, Dict[str, str]]:
        """Validate all schedules"""
        errors = {}
        for store_name, schedule in self.schedules.items():
            valid, msg = schedule.validate()
            if not valid:
                errors[store_name] = msg
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "schedules": {name: sched.to_dict() for name, sched in self.schedules.items()},
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "enable_monitoring": self.enable_monitoring,
            "alert_email": self.alert_email,
        }
    
    def save_to_json(self, path: str) -> bool:
        """Save configuration to JSON file"""
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"[OK] Configuration saved to {path}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to save config: {e}")
            return False
    
    @classmethod
    def load_from_json(cls, path: str) -> Optional['ProductionConfig']:
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            config = cls(
                redis_host=data.get("redis_host", "localhost"),
                redis_port=data.get("redis_port", 6379),
                redis_db=data.get("redis_db", 0),
                log_level=data.get("log_level", "INFO"),
                log_file=data.get("log_file", "logs/production.log"),
                enable_monitoring=data.get("enable_monitoring", True),
                alert_email=data.get("alert_email", ""),
            )
            
            # Load schedules
            for store_name, sched_data in data.get("schedules", {}).items():
                schedule = DetectionSchedule.from_dict(sched_data)
                config.add_schedule(schedule)
            
            logger.info(f"[OK] Configuration loaded from {path}")
            return config
        except Exception as e:
            logger.error(f"[ERROR] Failed to load config: {e}")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config = ProductionConfig()
    
    schedule = DetectionSchedule(
        store_name="mg_cite_olympique",
        enabled=True,
        start_time="09:00",
        end_time="22:00",
        days_of_week=[0, 1, 2, 3, 4, 5],
        duration_minutes=60,
        webhook_url="https://api.backend.com/analytics/heatmap",
        webhook_enabled=True,
        api_token="your_token_here",
    )
    
    config.add_schedule(schedule)
    
    # Validate
    valid, errors = config.validate_all()
    print(f"[INFO] Configuration valid: {valid}")
    
    if errors:
        for store, error in errors.items():
            print(f"[ERROR] {store}: {error}")
    
    # Save
    config.save_to_json("production_config.json")
