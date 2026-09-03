"""Health monitor tool - provides /health command and management interface."""

import json
import os
from typing import Dict, Any, Optional

# Import with fallback for both package and standalone modes
try:
    from .monitor import HealthMonitor
    from .monitors.base import HealthStatus
except ImportError:
    from monitor import HealthMonitor
    from monitors.base import HealthStatus


class HealthTool:
    """Tool handler for health monitoring commands."""
    
    _instance: Optional[HealthMonitor] = None
    
    @classmethod
    def get_monitor(cls) -> HealthMonitor:
        """Get or create the singleton health monitor."""
        if cls._instance is None:
            check_interval = int(os.getenv("HERMES_HEALTH_CHECK_INTERVAL", "300"))
            self_heal = os.getenv("HERMES_HEALTH_SELF_HEAL", "1") == "1"
            alert_threshold = int(os.getenv("HERMES_HEALTH_ALERT_THRESHOLD", "2"))
            
            cls._instance = HealthMonitor(
                check_interval=check_interval,
                self_heal=self_heal,
                alert_threshold=alert_threshold
            )
        return cls._instance
    
    @classmethod
    def handle(cls, args: Dict[str, Any], **kwargs) -> str:
        """Handle health tool commands."""
        action = args.get("action", "status")
        
        try:
            if action == "status":
                return cls._status(args)
            elif action == "check":
                return cls._check(args)
            elif action == "start":
                return cls._start(args)
            elif action == "stop":
                return cls._stop(args)
            elif action == "config":
                return cls._config(args)
            else:
                return json.dumps({
                    "error": f"Unknown action: {action}",
                    "available_actions": ["status", "check", "start", "stop", "config"]
                })
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    @classmethod
    def _status(cls, args: Dict[str, Any]) -> str:
        """Get current health status."""
        monitor = cls.get_monitor()
        format = args.get("format", "text")
        
        # Run a fresh check
        monitor.check_all()
        
        return monitor.get_health_report(format)
    
    @classmethod
    def _check(cls, args: Dict[str, Any]) -> str:
        """Force an immediate health check."""
        monitor = cls.get_monitor()
        health = monitor.check_all()
        
        format = args.get("format", "json")
        return monitor.get_health_report(format)
    
    @classmethod
    def _start(cls, args: Dict[str, Any]) -> str:
        """Start background health monitoring."""
        monitor = cls.get_monitor()
        monitor.start_background_monitoring()
        
        return json.dumps({
            "status": "started",
            "check_interval": monitor.check_interval,
            "self_heal": monitor.self_heal,
            "message": "Background health monitoring started"
        })
    
    @classmethod
    def _stop(cls, args: Dict[str, Any]) -> str:
        """Stop background health monitoring."""
        monitor = cls.get_monitor()
        monitor.stop_background_monitoring()
        
        return json.dumps({
            "status": "stopped",
            "message": "Background health monitoring stopped"
        })
    
    @classmethod
    def _config(cls, args: Dict[str, Any]) -> str:
        """Get or set configuration."""
        monitor = cls.get_monitor()
        
        if "get" in args:
            return json.dumps({
                "check_interval": monitor.check_interval,
                "self_heal": monitor.self_heal,
                "alert_threshold": monitor.alert_threshold
            })
        elif "set" in args:
            config = args["set"]
            if "check_interval" in config:
                monitor.check_interval = int(config["check_interval"])
            if "self_heal" in config:
                monitor.self_heal = bool(config["self_heal"])
            if "alert_threshold" in config:
                monitor.alert_threshold = int(config["alert_threshold"])
            
            return json.dumps({
                "status": "updated",
                "config": {
                    "check_interval": monitor.check_interval,
                    "self_heal": monitor.self_heal,
                    "alert_threshold": monitor.alert_threshold
                }
            })
        else:
            return json.dumps({
                "error": "Specify 'get' or 'set' parameter"
            })
