"""System health monitor - checks disk space, file growth, log rotation."""

from typing import Dict, Any
from datetime import datetime
import os
import shutil
import glob
import logging

from .base import ComponentHealth, HealthStatus, HealthMetric
from .monitor_base import BaseMonitor


logger = logging.getLogger(__name__)


class SystemMonitor(BaseMonitor):
    """Monitor system health - disk space, session files, log rotation."""
    
    def __init__(self):
        super().__init__("system")
        
    def check(self) -> ComponentHealth:
        """Check system health."""
        try:
            metrics = []
            details = {}
            errors = []
            
            # Check disk space
            disk_usage = self._check_disk_space()
            disk_status = HealthStatus.HEALTHY
            if disk_usage["percent"] > 85:
                disk_status = HealthStatus.DEGRADED
            elif disk_usage["percent"] > 95:
                disk_status = HealthStatus.UNHEALTHY
                errors.append(f"Critical disk space: {disk_usage['percent']:.1f}% used")
            
            metrics.append(HealthMetric(
                name="disk_usage",
                status=disk_status,
                value=f"{disk_usage['percent']:.1f}%",
                unit="percent",
                message=f"Disk usage: {disk_usage['used_gb']:.1f}GB / {disk_usage['total_gb']:.1f}GB"
            ))
            details["disk"] = disk_usage
            
            # Check session file growth
            session_stats = self._check_session_files()
            session_status = HealthStatus.HEALTHY
            if session_stats["count"] > 1000:
                session_status = HealthStatus.DEGRADED
            elif session_stats["count"] > 5000:
                session_status = HealthStatus.UNHEALTHY
                errors.append(f"Excessive session files: {session_stats['count']}")
            
            metrics.append(HealthMetric(
                name="session_files",
                status=session_status,
                value=session_stats["count"],
                unit="files",
                message=f"Session files: {session_stats['count']} ({session_stats['size_mb']:.1f}MB)"
            ))
            details["sessions"] = session_stats
            
            # Check log file sizes
            log_stats = self._check_log_files()
            if log_stats["total_size_mb"] > 1024:  # >1GB total logs
                log_status = HealthStatus.DEGRADED
            else:
                log_status = HealthStatus.HEALTHY
            
            metrics.append(HealthMetric(
                name="log_files",
                status=log_status,
                value=f"{log_stats['total_size_mb']:.0f}MB",
                unit="MB",
                message=f"Total log size: {log_stats['total_size_mb']:.0f}MB across {log_stats['count']} files"
            ))
            details["logs"] = log_stats
            
            # Check memory pressure
            memory_stats = self._check_memory_pressure()
            memory_status = HealthStatus.HEALTHY
            if memory_stats["available_percent"] < 20:
                memory_status = HealthStatus.DEGRADED
            elif memory_stats["available_percent"] < 10:
                memory_status = HealthStatus.UNHEALTHY
                errors.append(f"Low available memory: {memory_stats['available_percent']:.1f}%")
            
            metrics.append(HealthMetric(
                name="memory_pressure",
                status=memory_status,
                value=f"{memory_stats['available_percent']:.1f}%",
                unit="percent",
                message=f"Available memory: {memory_stats['available_gb']:.1f}GB"
            ))
            details["memory"] = memory_stats
            
            # Determine overall status
            if any(m.status == HealthStatus.UNHEALTHY for m in metrics):
                health = self._create_unhealthy(
                    error="; ".join(errors) if errors else "System health check failed",
                    metrics=metrics,
                    details=details
                )
            elif any(m.status == HealthStatus.DEGRADED for m in metrics):
                health = self._create_degraded(
                    metrics=metrics,
                    details=details
                )
            else:
                health = self._create_healthy(
                    metrics=metrics,
                    details=details
                )
            
            return health
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return self._create_unhealthy(
                error=f"Health check exception: {str(e)}"
            )
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            usage = shutil.disk_usage(hermes_home)
            
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            percent = (usage.used / usage.total) * 100
            
            return {
                "total_gb": round(total_gb, 1),
                "used_gb": round(used_gb, 1),
                "free_gb": round(usage.free / (1024**3), 1),
                "percent": round(percent, 1)
            }
            
        except Exception as e:
            logger.debug(f"Could not check disk space: {e}")
            return {
                "total_gb": 0,
                "used_gb": 0,
                "free_gb": 0,
                "percent": 0
            }
    
    def _check_session_files(self) -> Dict[str, Any]:
        """Check session file count and size."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            sessions_dir = os.path.join(hermes_home, "sessions")
            
            if not os.path.exists(sessions_dir):
                return {"count": 0, "size_mb": 0}
            
            files = os.listdir(sessions_dir)
            count = len(files)
            total_size = sum(
                os.path.getsize(os.path.join(sessions_dir, f))
                for f in files if os.path.isfile(os.path.join(sessions_dir, f))
            )
            
            return {
                "count": count,
                "size_mb": round(total_size / (1024**2), 1)
            }
            
        except Exception:
            return {"count": 0, "size_mb": 0}
    
    def _check_log_files(self) -> Dict[str, Any]:
        """Check log file sizes."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            
            # Find all log files
            log_patterns = [
                os.path.join(hermes_home, "*.log"),
                os.path.join(hermes_home, "logs", "*.log"),
                os.path.join(hermes_home, "gateway", "*.log"),
                os.path.join(hermes_home, "cron", "*.log"),
            ]
            
            log_files = []
            for pattern in log_patterns:
                log_files.extend(glob.glob(pattern))
            
            count = len(log_files)
            total_size = sum(os.path.getsize(f) for f in log_files if os.path.isfile(f))
            
            return {
                "count": count,
                "total_size_mb": round(total_size / (1024**2), 1)
            }
            
        except Exception:
            return {"count": 0, "total_size_mb": 0}
    
    def _check_memory_pressure(self) -> Dict[str, Any]:
        """Check system memory pressure."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            available_percent = memory.available / memory.total * 100
            
            return {
                "total_gb": round(total_gb, 1),
                "available_gb": round(available_gb, 1),
                "used_gb": round((memory.total - memory.available) / (1024**3), 1),
                "available_percent": round(available_percent, 1)
            }
            
        except Exception:
            return {
                "total_gb": 0,
                "available_gb": 0,
                "used_gb": 0,
                "available_percent": 0
            }
