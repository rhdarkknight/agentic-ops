"""Cron health monitor - checks scheduler, job execution, queue."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
import json
import glob
import logging

from .base import ComponentHealth, HealthStatus, HealthMetric
from .monitor_base import BaseMonitor


logger = logging.getLogger(__name__)


class CronMonitor(BaseMonitor):
    """Monitor cron scheduler health - ticker, job execution, queue."""
    
    def __init__(self):
        super().__init__("cron")
        self._job_history: List[Dict[str, Any]] = []
        
    def check(self) -> ComponentHealth:
        """Check cron health."""
        try:
            metrics = []
            details = {}
            errors = []
            
            # Check if scheduler is running
            scheduler_running = self._check_scheduler_process()
            metrics.append(HealthMetric(
                name="scheduler_running",
                status=HealthStatus.HEALTHY if scheduler_running else HealthStatus.UNHEALTHY,
                value=scheduler_running,
                message="Cron scheduler is running" if scheduler_running else "Cron scheduler not detected"
            ))
            
            if not scheduler_running:
                errors.append("Cron scheduler is not running")
                return self._create_unhealthy(
                    error="; ".join(errors),
                    metrics=metrics
                )
            
            # Check job queue
            queue_status = self._check_job_queue()
            metrics.append(HealthMetric(
                name="queue_length",
                status=queue_status["status"],
                value=queue_status["count"],
                unit="jobs",
                message=f"Job queue length: {queue_status['count']}"
            ))
            details["queue"] = queue_status
            
            # Check recent job execution
            job_stats = self._check_recent_jobs()
            metrics.append(HealthMetric(
                name="recent_jobs",
                status=job_stats["status"],
                value=f"{job_stats['success']}/{job_stats['total']}",
                message=f"Recent job success rate: {job_stats['success_rate']:.1%}"
            ))
            details["recent_jobs"] = job_stats
            
            # Check for stalled jobs
            stalled_jobs = self._check_stalled_jobs()
            if stalled_jobs > 0:
                metrics.append(HealthMetric(
                    name="stalled_jobs",
                    status=HealthStatus.DEGRADED if stalled_jobs < 3 else HealthStatus.UNHEALTHY,
                    value=stalled_jobs,
                    unit="jobs",
                    message=f"Stalled jobs detected: {stalled_jobs}"
                ))
            
            # Determine overall status
            if any(m.status == HealthStatus.UNHEALTHY for m in metrics):
                health = self._create_unhealthy(
                    error="; ".join(errors) if errors else "Cron health check failed",
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
            logger.error(f"Cron health check failed: {e}")
            return self._create_unhealthy(
                error=f"Health check exception: {str(e)}"
            )
    
    def _check_scheduler_process(self) -> bool:
        """Check if cron scheduler is running (embedded in gateway process)."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))

            # Cron scheduler runs inside gateway — check gateway PID
            gateway_pid_file = os.path.join(hermes_home, "gateway.pid")
            if os.path.exists(gateway_pid_file):
                try:
                    with open(gateway_pid_file, 'r') as f:
                        content = f.read().strip()
                    # gateway.pid is JSON: {"pid": N, ...}
                    import json
                    data = json.loads(content)
                    pid = int(data.get("pid", 0))
                    if pid:
                        os.kill(pid, 0)
                        return True
                except (ProcessLookupError, ValueError, KeyError, ImportError):
                    pass

            # Fallback: check if gateway process is running by name
            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", "gateway.*run"],
                capture_output=True, text=True
            )
            return bool(result.stdout.strip())

        except Exception:
            return False
    
    def _check_job_queue(self) -> Dict[str, Any]:
        """Check job queue status."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            queue_dir = os.path.join(hermes_home, "cron", "queue")
            
            count = 0
            if os.path.exists(queue_dir):
                count = len(os.listdir(queue_dir))
            
            status = HealthStatus.HEALTHY
            if count > 50:
                status = HealthStatus.DEGRADED
            elif count > 200:
                status = HealthStatus.UNHEALTHY
            
            return {
                "count": count,
                "status": status
            }
            
        except Exception:
            return {"count": 0, "status": HealthStatus.UNKNOWN}
    
    def _check_recent_jobs(self) -> Dict[str, Any]:
        """Check recent job execution statistics."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            output_dir = os.path.join(hermes_home, "cron", "output")
            
            if not os.path.exists(output_dir):
                return {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "success_rate": 0.0,
                    "status": HealthStatus.UNKNOWN
                }
            
            # Check last 20 job outputs
            job_files = sorted(glob.glob(os.path.join(output_dir, "*.json")), reverse=True)[:20]
            
            total = len(job_files)
            success = 0
            failed = 0
            
            for job_file in job_files:
                try:
                    with open(job_file, 'r') as f:
                        data = json.load(f)
                    if data.get("status") == "ok":
                        success += 1
                    else:
                        failed += 1
                except:
                    failed += 1
            
            success_rate = success / total if total > 0 else 0.0
            
            status = HealthStatus.HEALTHY
            if success_rate < 0.8:
                status = HealthStatus.DEGRADED
            elif success_rate < 0.5:
                status = HealthStatus.UNHEALTHY
            
            return {
                "total": total,
                "success": success,
                "failed": failed,
                "success_rate": success_rate,
                "status": status
            }
            
        except Exception:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "success_rate": 0.0,
                "status": HealthStatus.UNKNOWN
            }
    
    def _check_stalled_jobs(self) -> int:
        """Check for stalled jobs (running too long)."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            running_dir = os.path.join(hermes_home, "cron", "running")
            
            if not os.path.exists(running_dir):
                return 0
            
            stalled = 0
            now = datetime.utcnow()
            
            for job_file in os.listdir(running_dir):
                if job_file.endswith('.json'):
                    try:
                        filepath = os.path.join(running_dir, job_file)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        started = datetime.fromisoformat(data.get("started", ""))
                        if (now - started).total_seconds() > 3600:  # >1 hour
                            stalled += 1
                    except:
                        pass
            
            return stalled
            
        except Exception:
            return 0
