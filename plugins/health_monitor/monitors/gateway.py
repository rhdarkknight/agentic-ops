"""Gateway health monitor - checks connection state, message queue, error rates."""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
import socket
import json
import logging

from .base import ComponentHealth, HealthStatus, HealthMetric
from .monitor_base import BaseMonitor


logger = logging.getLogger(__name__)


class GatewayMonitor(BaseMonitor):
    """Monitor gateway health - connection state, message processing, error rates."""
    
    def __init__(self):
        super().__init__("gateway")
        self._gateway_status_cache: Optional[Dict[str, Any]] = None
        self._last_status_check: Optional[datetime] = None
        
    def check(self) -> ComponentHealth:
        """Check gateway health."""
        try:
            metrics = []
            details = {}
            errors = []
            
            # Check if gateway process is running
            gateway_running = self._check_gateway_process()
            metrics.append(HealthMetric(
                name="process_running",
                status=HealthStatus.HEALTHY if gateway_running else HealthStatus.UNHEALTHY,
                value=gateway_running,
                message="Gateway process is running" if gateway_running else "Gateway process not detected"
            ))
            
            if not gateway_running:
                errors.append("Gateway process is not running")
                return self._create_unhealthy(
                    error="; ".join(errors),
                    metrics=metrics
                )
            
            # Check platform connections
            platform_status = self._check_platform_connections()
            connected_platforms = sum(1 for p in platform_status.values() if p.get("connected", False))
            total_platforms = len(platform_status)
            
            metrics.append(HealthMetric(
                name="platform_connections",
                status=HealthStatus.HEALTHY if connected_platforms == total_platforms else HealthStatus.DEGRADED,
                value=f"{connected_platforms}/{total_platforms}",
                message=f"{connected_platforms} of {total_platforms} platforms connected"
            ))
            details["platforms"] = platform_status
            
            # Check message queue depth
            queue_depth = self._check_message_queue()
            queue_status = HealthStatus.HEALTHY
            if queue_depth > 100:
                queue_status = HealthStatus.DEGRADED
            elif queue_depth > 500:
                queue_status = HealthStatus.UNHEALTHY
            
            metrics.append(HealthMetric(
                name="queue_depth",
                status=queue_status,
                value=queue_depth,
                unit="messages",
                message=f"Message queue depth: {queue_depth}"
            ))
            
            # Check recent error rate
            error_rate = self._check_error_rate()
            error_status = HealthStatus.HEALTHY
            if error_rate > 0.05:  # >5% error rate
                error_status = HealthStatus.DEGRADED
            elif error_rate > 0.15:  # >15% error rate
                error_status = HealthStatus.UNHEALTHY
            
            metrics.append(HealthMetric(
                name="error_rate",
                status=error_status,
                value=round(error_rate, 3),
                unit="ratio",
                message=f"Recent error rate: {error_rate:.1%}"
            ))
            
            # Determine overall status
            if any(m.status == HealthStatus.UNHEALTHY for m in metrics):
                health = self._create_unhealthy(
                    error="; ".join(errors) if errors else "Gateway health check failed",
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
            logger.error(f"Gateway health check failed: {e}")
            return self._create_unhealthy(
                error=f"Health check exception: {str(e)}"
            )
    
    def _check_gateway_process(self) -> bool:
        """Check if gateway process is running."""
        try:
            import subprocess

            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            pid_file = os.path.join(hermes_home, "gateway.pid")

            if os.path.exists(pid_file):
                try:
                    with open(pid_file, 'r') as f:
                        raw = f.read().strip()
                    # PID file may be plain int OR JSON {"pid": 123, ...}
                    try:
                        payload = json.loads(raw)
                        pid = int(payload.get("pid", payload) if isinstance(payload, dict) else payload)
                    except (json.JSONDecodeError, ValueError):
                        pid = int(raw)
                    # Verify process exists AND looks like gateway
                    os.kill(pid, 0)
                    # Validate cmdline to avoid PID reuse false positives
                    try:
                        with open(f"/proc/{pid}/cmdline", "rb") as f:
                            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "ignore")
                        if "hermes" in cmdline and ("gateway" in cmdline or "hermes_cli" in cmdline):
                            return True
                    except (OSError, FileNotFoundError):
                        pass
                    # cmdline unreadable but process exists — be lenient
                    return True
                except (ProcessLookupError, ValueError, KeyError, PermissionError):
                    pass

            # Fallback 1: pgrep for hermes gateway process
            try:
                result = subprocess.run(
                    ["pgrep", "-fa", "hermes_cli.*gateway|hermes.*gateway"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except Exception:
                pass

            # Fallback 2: systemd user service check
            try:
                result = subprocess.run(
                    ["systemctl", "--user", "is-active", "hermes-gateway"],
                    capture_output=True, text=True, timeout=3
                )
                if result.returncode == 0:
                    return True
            except Exception:
                pass

            return False

        except Exception:
            return False
    
    def _check_platform_connections(self) -> Dict[str, Dict[str, Any]]:
        """Check status of platform connections (Telegram, Discord, etc.)."""
        platforms = {}
        
        try:
            # Check for platform status files
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            status_dir = os.path.join(hermes_home, "gateway", "status")
            
            if os.path.exists(status_dir):
                for platform_file in os.listdir(status_dir):
                    if platform_file.endswith('.json'):
                        platform_name = platform_file[:-5]  # Remove .json
                        try:
                            with open(os.path.join(status_dir, platform_file), 'r') as f:
                                status = json.load(f)
                            platforms[platform_name] = {
                                "connected": status.get("connected", False),
                                "last_seen": status.get("last_seen"),
                                "error": status.get("error")
                            }
                        except:
                            platforms[platform_name] = {"connected": False}
        except Exception as e:
            logger.debug(f"Could not read platform status: {e}")
        
        # If no status files found, assume connected if process is running
        if not platforms:
            platforms["default"] = {"connected": True, "note": "No status files found"}
        
        return platforms
    
    def _check_message_queue(self) -> int:
        """Check message queue depth."""
        try:
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            queue_dir = os.path.join(hermes_home, "gateway", "queue")
            
            if os.path.exists(queue_dir):
                return len(os.listdir(queue_dir))
        except:
            pass
        
        return 0
    
    def _check_error_rate(self) -> float:
        """Check recent error rate."""
        try:
            # Check recent error logs
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            log_file = os.path.join(hermes_home, "gateway.log")
            
            if not os.path.exists(log_file):
                return 0.0
            
            # Count errors in last 100 lines
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-100:]
                
                error_count = sum(1 for line in lines if 'ERROR' in line or 'CRITICAL' in line)
                return error_count / len(lines) if lines else 0.0
            except:
                return 0.0
                
        except:
            return 0.0
