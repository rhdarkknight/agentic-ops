"""Health monitor orchestrator - coordinates checks, scheduling, and self-healing."""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import threading
import time
import json
import os
import logging

# Import with fallback for both package and standalone modes
try:
    from .monitors.base import SystemHealth, HealthStatus, ComponentHealth
    from .monitors.gateway import GatewayMonitor
    from .monitors.cron import CronMonitor
    from .monitors.agent import AgentMonitor
    from .monitors.system import SystemMonitor
except ImportError:
    from monitors.base import SystemHealth, HealthStatus, ComponentHealth
    from monitors.gateway import GatewayMonitor
    from monitors.cron import CronMonitor
    from monitors.agent import AgentMonitor
    from monitors.system import SystemMonitor


logger = logging.getLogger(__name__)


class HealthMonitor:
    """Orchestrates health monitoring across all components."""
    
    def __init__(self, check_interval: int = 300, self_heal: bool = True, alert_threshold: int = 2):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            self_heal: Enable automatic self-healing actions
            alert_threshold: Number of consecutive failures before alerting
        """
        self.check_interval = check_interval
        self.self_heal = self_heal
        self.alert_threshold = alert_threshold
        
        # Initialize monitors
        self.monitors = {
            "gateway": GatewayMonitor(),
            "cron": CronMonitor(),
            "agent": AgentMonitor(),
            "system": SystemMonitor(),
        }
        
        # State
        self.current_health = SystemHealth()
        self.last_check: Optional[datetime] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[str, SystemHealth], None]] = []
        
        # Self-healing actions
        self._self_heal_actions: Dict[str, Callable[[ComponentHealth], bool]] = {}
        self._last_heal_time: Dict[str, datetime] = {}
        self._heal_cooldown = timedelta(minutes=5)  # Cooldown between heal attempts
        
        # Meta-monitoring
        self._health_check_failures = 0
        self._last_meta_check: Optional[datetime] = None
        
        # Register default self-healing actions
        self._register_default_self_heal_actions()
    
    def register_alert_callback(self, callback: Callable[[str, SystemHealth], None]):
        """Register a callback for alerts."""
        self._alert_callbacks.append(callback)
    
    def register_self_heal_action(self, component: str, action: Callable[[ComponentHealth], bool]):
        """Register a self-healing action for a component."""
        self._self_heal_actions[component] = action
    
    def _register_default_self_heal_actions(self):
        """Register default self-healing actions."""
        # Gateway reconnect
        self.register_self_heal_action("gateway", self._heal_gateway_reconnect)
        
        # Cron restart
        self.register_self_heal_action("cron", self._heal_cron_restart)
        
        # Disk cleanup suggestion
        self.register_self_heal_action("system", self._heal_system_cleanup)
    
    def check_all(self) -> SystemHealth:
        """Run health checks on all components."""
        try:
            health = SystemHealth()
            alerts = []
            
            for name, monitor in self.monitors.items():
                try:
                    component_health = monitor.check()
                    health.add_component(component_health)
                    monitor.record_check(component_health)
                    
                    # Check if alerting is needed
                    if monitor.consecutive_failures >= self.alert_threshold:
                        alert_msg = f"{name} is {component_health.status.value}: {component_health.error or ''}"
                        alerts.append(alert_msg)
                    
                    # Attempt self-healing if enabled and component is unhealthy
                    if self.self_heal and component_health.status == HealthStatus.UNHEALTHY:
                        self._attempt_self_heal(name, component_health)
                    
                except Exception as e:
                    logger.error(f"Health check failed for {name}: {e}")
                    health.add_component(ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        error=str(e),
                        last_check=datetime.utcnow()
                    ))
            
            health.alerts = alerts
            health.timestamp = datetime.utcnow()
            self.current_health = health
            self.last_check = datetime.utcnow()
            
            # Trigger alerts
            if alerts:
                self._trigger_alerts(alerts, health)
            
            # Meta-monitoring: check if health checks themselves are failing
            self._check_meta_health()
            
            return health
            
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            self._health_check_failures += 1
            return SystemHealth(
                overall_status=HealthStatus.UNHEALTHY,
                alerts=[f"Health monitoring system error: {str(e)}"]
            )
    
    def _attempt_self_heal(self, component: str, health: ComponentHealth):
        """Attempt self-healing for an unhealthy component."""
        if component not in self._self_heal_actions:
            return
        
        # Check cooldown
        last_heal = self._last_heal_time.get(component, datetime.min)
        if datetime.utcnow() - last_heal < self._heal_cooldown:
            logger.debug(f"Self-heal for {component} in cooldown")
            return
        
        action = self._self_heal_actions[component]
        try:
            success = action(health)
            if success:
                logger.info(f"Self-healing succeeded for {component}")
                self._last_heal_time[component] = datetime.utcnow()
            else:
                logger.warning(f"Self-healing failed for {component}")
        except Exception as e:
            logger.error(f"Self-healing action failed for {component}: {e}")
    
    def _heal_gateway_reconnect(self, health: ComponentHealth) -> bool:
        """Attempt to reconnect gateway via systemd restart."""
        import subprocess
        try:
            logger.info("Self-heal: restarting gateway via systemd...")
            result = subprocess.run(
                ["systemctl", "--user", "restart", "hermes-gateway"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                logger.info("Self-heal: gateway systemd restart succeeded")
                self._last_heal_time["gateway"] = datetime.utcnow()
                return True
            logger.warning(f"Self-heal: gateway restart failed: {result.stderr[:200]}")
        except FileNotFoundError:
            logger.error("Self-heal: systemctl not found")
        except subprocess.TimeoutExpired:
            logger.error("Self-heal: gateway restart timed out")
        except Exception as e:
            logger.error(f"Self-heal: gateway restart error: {e}")
        return False
    
    def _heal_cron_restart(self, health: ComponentHealth) -> bool:
        """Attempt to restart cron scheduler by restarting gateway.

        Cron ticker runs as a daemon thread inside the gateway process.
        Cannot be restarted independently — must restart the gateway.
        """
        try:
            logger.info("Attempting cron scheduler recovery via gateway restart...")

            import subprocess

            # Use 'hermes gateway restart' which handles graceful shutdown
            # and replaces the running gateway process
            result = subprocess.run(
                ["hermes", "gateway", "restart"],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                logger.info("Gateway restart succeeded — cron scheduler should be recovered")
                self._last_heal_time["cron"] = datetime.utcnow()
                return True
            else:
                logger.error(f"Gateway restart failed (exit {result.returncode}): {result.stderr[:200]}")
                return False

        except FileNotFoundError:
            logger.error("'hermes' CLI not found in PATH — cannot restart gateway")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Gateway restart timed out (60s)")
            return False
        except Exception as e:
            logger.error(f"Cron restart failed: {e}")
            return False
    
    def _heal_system_cleanup(self, health: ComponentHealth) -> bool:
        """Attempt system cleanup (disk space)."""
        try:
            # Check if disk space is the issue
            disk_metric = next((m for m in health.metrics if m.name == "disk_usage"), None)
            if disk_metric and disk_metric.status == HealthStatus.UNHEALTHY:
                logger.info("Suggesting system cleanup due to low disk space")
                # Could implement automatic log rotation or old session cleanup
                return False  # Manual intervention recommended
            return True  # No action needed
        except Exception:
            return False
    
    def _trigger_alerts(self, alerts: List[str], health: SystemHealth):
        """Trigger alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                for alert in alerts:
                    callback(alert, health)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _check_meta_health(self):
        """Meta-monitoring: check if the health monitor itself is healthy."""
        if self._last_meta_check and datetime.utcnow() - self._last_meta_check < timedelta(minutes=5):
            return
        
        self._last_meta_check = datetime.utcnow()
        
        if self._health_check_failures > 5:
            logger.error("Health monitor itself may be unhealthy - too many check failures")
            self._health_check_failures = 0  # Reset counter
    
    def start_background_monitoring(self):
        """Start background health monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitoring started in background")
    
    def stop_background_monitoring(self):
        """Stop background health monitoring."""
        self.running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.running and not self._stop_event.is_set():
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next check or stop event
            self._stop_event.wait(timeout=self.check_interval)
    
    def get_health_report(self, format: str = "text") -> str:
        """Get health report in specified format."""
        if format == "json":
            return self.current_health.to_json()
        elif format == "text":
            return self.current_health.to_text_report()
        else:
            return str(self.current_health.to_dict())
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a concise status summary for CLI/gateway display."""
        return {
            "overall_status": self.current_health.overall_status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "components": {
                name: {
                    "status": comp.status.value,
                    "last_check": comp.last_check.isoformat() if comp.last_check else None,
                    "error": comp.error
                }
                for name, comp in self.current_health.components.items()
            },
            "alerts": self.current_health.alerts
        }
