"""Agent health monitor - checks API performance, token usage, memory."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import os
import json
import psutil
import logging

from .base import ComponentHealth, HealthStatus, HealthMetric
from .monitor_base import BaseMonitor


logger = logging.getLogger(__name__)


class AgentMonitor(BaseMonitor):
    """Monitor agent health - API response times, token usage, memory."""
    
    def __init__(self):
        super().__init__("agent")
        self._response_times: List[float] = []
        self._token_usage_history: List[Dict[str, Any]] = []
        
    def check(self) -> ComponentHealth:
        """Check agent health."""
        try:
            metrics = []
            details = {}
            errors = []
            
            # Check API connectivity
            api_healthy = self._check_api_connectivity()
            metrics.append(HealthMetric(
                name="api_connectivity",
                status=HealthStatus.HEALTHY if api_healthy else HealthStatus.UNHEALTHY,
                value=api_healthy,
                message="API connectivity OK" if api_healthy else "API connectivity issues"
            ))
            
            if not api_healthy:
                errors.append("API connectivity issues detected")
            
            # Check response times
            avg_response_time = self._check_response_times()
            response_status = HealthStatus.HEALTHY
            if avg_response_time > 10.0:  # >10 seconds
                response_status = HealthStatus.DEGRADED
            elif avg_response_time > 30.0:  # >30 seconds
                response_status = HealthStatus.UNHEALTHY
            
            metrics.append(HealthMetric(
                name="avg_response_time",
                status=response_status,
                value=round(avg_response_time, 2),
                unit="seconds",
                message=f"Average API response time: {avg_response_time:.2f}s"
            ))
            
            # Check memory usage
            memory_usage = self._check_memory_usage()
            memory_status = HealthStatus.HEALTHY
            if memory_usage["percent"] > 80:
                memory_status = HealthStatus.DEGRADED
            elif memory_usage["percent"] > 95:
                memory_status = HealthStatus.UNHEALTHY
            
            metrics.append(HealthMetric(
                name="memory_usage",
                status=memory_status,
                value=f"{memory_usage['percent']:.1f}%",
                unit="percent",
                message=f"Memory usage: {memory_usage['used_gb']:.1f}GB / {memory_usage['total_gb']:.1f}GB"
            ))
            details["memory"] = memory_usage
            
            # Check token usage trends
            token_trend = self._check_token_trends()
            if token_trend["status"] != HealthStatus.HEALTHY:
                metrics.append(HealthMetric(
                    name="token_usage_trend",
                    status=token_trend["status"],
                    value=token_trend["trend"],
                    message=f"Token usage trend: {token_trend['trend']}"
                ))
            
            # Determine overall status
            if any(m.status == HealthStatus.UNHEALTHY for m in metrics):
                health = self._create_unhealthy(
                    error="; ".join(errors) if errors else "Agent health check failed",
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
            logger.error(f"Agent health check failed: {e}")
            return self._create_unhealthy(
                error=f"Health check exception: {str(e)}"
            )
    
    def _check_api_connectivity(self) -> bool:
        """Check if LLM API is reachable."""
        try:
            # Simple connectivity check - try to import and ping
            import http.client
            conn = http.client.HTTPSConnection("api.openrouter.ai", timeout=5)
            conn.request("GET", "/api/v1/models")
            response = conn.getresponse()
            return response.status in (200, 401)  # 401 is OK (auth required)
        except:
            return False
    
    def _check_response_times(self) -> float:
        """Check average API response time."""
        try:
            # Read from recent session logs
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            sessions_dir = os.path.join(hermes_home, "sessions")
            
            if not os.path.exists(sessions_dir):
                return 0.0
            
            # Check recent sessions for timing data
            session_files = sorted(os.listdir(sessions_dir), reverse=True)[:10]
            times = []
            
            for session_file in session_files:
                try:
                    filepath = os.path.join(sessions_dir, session_file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Extract timing information if available
                    if "timing" in data:
                        timing = data["timing"]
                        if "api_call_ms" in timing:
                            times.append(timing["api_call_ms"] / 1000.0)
                except:
                    pass
            
            return sum(times) / len(times) if times else 0.0
            
        except Exception:
            return 0.0
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check process memory usage."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            total_memory = psutil.virtual_memory().total
            
            return {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "used_gb": memory_info.rss / (1024**3),
                "total_gb": total_memory / (1024**3),
                "percent": (memory_info.rss / total_memory) * 100
            }
            
        except Exception:
            return {
                "rss": 0,
                "vms": 0,
                "used_gb": 0,
                "total_gb": 0,
                "percent": 0
            }
    
    def _check_token_trends(self) -> Dict[str, Any]:
        """Check token usage trends."""
        try:
            # Analyze recent sessions for token usage patterns
            hermes_home = os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes"))
            sessions_dir = os.path.join(hermes_home, "sessions")
            
            if not os.path.exists(sessions_dir):
                return {"trend": "stable", "status": HealthStatus.UNKNOWN}
            
            # Simple trend analysis
            session_files = sorted(os.listdir(sessions_dir), reverse=True)[:20]
            token_counts = []
            
            for session_file in session_files:
                try:
                    filepath = os.path.join(sessions_dir, session_file)
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    if "usage" in data:
                        usage = data["usage"]
                        total_tokens = usage.get("total_tokens", 0)
                        token_counts.append(total_tokens)
                except:
                    pass
            
            if len(token_counts) < 2:
                return {"trend": "stable", "status": HealthStatus.UNKNOWN}
            
            # Simple trend: compare recent average to older average
            mid = len(token_counts) // 2
            recent_avg = sum(token_counts[:mid]) / mid
            older_avg = sum(token_counts[mid:]) / (len(token_counts) - mid)
            
            if older_avg > 0:
                change = (recent_avg - older_avg) / older_avg
                if change > 0.5:  # >50% increase
                    return {"trend": "increasing", "status": HealthStatus.DEGRADED}
                elif change > 1.0:  # >100% increase
                    return {"trend": "rapidly_increasing", "status": HealthStatus.UNHEALTHY}
            
            return {"trend": "stable", "status": HealthStatus.HEALTHY}
            
        except Exception:
            return {"trend": "unknown", "status": HealthStatus.UNKNOWN}
