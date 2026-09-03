"""Base monitor class for health monitoring system."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .base import ComponentHealth, HealthStatus, HealthMetric


logger = logging.getLogger(__name__)


class BaseMonitor(ABC):
    """Abstract base class for health monitors."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_check: Optional[datetime] = None
        self.consecutive_failures: int = 0
        self.failure_history: List[datetime] = []
        self.max_failure_history: int = 100
        
    @abstractmethod
    def check(self) -> ComponentHealth:
        """Perform health check and return component health status."""
        pass
    
    def _create_healthy(self, metrics: List[HealthMetric] = None, details: Dict[str, Any] = None) -> ComponentHealth:
        """Create a healthy component status."""
        return ComponentHealth(
            name=self.name,
            status=HealthStatus.HEALTHY,
            metrics=metrics or [],
            last_check=datetime.utcnow(),
            details=details or {}
        )
    
    def _create_degraded(self, metrics: List[HealthMetric] = None, error: str = None, details: Dict[str, Any] = None) -> ComponentHealth:
        """Create a degraded component status."""
        return ComponentHealth(
            name=self.name,
            status=HealthStatus.DEGRADED,
            metrics=metrics or [],
            last_check=datetime.utcnow(),
            error=error,
            details=details or {}
        )
    
    def _create_unhealthy(self, error: str, metrics: List[HealthMetric] = None, details: Dict[str, Any] = None) -> ComponentHealth:
        """Create an unhealthy component status."""
        return ComponentHealth(
            name=self.name,
            status=HealthStatus.UNHEALTHY,
            metrics=metrics or [],
            last_check=datetime.utcnow(),
            error=error,
            details=details or {}
        )
    
    def record_check(self, health: ComponentHealth):
        """Record the result of a health check."""
        self.last_check = datetime.utcnow()
        
        if health.status != HealthStatus.HEALTHY:
            self.consecutive_failures += 1
            self.failure_history.append(self.last_check)
            if len(self.failure_history) > self.max_failure_history:
                self.failure_history = self.failure_history[-self.max_failure_history:]
        else:
            self.consecutive_failures = 0
    
    def get_failure_rate(self, window_minutes: int = 60) -> float:
        """Calculate failure rate over the specified time window."""
        if not self.failure_history:
            return 0.0
        
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_failures = [f for f in self.failure_history if f > cutoff]
        
        return len(recent_failures) / window_minutes if window_minutes > 0 else 0.0
