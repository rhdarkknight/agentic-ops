"""Health status data model and base classes for the health monitoring system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    status: HealthStatus
    value: Any
    unit: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "unit": self.unit,
            "message": self.message,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "last_check": self.last_check.isoformat(),
            "error": self.error,
            "details": self.details
        }
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health aggregation."""
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    timestamp: datetime = field(default_factory=datetime.utcnow)
    alerts: List[str] = field(default_factory=list)
    
    def add_component(self, component: ComponentHealth):
        self.components[component.name] = component
        self._recalculate_overall()
    
    def _recalculate_overall(self):
        """Recalculate overall status based on components."""
        if not self.components:
            self.overall_status = HealthStatus.UNKNOWN
            return
        
        statuses = [c.status for c in self.components.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            self.overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            self.overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            self.overall_status = HealthStatus.DEGRADED
        else:
            self.overall_status = HealthStatus.UNKNOWN
    
    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "timestamp": self.timestamp.isoformat(),
            "alerts": self.alerts
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def to_text_report(self) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append(f"Health Report - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 60)
        lines.append(f"Overall Status: {self.overall_status.value.upper()}")
        lines.append("")
        
        for name, component in sorted(self.components.items()):
            status_icon = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.UNHEALTHY: "✗",
                HealthStatus.UNKNOWN: "?"
            }.get(component.status, "?")
            
            lines.append(f"[{status_icon}] {name}: {component.status.value}")
            
            if component.metrics:
                for metric in component.metrics:
                    unit_str = f" {metric.unit}" if metric.unit else ""
                    lines.append(f"    - {metric.name}: {metric.value}{unit_str}")
            
            if component.error:
                lines.append(f"    ERROR: {component.error}")
            
            lines.append("")
        
        if self.alerts:
            lines.append("Alerts:")
            for alert in self.alerts:
                lines.append(f"  ⚠ {alert}")
        
        return "\n".join(lines)
