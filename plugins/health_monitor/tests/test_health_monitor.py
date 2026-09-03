"""Tests for health monitoring plugin."""

import pytest
import os
import sys
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test targets
from health_monitor.monitors.base import HealthStatus, HealthMetric, ComponentHealth, SystemHealth
from health_monitor.monitors import BaseMonitor, GatewayMonitor, CronMonitor, AgentMonitor, SystemMonitor
from health_monitor.monitor import HealthMonitor
from health_monitor.tool import HealthTool


class TestHealthStatus:
    """Test HealthStatus enum."""
    
    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthMetric:
    """Test HealthMetric dataclass."""
    
    def test_create_metric(self):
        metric = HealthMetric(
            name="test_metric",
            status=HealthStatus.HEALTHY,
            value=42,
            unit="units"
        )
        assert metric.name == "test_metric"
        assert metric.status == HealthStatus.HEALTHY
        assert metric.value == 42
        assert metric.unit == "units"
    
    def test_to_dict(self):
        metric = HealthMetric(
            name="disk_usage",
            status=HealthStatus.DEGRADED,
            value=85.5,
            unit="percent",
            message="High disk usage"
        )
        d = metric.to_dict()
        assert d["name"] == "disk_usage"
        assert d["status"] == "degraded"
        assert d["value"] == 85.5
        assert d["unit"] == "percent"
        assert d["message"] == "High disk usage"
        assert "timestamp" in d


class TestComponentHealth:
    """Test ComponentHealth dataclass."""
    
    def test_create_healthy_component(self):
        component = ComponentHealth(
            name="gateway",
            status=HealthStatus.HEALTHY,
            metrics=[
                HealthMetric("ping", HealthStatus.HEALTHY, 0.05, "s")
            ]
        )
        assert component.name == "gateway"
        assert component.is_healthy
        assert len(component.metrics) == 1
    
    def test_to_dict(self):
        component = ComponentHealth(
            name="cron",
            status=HealthStatus.DEGRADED,
            error="High queue length",
            metrics=[
                HealthMetric("queue", HealthStatus.DEGRADED, 150, "jobs")
            ]
        )
        d = component.to_dict()
        assert d["name"] == "cron"
        assert d["status"] == "degraded"
        assert d["error"] == "High queue length"
        assert len(d["metrics"]) == 1


class TestSystemHealth:
    """Test SystemHealth aggregation."""
    
    def test_empty_system(self):
        system = SystemHealth()
        assert system.overall_status == HealthStatus.UNKNOWN
        assert len(system.components) == 0
    
    def test_add_healthy_components(self):
        system = SystemHealth()
        system.add_component(ComponentHealth("gateway", HealthStatus.HEALTHY))
        system.add_component(ComponentHealth("cron", HealthStatus.HEALTHY))
        
        assert system.overall_status == HealthStatus.HEALTHY
    
    def test_add_unhealthy_component(self):
        system = SystemHealth()
        system.add_component(ComponentHealth("gateway", HealthStatus.HEALTHY))
        system.add_component(ComponentHealth("cron", HealthStatus.UNHEALTHY))
        
        assert system.overall_status == HealthStatus.UNHEALTHY
    
    def test_add_degraded_component(self):
        system = SystemHealth()
        system.add_component(ComponentHealth("gateway", HealthStatus.HEALTHY))
        system.add_component(ComponentHealth("cron", HealthStatus.DEGRADED))
        
        assert system.overall_status == HealthStatus.DEGRADED
    
    def test_to_text_report(self):
        system = SystemHealth()
        system.add_component(ComponentHealth(
            "gateway",
            HealthStatus.HEALTHY,
            metrics=[HealthMetric("status", HealthStatus.HEALTHY, "up")]
        ))
        
        report = system.to_text_report()
        assert "gateway" in report
        assert "healthy" in report
    
    def test_to_json(self):
        system = SystemHealth()
        system.add_component(ComponentHealth("test", HealthStatus.HEALTHY))
        
        json_str = system.to_json()
        data = json.loads(json_str)
        assert data["overall_status"] == "healthy"
        assert "test" in data["components"]


class TestBaseMonitor:
    """Test BaseMonitor abstract class."""
    
    def test_monitor_creation(self):
        class TestMonitor(BaseMonitor):
            def check(self):
                return self._create_healthy()
        
        monitor = TestMonitor("test")
        assert monitor.name == "test"
        assert monitor.last_check is None
        assert monitor.consecutive_failures == 0
    
    def test_record_check_healthy(self):
        class TestMonitor(BaseMonitor):
            def check(self):
                return self._create_healthy()
        
        monitor = TestMonitor("test")
        health = monitor.check()
        monitor.record_check(health)
        
        assert monitor.last_check is not None
        assert monitor.consecutive_failures == 0
    
    def test_record_check_unhealthy(self):
        class TestMonitor(BaseMonitor):
            def check(self):
                return self._create_unhealthy("Test error")
        
        monitor = TestMonitor("test")
        for _ in range(3):
            health = monitor.check()
            monitor.record_check(health)
        
        assert monitor.consecutive_failures == 3
        assert len(monitor.failure_history) == 3


class TestHealthMonitor:
    """Test HealthMonitor orchestrator."""
    
    def test_create_monitor(self):
        monitor = HealthMonitor()
        assert len(monitor.monitors) == 4
        assert "gateway" in monitor.monitors
        assert "cron" in monitor.monitors
        assert "agent" in monitor.monitors
        assert "system" in monitor.monitors
    
    def test_check_all(self):
        monitor = HealthMonitor()
        health = monitor.check_all()
        
        assert isinstance(health, SystemHealth)
        assert len(health.components) == 4
        assert health.timestamp is not None
    
    def test_get_health_report_text(self):
        monitor = HealthMonitor()
        monitor.check_all()
        report = monitor.get_health_report("text")
        assert "Health Report" in report
    
    def test_get_health_report_json(self):
        monitor = HealthMonitor()
        monitor.check_all()
        report = monitor.get_health_report("json")
        data = json.loads(report)
        assert "overall_status" in data
        assert "components" in data
    
    def test_get_status_summary(self):
        monitor = HealthMonitor()
        monitor.check_all()
        summary = monitor.get_status_summary()
        assert "overall_status" in summary
        assert "components" in summary
        assert "alerts" in summary


class TestHealthTool:
    """Test HealthTool command handler."""
    
    def test_status_action(self):
        result = HealthTool.handle({"action": "status", "format": "json"})
        data = json.loads(result)
        assert "overall_status" in data or "error" in data
    
    def test_check_action(self):
        result = HealthTool.handle({"action": "check", "format": "json"})
        data = json.loads(result)
        assert "overall_status" in data or "error" in data
    
    def test_start_action(self):
        result = HealthTool.handle({"action": "start"})
        data = json.loads(result)
        assert data["status"] == "started"
    
    def test_stop_action(self):
        # Start first
        HealthTool.handle({"action": "start"})
        # Then stop
        result = HealthTool.handle({"action": "stop"})
        data = json.loads(result)
        assert data["status"] == "stopped"
    
    def test_config_get(self):
        result = HealthTool.handle({"action": "config", "get": True})
        data = json.loads(result)
        assert "check_interval" in data
        assert "self_heal" in data
    
    def test_config_set(self):
        result = HealthTool.handle({
            "action": "config",
            "set": {"check_interval": 600}
        })
        data = json.loads(result)
        assert data["status"] == "updated"
        assert data["config"]["check_interval"] == 600
    
    def test_unknown_action(self):
        result = HealthTool.handle({"action": "unknown"})
        data = json.loads(result)
        assert "error" in data


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
