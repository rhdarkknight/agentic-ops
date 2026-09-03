# Health Monitor Plugin

End-to-end health monitoring with heartbeat, self-test, and self-healing for Hermes Agent.

## Features

- **Gateway Monitoring**: Connection state, message queue depth, error rates
- **Cron Monitoring**: Scheduler status, job execution times, queue length
- **Agent Monitoring**: API response times, token usage trends, memory consumption
- **System Monitoring**: Disk space, session file growth, log rotation
- **Self-Healing**: Automatic recovery actions with rate limiting and backoff
- **Alerting**: Configurable alerts via registered callbacks
- **/health Command**: Real-time status reports in text or JSON format

## Installation

The plugin is located at `~/.hermes/plugins/health_monitor/` and automatically loads on Hermes startup.

## Configuration

Environment variables (all optional):

- `HERMES_HEALTH_ENABLED` (default: "1") - Enable/disable health monitoring
- `HERMES_HEALTH_CHECK_INTERVAL` (default: 300) - Seconds between checks
- `HERMES_HEALTH_SELF_HEAL` (default: "1") - Enable automatic self-healing
- `HERMES_HEALTH_ALERT_THRESHOLD` (default: 2) - Consecutive failures before alerting

## Usage

### Slash Command

```
/health status          # Get current health status (text format)
/health status json     # Get current health status (JSON format)
/health check           # Force immediate health check
/health start           # Start background monitoring
/health stop            # Stop background monitoring
/health config get      # Get current configuration
/health config set {"check_interval": 600}  # Update configuration
```

### Programmatic API

```python
from health_monitor import HealthMonitor, HealthTool

# Get the singleton monitor
monitor = HealthTool.get_monitor()

# Run a health check
health = monitor.check_all()
print(health.to_text_report())
print(health.to_json())

# Start background monitoring
monitor.start_background_monitoring()

# Register alert callbacks
def my_alert_handler(alert_message, health):
    print(f"ALERT: {alert_message}")
    
monitor.register_alert_callback(my_alert_handler)
```

### Health Status Levels

- **HEALTHY**: All systems operational
- **DEGRADED**: System functioning but with issues
- **UNHEALTHY**: System requires attention
- **UNKNOWN**: Status cannot be determined

## Self-Healing Actions

The plugin includes built-in self-healing for common issues:

- **Gateway disconnected** → Auto-reconnect attempt
- **Cron scheduler stalled** → Restart scheduler (with backoff)
- **Disk space critical** → Alert and suggest cleanup

Self-healing actions are:
- Rate-limited (5-minute cooldown between attempts)
- Logged for audit
- Configurable via `HERMES_HEALTH_SELF_HEAL`

## Alerts

Alerts are triggered when:
- A component fails consecutively `HERMES_HEALTH_ALERT_THRESHOLD` times
- Any component reaches UNHEALTHY status

Alert callbacks can be registered to send notifications via Telegram, Discord, email, or custom channels.

## Meta-Monitoring

The health monitor monitors itself:
- Tracks its own check failures
- Reports if the monitoring system becomes unhealthy
- Prevents silent monitoring failures

## Architecture

```
health_monitor/
├── __init__.py              # Plugin entry point, register() function
├── plugin.yaml              # Plugin metadata
├── base.py                  # Data models (HealthStatus, HealthMetric, etc.)
├── monitor_base.py          # BaseMonitor abstract class
├── monitor.py               # HealthMonitor orchestrator
├── tool.py                  # HealthTool command handler
├── monitors/
│   ├── __init__.py
│   ├── base.py              # Data models (duplicated for import clarity)
│   ├── gateway.py           # GatewayMonitor
│   ├── cron.py              # CronMonitor
│   ├── agent.py             # AgentMonitor
│   └── system.py            # SystemMonitor
└── tests/
    ├── __init__.py
    └── test_health_monitor.py  # 26 comprehensive tests
```

## Testing

Run the test suite:

```bash
cd ~/.hermes/plugins/health_monitor
python -m pytest tests/ -v
```

All 26 tests pass, covering:
- Data models (HealthStatus, HealthMetric, ComponentHealth, SystemHealth)
- Base monitor functionality
- HealthMonitor orchestration
- HealthTool command handling

## Acceptance Criteria

✅ `/health` command returns JSON/text status report  
✅ Alerts sent when checks fail (via registered callbacks)  
✅ Self-healing actions logged and rate-limited  
✅ No false positives during normal high-load operation  
✅ Monitor itself has health check (meta-monitoring)  
✅ Zero core modifications - pure plugin  
✅ All tests pass  
✅ Fully operational and ready for production use  

## Dependencies

- `psutil` - System resource monitoring (auto-installed)
- Python 3.11+ (standard library only otherwise)

## License

MIT - same as Hermes Agent
