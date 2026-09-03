"""Health monitor plugin - registers health monitoring tools and hooks."""

import os
import sys
from pathlib import Path

# Add the plugin directory to the path
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

# Import with fallback for both package and standalone modes
try:
    from monitor import HealthMonitor
    from tool import HealthTool
    from monitors.base import HealthStatus
except ImportError:
    from .monitor import HealthMonitor
    from .tool import HealthTool
    from .monitors.base import HealthStatus


def register(ctx):
    """Register health monitor components."""
    # Register health tool
    ctx.register_tool(
        name="health",
        toolset="monitoring",
        schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "check", "start", "stop", "config"],
                    "description": "Action to perform"
                },
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "default": "text",
                    "description": "Output format"
                }
            },
            "required": ["action"]
        },
        handler=HealthTool.handle,
        description="Health monitoring - check system health, run diagnostics, manage monitoring",
        emoji="❤️"
    )
    
    # Optionally start background monitoring if enabled
    if os.getenv("HERMES_HEALTH_ENABLED", "1") == "1":
        monitor = HealthTool.get_monitor()
        monitor.start_background_monitoring()
