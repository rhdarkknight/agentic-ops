# Rate Limit Guard Plugin

## Purpose

Mitigates API rate limits for your primary model (e.g., `kwaipilot/kat-coder-pro-v2`) without changing the model configuration. Automatically handles 429 responses with caching, queuing, exponential backoff, and fallback routing.

## Features

| Feature | What It Does |
|---------|-------------|
| **Response caching** | Dedupes identical prompts within TTL (default 1h) — avoids repeat API calls |
| **Request queuing** | Buffers non-urgent requests during rate limit windows, processes later |
| **Exponential backoff** | Retries with increasing delays + jitter before giving up |
| **Automatic fallback** | After N retries, routes to a cheaper/faster secondary model |
| **Priority routing** | Critical requests bypass queue, get immediate backoff+retry |
| **Session-local state** | Tracks per-session stats: cache hits, retries, fallbacks |

## Installation

Plugin is at `~/.hermes/plugins/rate_limit_guard/`. Just restart Hermes —
plugin discovery is automatic.

```bash
# Verify plugin loaded
python3 -c "from plugins.rate_limit_guard import register; print('OK')"
```

## Configuration

All via environment variables in `~/.hermes/.env` or your shell:

| Variable | Default | Description |
|----------|---------|-------------|
| `HERMES_RATE_LIMIT_GUARD_ENABLED` | `1` | Toggle on/off |
| `HERMES_RATE_LIMIT_GUARD_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `HERMES_RATE_LIMIT_GUARD_FALLBACK_MODEL` | `qwen/qwen3.6-plus` | Fallback model when primary is limited |
| `HERMES_RATE_LIMIT_GUARD_MAX_QUEUE_SIZE` | `50` | Max pending queued requests |
| `HERMES_RATE_LIMIT_GUARD_BACKOFF_BASE` | `2` | Base delay (seconds) for exponential backoff |
| `HERMES_RATE_LIMIT_GUARD_BACKOFF_MAX` | `60` | Maximum backoff delay |
| `HERMES_RATE_LIMIT_GUARD_RETRY_ATTEMPTS` | `3` | Retries before switching to fallback |

Example `.env` customization:

```bash
HERMES_RATE_LIMIT_GUARD_CACHE_TTL=7200
HERMES_RATE_LIMIT_GUARD_FALLBACK_MODEL=google/gemma-4-27b-it
HERMES_RATE_LIMIT_GUARD_RETRY_ATTEMPTS=5
```

## How It Works

### Request Flow

```
User Request
    ↓
pre_api_request hook
    ↓
┌─────────────────────────────────────┐
│ 1. Check cache → HIT? Return cached │
│ 2. Rate limited?                    │
│    ├─ Retry < max? → Backoff + wait │
│    ├─ Retry >= max? → Use fallback  │
│    └─ Low priority? → Queue         │
│ 3. No limit → Pass through          │
└─────────────────────────────────────┘
    ↓
API Call (or fallback, or cached)
    ↓
post_api_request hook
    ├─ 429? Mark rate limited, increment retry
    ├─ 2xx? Cache response, clear rate limit
    └─ Log stats every 10 requests
```

### Cache Behavior

- Key: SHA256 hash of `(model, messages)` — deterministic
- TTL: Configurable, default 1 hour
- Auto-prunes expired entries
- Thread-safe (mutex-protected)

### Rate Limit Detection

- Triggered by HTTP 429 status code
- Auto-resets after 5 minutes of no 429s
- Tracks consecutive 429s for adaptive backoff

### Fallback Logic

After `RETRY_ATTEMPTS` (default 3) consecutive 429s in a session:
- Switches to `FALLBACK_MODEL`
- Logs fallback event
- Continues using fallback until rate limit clears

## Queue Processing

Queued requests are stored in memory and can be processed later:

```python
from plugins.rate_limit_guard import process_queue

# In a cron job or background task:
processed = process_queue(
    agent_func=lambda model, messages, **kw: agent.chat(messages),
    max_items=10
)
```

Or set up a cron job:

```bash
hermes cron create \
  --name "Process rate limit queue" \
  --schedule "every 5m" \
  --prompt "Process queued rate-limited requests using process_queue from plugins.rate_limit_guard"
```

## Status & Monitoring

### Check status programmatically

```python
from plugins.rate_limit_guard import get_status

# Overall status
status = get_status()
print(f"Queue: {status['queue_size']}, Cache: {status['cache_size']}, "
      f"Rate limited: {status['rate_limit_active']}")

# Per-session stats
session_status = get_status(session_id="abc123")
print(f"Requests: {session_status['session']['total_requests']}, "
      f"Cache hits: {session_status['session']['cache_hits']}")
```

### Logs to watch

```
INFO:rate_limit_guard: loaded (cache_ttl=3600s, fallback=qwen/qwen3.6-plus)
INFO:rate_limit_guard: FALLBACK to qwen/qwen3.6-plus after 3 retries
WARNING:rate_limit_guard: 429 received (consecutive: 2, session retries: 2)
DEBUG:rate_limit_guard: CACHE HIT for session abc123
```

## Integration with Hermes Core

The plugin uses existing hooks — **no core modifications required**:

- `pre_api_request` — Intercepts before API call, can modify model/args
- `post_api_request` — Observes status code, updates cache/state
- `on_session_start` — Initializes per-session tracking

For the hooks to fire, the core must call `invoke_hook()` at the right points.
The current Hermes Agent already has `pre_api_request` / `post_api_request` hooks
wired in `run_agent.py` around the LLM client call. If your version doesn't,
see `hermes-hook-extension-pattern` skill to add them.

## Priority Routing

Mark a request as high-priority to bypass queuing:

```python
# In your code that calls the agent, add a flag:
response = agent.chat(message, priority=True)
```

The `priority` kwarg flows through to the hook and prevents queueing.

## Limitations

- **Cache is in-memory** — not persistent across restarts
- **Queue is in-memory** — lost on restart (by design, to avoid stale requests)
- **Fallback model must be configured** — set via env var
- **Only works with hooks** — requires `pre_api_request`/`post_api_request` in core

## Troubleshooting

### Plugin not loading

```bash
# Check plugin discovery
python3 -c "from hermes_cli.plugins import discover_plugins; print(discover_plugins())"
```

### Cache not working

- Verify `HERMES_RATE_LIMIT_GUARD_CACHE_TTL > 0`
- Check that prompts are identical (cache key is exact match)

### Fallback not triggering

- Check `HERMES_RATE_LIMIT_GUARD_RETRY_ATTEMPTS` — must be <= actual 429 count
- Verify `HERMES_RATE_LIMIT_GUARD_FALLBACK_MODEL` is valid and has API keys

### Queue growing unbounded

- Set up a cron job to process the queue
- Increase `HERMES_RATE_LIMIT_GUARD_MAX_QUEUE_SIZE` if dropping requests

## Related Skills

- `hermes-plugin-from-existing-hooks` — Building plugins with standard hooks
- `hermes-hook-extension-pattern` — Adding new hooks to core if needed
- `hermes-tool-retry-pattern` — Lower-level retry logic for individual tools
