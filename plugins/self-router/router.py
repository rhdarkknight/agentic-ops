"""Router: the self-router decision + execution layer.

The unifying capability: decide whether Hermes should execute a task itself or
dispatch it to a specialist/stateless harness it can drive. The decision is
pure logic (testable). The dispatch is gated on the VERIFIED executor set
(`self_router.executors`) — an empty set means the router recommends but never
auto-dispatches (default, trust-preserving).

Every non-trivial task flows through ``maybe_route``. The trigger (pre_tool_call)
calls this with the task description + closest recall match.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from . import self_assess, anchoring
from .config import load_config

logger = logging.getLogger(__name__)


class Router:
    """The self-router decision engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        self._last_route_check: float = 0.0
        self._store = anchoring.AnchoringStore()

    # -- config passthroughs -------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    @property
    def executors(self) -> list:
        return list(self.config.get("executors", []) or [])

    @property
    def auto_dispatch(self) -> bool:
        return bool(self.config.get("auto_dispatch", False))

    @property
    def cooldown_sec(self) -> int:
        return int(self.config.get("trigger", {}).get("cooldown_sec", 30))

    @property
    def require_user_confirmation(self) -> bool:
        return bool(
            self.config.get("trigger", {}).get("require_user_confirmation", True)
        )

    # -- the decision --------------------------------------------------------
    def maybe_route(
        self,
        task: str,
        closest_match: float = 0.0,
        confidence: float = 0.5,
        task_signature: str = "",
        description: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Decide what to do with a task. Returns a routing-decision dict.

        Decision values:
          - "keep_self"      -> Hermes executes normally.
          - "recommend"      -> recommend dispatch to a specialist/stateless
                                harness (require_user_confirmation). Hermes
                                waits for user confirmation.
          - "dispatch"       -> dispatch to a specialist now (auto_dispatch +
                                verified executor). Only when both are true.
        """
        if not self.enabled:
            return {"decision": "keep_self", "reason": "self-router disabled", "task": task}

        a = self_assess.assess(task)
        novelty = anchoring.detect_novelty(
            closest_match=closest_match,
            novelty_threshold=float(self.config.get("novelty_threshold", 0.6)),
            task_signature=task_signature or task,
            description=description or task,
            store=self._store,
        )
        anchoring_risk = anchoring.compute_anchoring_risk(closest_match, confidence)

        # Anchor-aware: strong memory match BUT novel -> distrust the match,
        # route to a stateless harness (no anchor).
        if (
            novelty.get("no_prior_art")
            and anchoring_risk > float(self.config.get("anchoring_risk_threshold", 0.7))
        ):
            return self._route_novel(task, a, novelty, anchoring_risk, session_id)

        # Specialist wins narrowly?
        route, specialist_name, _ = self_assess.should_route_to_specialist(
            task, win_margin=float(self.config.get("win_margin", 0.15))
        )
        if route:
            return self._route_specialist(task, specialist_name, a, novelty, anchoring_risk, session_id)

        # Hermes wins comfortably.
        return {
            "decision": "keep_self",
            "reason": "Hermes wins (or specialist margin not met)",
            "task": task,
            "task_type": a["task_type"],
            "keep_self": a["keep_self"],
            "best_specialist": a["best_specialist"],
            "best_specialist_score": a["best_specialist_score"],
            "novelty": novelty,
            "anchoring_risk": anchoring_risk,
            "session_id": session_id,
        }

    def _route_novel(self, task, a, novelty, anchoring_risk, session_id="") -> Dict[str, Any]:
        harness = str(self.config.get("route_novel_to", "codex"))
        return self._dispatch_or_recommend(
            task=task,
            harness=harness,
            reason=(
                "novel task with anchoring risk (no prior art but strong "
                f"memory match); routing to stateless harness '{harness}'"
            ),
            a=a,
            novelty=novelty,
            anchoring_risk=anchoring_risk,
            session_id=session_id,
        )

    def _route_specialist(self, task, specialist_name, a, novelty, anchoring_risk, session_id="") -> Dict[str, Any]:
        return self._dispatch_or_recommend(
            task=task,
            harness=specialist_name,
            reason=(
                f"specialist '{specialist_name}' beats Hermes by the win margin "
                f"(score {a['specialist_fit'].get(specialist_name)} vs "
                f"keep_self {a['keep_self']})"
            ),
            a=a,
            novelty=novelty,
            anchoring_risk=anchoring_risk,
            session_id=session_id,
        )

    def _dispatch_or_recommend(
        self, task, harness, reason, a, novelty, anchoring_risk, session_id=""
    ) -> Dict[str, Any]:
        verified = harness in self.executors
        if self.auto_dispatch and verified:
            # Model cascade: route the executor's model from the ACTIVE session
            # so personal/professional inference costs never cross. Best-effort
            # (a cascade failure must not block the dispatch).
            cascade_result = None
            if self.config.get("model_cascade", False):
                try:
                    from . import cascade as _cascade
                    from . import get_active_session_model
                    active_model = get_active_session_model(session_id)
                    cascade_result = _cascade.cascade(active_model=active_model)
                except Exception as exc:
                    logger.warning("self-router cascade failed: %s", exc)
            return {
                "decision": "dispatch",
                "reason": reason,
                "harness": harness,
                "task": task,
                "task_type": a["task_type"],
                "verified_executor": True,
                "cascade": cascade_result,
                "novelty": novelty,
                "anchoring_risk": anchoring_risk,
            }
        # Not verified OR not auto -> recommend (never silently dispatch).
        return {
            "decision": "recommend",
            "reason": (
                f"RECOMMEND dispatch to '{harness}': {reason} "
                + (
                    ""
                    if verified
                    else f"[harness '{harness}' NOT in verified executors "
                    f"{self.executors}; requires Phase-0 probe + config]"
                )
            ),
            "harness": harness,
            "task": task,
            "task_type": a["task_type"],
            "novelty": novelty,
            "anchoring_risk": anchoring_risk,
            "auto_dispatch": self.auto_dispatch,
            "verified_executor": verified,
        }

    # -- the trigger helper --------------------------------------------------
    def should_check(
        self,
        tool_name: str,
    ) -> bool:
        """Whether the pre_tool_call hook should run the router for a tool.

        Only for the FIRST mutating call of a task, outside the cooldown, and
        only if dispatch is substantively possible. Read-only tools never
        trigger routing (the cost of an assessment on every read is prohibitive).
        """
        from .config import MUTATING_TOOLS, READ_ONLY_TOOLS

        if not self.enabled:
            return False
        if tool_name in READ_ONLY_TOOLS:
            return False
        if tool_name not in MUTATING_TOOLS:
            return False
        now = time.time()
        if now - self._last_route_check < self.cooldown_sec:
            return False
        self._last_route_check = now
        return True


# Module-level singleton for the hook (per-session state lives in Router;
# the singleton is stateless-thin over config).
_router: Optional[Router] = None


def get_router(config: Optional[Dict[str, Any]] = None) -> Router:
    global _router
    if _router is None or config is not None:
        _router = Router(config)
    return _router