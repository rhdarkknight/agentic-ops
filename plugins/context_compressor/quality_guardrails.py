"""
Quality Guardrails Plugin
=========================

Verifies compression summaries for completeness. Checks that summaries
preserve critical information:

- File paths mentioned in the conversation
- Error messages and tracebacks
- User corrections and explicit requests
- Tool names and key parameters

If gaps are found, can request regeneration with gap-filling prompts.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Patterns to check for in summaries
CRITICAL_PATTERNS = [
    (r'[/\w.-]+\.py', 'Python file paths'),
    (r'[/\w.-]+\.yaml', 'YAML config paths'),
    (r'[/\w.-]+\.json', 'JSON config paths'),
    (r'Traceback \(most recent call last\)', 'Error tracebacks'),
    (r'Error:', 'Error messages'),
    (r'\b(wrong|incorrect|no\\s+not|actually|undo|revert)\b', 'User corrections'),
    (r'\b(should|must|need to|have to)\b', 'User requirements'),
]


class QualityGuardrails:
    """Verifies summary quality and requests regeneration if gaps found."""

    def post_compress(self, ctx, session_id, original_messages, compressed_messages, summary_text, compression_count):
        """Verify summary completeness and optionally request regeneration."""
        if not summary_text:
            return {"verified": True}

        result = self._verify_summary(original_messages, summary_text)

        if not result["verified"]:
            logger.warning("Summary quality check found gaps: %s", result["gaps"])
            # Request regeneration (but only every other time to avoid loops)
            if compression_count % 2 == 0:
                result["regenerate"] = True
                logger.info("Requested summary regeneration to fill gaps")

        return result

    def _verify_summary(self, original_messages: List[Dict], summary_text: str) -> Dict:
        """Verify summary completeness. Returns gaps found."""
        gaps = []

        # Concatenate original message content
        original_text = ""
        for msg in original_messages:
            content = msg.get("content", "") or ""
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    content += " " + (tc.get("function", {}).get("arguments", "") or "")
            original_text += content + " "

        # Check each pattern
        for pattern, description in CRITICAL_PATTERNS:
            matches_in_original = set(re.findall(pattern, original_text, re.IGNORECASE))
            matches_in_summary = set(re.findall(pattern, summary_text, re.IGNORECASE))

            missing = matches_in_original - matches_in_summary
            if missing:
                gaps.append(f"Missing {description}: {', '.join(list(missing)[:3])}")

        return {"gaps": gaps[:6], "verified": len(gaps) == 0}
