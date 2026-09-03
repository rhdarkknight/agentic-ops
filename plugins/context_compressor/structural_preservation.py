"""
Structural Preservation Plugin
==============================

Detects and tags structured content in messages to prevent fragmentation
during compression summarization. Protects:

- Code blocks (Python, JS, Rust, Go, etc.)
- JSON/YAML/TOML configs
- Markdown with headings
- XML/HTML markup

Tagged structures are passed to the compressor which can preserve them
intact during truncation.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

STRUCTURE_PATTERNS = {
    'code_python': (r'```python\n(.*?)```', 'code'),
    'code_generic': (r'```\w*\n(.*?)```', 'code'),
    'json': (r'^\s*[\[{].*?[\]}]\s*$', 'config_json'),
    'yaml': (r'^\w[\w\s]*:\s*.*$', 'config_yaml'),
    'markdown_headings': (r'^#{1,6}\s+.+$', 'markdown'),
}


class StructurePreserver:
    """Detects and tags structured content for preservation."""

    def pre_compress(self, ctx, session_id, messages, compression_count, approx_tokens, **kwargs):
        """Tag structured content in all messages.

        DEPRECATED (2026-08-19): the pre_compress hook does not exist in
        VALID_HOOKS. Kept for backward compatibility; the plugin now uses
        build_preservation_note() via the pre_llm_call hook.
        """
        structure_protections = {}

        for i, msg in enumerate(messages):
            content = msg.get("content", "") or ""
            protections = self._detect_structures(content)
            if protections:
                structure_protections[i] = protections

        return {"structure_protections": structure_protections}

    def build_preservation_note(self, messages) -> str:
        """Return a prompt block telling the model to preserve structured
        content (code/config blocks) during compression summarization.

        Called from the valid pre_llm_call hook. Returns "" when no
        structured content is detected.
        """
        if not messages:
            return ""
        kinds: set[str] = set()
        for msg in messages:
            content = msg.get("content", "") or ""
            if not isinstance(content, str):
                continue
            protections = self._detect_structures(content)
            for _, _, stype in protections:
                kinds.add(stype)
        if not kinds:
            return ""
        kind_str = ", ".join(sorted(kinds))
        return (
            "[context_compressor_enhancements: structured content detected — "
            f"preserve these blocks exactly during any compression: {kind_str}. "
            "Keep code blocks, configs, and markup intact; do not paraphrase them.]"
        )

    def _detect_structures(self, message_content: str) -> List[Tuple[int, int, str]]:
        """Find structured ranges in message content."""
        protections = []

        for name, (pattern, struct_type) in STRUCTURE_PATTERNS.items():
            for match in re.finditer(pattern, message_content, re.MULTILINE | re.DOTALL):
                protections.append((match.start(), match.end(), struct_type))

        # Merge overlapping ranges
        protections.sort(key=lambda x: x[0])
        merged = []
        for start, end, stype in protections:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
            else:
                merged.append((start, end, stype))

        return merged
