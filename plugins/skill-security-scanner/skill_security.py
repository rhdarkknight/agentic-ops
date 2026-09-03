"""
Skill security gate.

Lifted and adapted from NVIDIA/SkillSpector (Apache 2.0).
Pattern catalog: ~30 active rule IDs (subset of 64) across categories.
AST taint tracker: source -> sink data-flow analysis.
YARA pass + OSV.dev: Phase 2 (lazily imported).
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import NamedTuple

from .osv_client import (
    ECOSYSTEM_NPM,
    ECOSYSTEM_PYPI,
    parse_requirements_txt,
    query_packages,
)

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Action(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class Location:
    file: str
    start_line: int
    end_line: int | None = None


@dataclass
class Finding:
    rule_id: str
    category: str
    pattern: str
    message: str
    severity: Severity
    confidence: float
    location: Location
    explanation: str
    remediation: str
    code_snippet: str = ""
    matched_text: str = ""


@dataclass
class ScanReport:
    skill_path: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def risk_score(self) -> int:
        weights = {Severity.CRITICAL: 40, Severity.HIGH: 15, Severity.MEDIUM: 5, Severity.LOW: 1}
        score = sum(int(weights[f.severity] * f.confidence) for f in self.findings)
        return min(100, score)

    @property
    def verdict(self) -> Action:
        if any(f.severity in (Severity.CRITICAL, Severity.HIGH) for f in self.findings):
            return Action.BLOCK
        if any(f.severity == Severity.MEDIUM for f in self.findings):
            return Action.WARN
        return Action.ALLOW

    def by_severity(self) -> dict[str, int]:
        out: dict[str, int] = {s.value: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.value] += 1
        return out


_CODE_EXAMPLE_INDICATORS: tuple[str, ...] = (
    "```", "example:", "for example", "e.g.", "such as", "documentation",
    "# warning:", "# note:", "**warning**", "**note**",
    "// good:", "// bad:", "// correct:", "// incorrect:", "// wrong:",
)


def is_code_example(context: str) -> bool:
    """Heuristic: context appears to be a code example or doc snippet."""
    ctx_lower = context.lower()
    return any(ind in ctx_lower for ind in _CODE_EXAMPLE_INDICATORS)


RULE_CATALOG: dict[str, dict] = {
    "P1": {"sev": Severity.HIGH, "cat": "Prompt Injection", "name": "Override Instructions",
           "patterns": [(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", 0.85),
                        (r"disregard\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|prompts?)", 0.85),
                        (r"forget\s+(?:everything|all)\s+(?:above|before|prior)", 0.8)]},
    "P2": {"sev": Severity.HIGH, "cat": "Prompt Injection", "name": "Hidden Instructions",
           "patterns": [(r"<!--.*?(?:ignore|disregard|forget|override).*?-->", 0.9),
                        (r"<style[^>]*display\s*:\s*none[^>]*>.*?</style>", 0.95),
                        (r"[\u200B-\u200F\uFEFF]{3,}", 0.95)]},
    "P3": {"sev": Severity.HIGH, "cat": "Prompt Injection", "name": "Exfiltration Instructions",
           "patterns": [(r"(?:send|transmit|post|upload)\s+(?:the\s+)?(?:user\s+)?(?:data|context|conversation|prompt)\s+to\s+", 0.85)]},
    "P4": {"sev": Severity.MEDIUM, "cat": "Prompt Injection", "name": "Subtle Steering",
           "patterns": [(r"(?:always|never|must)\s+(?:respond|reply|act)\s+(?:as|like)\s+(?:a|an)\s+", 0.6)]},
    "P5": {"sev": Severity.CRITICAL, "cat": "Prompt Injection", "name": "Harmful Content",
           "patterns": [(r"(?:how\s+to|instructions?\s+for)\s+(?:build|make|create|synthesize)\s+(?:a\s+)?(?:bomb|weapon|explosive|nerve\s+agent|toxin)", 0.95)]},
    "E1": {"sev": Severity.MEDIUM, "cat": "Data Exfiltration", "name": "External Transmission",
           "patterns": [(r"requests?\s*\.\s*(?:post|put|patch)\s*\(\s*['\"]https?://", 0.6),
                        (r"httpx\s*\.\s*(?:post|put|patch)\s*\(\s*['\"]https?://", 0.6),
                        (r"urllib\s*\.\s*request\s*\.\s*urlopen\s*\([^)]*data\s*=", 0.6),
                        (r"https?://(?:api|data|collect|telemetry|analytics)\.[\w.-]+/", 0.5),
                        (r"requests?\s*\.\s*(?:post|put|patch)\s*\(\s*[a-zA-Z_]", 0.55),
                        (r"https?\s*\+\s*['\"]//", 0.7)]},
    "E2": {"sev": Severity.HIGH, "cat": "Data Exfiltration", "name": "Env Variable Harvesting",
           "patterns": [(r"for\s+\w+\s*,\s*\w+\s+in\s+os\.environ\.items\(\)", 0.7),
                        (r"os\.environ\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)[^'\"]*['\"]\s*\]", 0.8),
                        (r"os\.environ\.get\s*\([^)]*(?:KEY|SECRET|TOKEN|PASSWORD)", 0.7),
                        (r"process\.env\s*\[\s*['\"][^'\"]*(?:KEY|SECRET|TOKEN|PASSWORD)[^'\"]*['\"]\s*\]", 0.7),
                        (r"Object\.keys\s*\(\s*process\.env\s*\)", 0.6),
                        (r"(?:extract|harvest|gather)\s+(?:api\s+)?keys?\s+from\s+environment", 0.8)]},
    "E3": {"sev": Severity.MEDIUM, "cat": "Data Exfiltration", "name": "File System Enumeration",
           "patterns": [(r"glob\s*\.\s*glob\s*\([^)]*(?:\.env|\.ssh|\.aws|credentials)", 0.8),
                        (r"Path\s*\.\s*home\s*\(\s*\)\s*\.\s*(?:glob|rglob)\s*\(", 0.7),
                        (r"os\s*\.\s*listdir\s*\([^)]*(?:\.ssh|\.aws|\.config|\.gnupg)", 0.8),
                        (r"find\s+[~\$/]\S*\s+.*?-name\s+['\"]?\*(?:\.env|\.pem|\.key|credential)", 0.8),
                        (r"ls\s+-[la]*R?\s+(?:~/|/home/|/Users/)", 0.6),
                        (r"os\s*\.\s*listdir\s*\(\s*os\s*\.\s*path\s*\.\s*expanduser", 0.75),
                        (r"Path\s*\(\s*['\"]~/?['\"]?\s*\)\s*\.\s*(?:iterdir|glob|rglob)", 0.8),
                        (r"shutil\s*\.\s*rmtree\s*\(\s*['\"]/", 0.85)]},
    "E4": {"sev": Severity.HIGH, "cat": "Data Exfiltration", "name": "Context Leakage",
           "patterns": [(r"(?:send|transmit|upload|log|record)\s+(?:the\s+)?(?:full\s+)?(?:conversation|chat|dialog|session)\s+(?:history|context|log)?", 0.85),
                        (r"(?:export|dump|save)\s+(?:the\s+)?(?:conversation|chat|messages?)\s+to\s+(?:external|remote|server)", 0.9),
                        (r"(?:always\s+)?include\s+(?:the\s+)?(?:full\s+)?(?:conversation|context)\s+(?:when|in)\s+(?:calling|making)\s+(?:external|api)", 0.8)]},
    "PE1": {"sev": Severity.LOW, "cat": "Privilege Escalation", "name": "Excessive Permissions",
            "patterns": [(r"chmod\s+777", 0.9), (r"chmod\s+-R\s+777", 0.95)]},
    "PE2": {"sev": Severity.MEDIUM, "cat": "Privilege Escalation", "name": "Sudo/Root Execution",
            "patterns": [(r"\bsudo\s+", 0.5), (r"\bsu\s+-", 0.5)]},
    "PE3": {"sev": Severity.HIGH, "cat": "Privilege Escalation", "name": "Credential File Access",
            "patterns": [(r"cat\s+[~\/].*(?:\.ssh\/(?:id_rsa|id_ed25519|authorized_keys)|\.aws\/credentials|\.netrc)", 0.9),
                         (r"open\s*\([^)]*(?:\.ssh\/(?:id_rsa|id_ed25519)|\.aws\/credentials|\.netrc)", 0.9)]},
    "SC1": {"sev": Severity.LOW, "cat": "Supply Chain", "name": "Unpinned Dependencies",
            "patterns": [(r"(?:requests|httpx|aiohttp)>=?\s*(?![\d])", 0.4)]},
    "SC2": {"sev": Severity.HIGH, "cat": "Supply Chain", "name": "Remote Code Execution",
            "patterns": [(r"curl\s+[^|]*\|\s*(?:bash|sh|zsh)", 0.95),
                         (r"wget\s+[^|]*\|\s*(?:bash|sh|zsh)", 0.95),
                         (r"pip\s+install\s+[^|]*\|\s*(?:bash|sh)", 0.95),
                         (r"npm\s+install\s+[^|]*\|\s*(?:bash|sh)", 0.9),
                         (r"iex\s*\(\s*(?:New-Object\s+)?Net\.WebClient\)", 0.9),
                         (r"Invoke-Expression\s*\(\s*(?:New-Object\s+)?Net\.WebClient", 0.9),
                         (r"\.DownloadString\([^)]+\)\s*\|\s*iex", 0.95),
                         (r"\.DownloadFile\([^)]+\)[^\n]*\|\s*iex", 0.9)]},
    "SC3": {"sev": Severity.HIGH, "cat": "Supply Chain", "name": "Obfuscated Code",
            "patterns": [(r"(?:exec|eval)\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\.decode\s*\(\s*['\"]base64", 0.95),
                         (r"base64\.\s*b64decode\s*\([^)]+\)\s*\)\s*$", 0.85),
                         (r"\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){8,}", 0.9),
                         (r"bytes\.fromhex\s*\(\s*['\"][0-9a-fA-F]{16,}", 0.95),
                         (r"codecs\.decode\s*\([^,]+,\s*['\"]rot", 0.9)]},
    "EA1": {"sev": Severity.HIGH, "cat": "Excessive Agency", "name": "Unrestricted Tool Access",
            "patterns": [(r"enabled_toolsets\s*=\s*\[['\"]?\*", 0.8),
                         (r"all\s+tools?\s+(?:enabled|available|granted)", 0.7)]},
    "EA2": {"sev": Severity.HIGH, "cat": "Excessive Agency", "name": "Autonomous High-Impact Decisions",
            "patterns": [(r"(?:delete|drop|truncate|rm\s+-rf)\s+.*\s+without\s+(?:confirmation|prompt|approval)", 0.85),
                         (r"transfer\s+\$[\d,]+", 0.7)]},
    "EA3": {"sev": Severity.MEDIUM, "cat": "Excessive Agency", "name": "Scope Creep",
            "patterns": [(r"beyond\s+(?:the\s+)?(?:stated|documented|declared)\s+(?:purpose|scope|functionality)", 0.7)]},
    "EA4": {"sev": Severity.MEDIUM, "cat": "Excessive Agency", "name": "Unbounded Resource Access",
            "patterns": [(r"while\s+True\s*:.*?(?:requests?|httpx)", 0.8)]},
    "OH1": {"sev": Severity.HIGH, "cat": "Output Handling", "name": "Unvalidated Output Injection",
            "patterns": [(r"subprocess\.(?:run|call|Popen)\s*\([^)]*shell\s*=\s*True\s*[,)]", 0.7),
                         (r"os\.system\s*\(\s*(?:f['\"]|.*?\+\s*\w)", 0.75)]},
    "OH2": {"sev": Severity.MEDIUM, "cat": "Output Handling", "name": "Cross-Context Output",
            "patterns": [(r"output\s+to\s+(?:different|another|other)\s+(?:context|domain|tenant)", 0.7)]},
    "OH3": {"sev": Severity.MEDIUM, "cat": "Output Handling", "name": "Unbounded Output",
            "patterns": [(r"return\s+['\"].*['\"]\s*\*\s*\d{6,}", 0.85)]},
    "P6": {"sev": Severity.HIGH, "cat": "System Prompt Leakage", "name": "Direct Leakage",
           "patterns": [(r"(?:print|echo|log|reveal|show|leak)\s+(?:the\s+)?(?:system\s+prompt|hidden\s+(?:prompt|instructions?)|internal\s+rules?)", 0.9)]},
    "P7": {"sev": Severity.MEDIUM, "cat": "System Prompt Leakage", "name": "Indirect Extraction",
           "patterns": [(r"(?:translate|rephrase|summarize)\s+(?:the\s+)?(?:system\s+)?prompt", 0.8)]},
    "P8": {"sev": Severity.HIGH, "cat": "System Prompt Leakage", "name": "Tool-Based Exfiltration",
           "patterns": [(r"(?:requests?|httpx|urllib)\.(?:get|post|put|patch)\s*\([^)]*(?:system_prompt|hidden_prompt|internal_rules)", 0.9)]},
    "MP1": {"sev": Severity.HIGH, "cat": "Memory Poisoning", "name": "Persistent Context Injection",
            "patterns": [(r"hindsight_retain\s*\([^)]*(?:system|override|admin|root)", 0.85),
                         (r"memory\.add\s*\([^)]*(?:system|override|admin)", 0.85)]},
    "MP2": {"sev": Severity.MEDIUM, "cat": "Memory Poisoning", "name": "Context Window Stuffing",
            "patterns": [(r"['\"].*['\"]\s*\*\s*\d{4,}", 0.7)]},
    "MP3": {"sev": Severity.HIGH, "cat": "Memory Poisoning", "name": "Memory Manipulation",
            "patterns": [(r"(?:hindsight|memory)\.(?:replace|remove|delete)\s*\([^)]*['\"]", 0.7)]},
    "TM1": {"sev": Severity.HIGH, "cat": "Tool Misuse", "name": "Tool Parameter Abuse",
            "patterns": [(r"--force[_-](?:push|delete|merge)", 0.7)]},
    "TM3": {"sev": Severity.MEDIUM, "cat": "Tool Misuse", "name": "Unsafe Defaults",
            "patterns": [(r"verify\s*=\s*False", 0.7),
                         (r"TLS_VERIFY\s*=\s*0", 0.9),
                         (r"check_hostname\s*=\s*False", 0.7)]},
    "RA1": {"sev": Severity.HIGH, "cat": "Rogue Agent", "name": "Self-Modification",
            "patterns": [(r"open\s*\(\s*__file__\s*,\s*['\"]w", 0.95),
                         (r"(?:Path|pathlib)\s*\(\s*__file__\s*\)\s*\.\s*write_text", 0.95),
                         (r"(?:disable|remove|delete|bypass)\s+(?:the\s+)?(?:safety|security|guard|protection)\s+(?:check|rule|mechanism)", 0.9),
                         (r"self[_-]?(?:modify|update|rewrite|patch|evolve)", 0.9)]},
    "RA2": {"sev": Severity.HIGH, "cat": "Rogue Agent", "name": "Session Persistence",
            "patterns": [(r"cronjob\s*\(\s*action\s*=\s*['\"]create", 0.6),
                         (r"crontab\s+-l\s*>>", 0.7),
                         (r"@reboot\s+", 0.6),
                         (r"systemd\s+enable\s+", 0.6)]},
    "TR1": {"sev": Severity.MEDIUM, "cat": "Trigger Abuse", "name": "Overly Broad Trigger",
            "patterns": [(r"triggers?\s*[:=]\s*\[?\s*['\"](?:help|run|go|start)['\"]\s*\]?", 0.6)]},
}

REMEDIATIONS: dict[str, tuple[str, str]] = {
    "P1": ("Skill instructs agent to override safety rules.", "Strip override language; require explicit user consent for sensitive ops."),
    "P2": ("Hidden directives in comments or invisible chars.", "Audit all comments and zero-width chars; remove any steering instructions."),
    "P3": ("Skill exfiltrates user data via instructions.", "Remove data-exfil directives; use documented, audited channels."),
    "P4": ("Implicit steering of agent decisions.", "Make instructions explicit; tie to documented purpose."),
    "P5": ("Potentially harmful content.", "Block; review with safety team before any install."),
    "E1": ("Code sends data to external URL.", "Verify endpoint; require user opt-in for outbound traffic."),
    "E2": ("Code harvests env vars (likely secrets).", "Reduce surface: read only specific named vars, never enumerate."),
    "E3": ("Code scans filesystem for sensitive paths.", "Restrict to explicit allowlist paths."),
    "E4": ("Skill leaks conversation context externally.", "Strip context-leak patterns; enforce redaction."),
    "PE1": ("Excessive file permissions.", "Use 600/700; never 777."),
    "PE2": ("sudo / root invocation.", "Drop privileges; use capability-based access."),
    "PE3": ("Reads SSH keys / cloud creds.", "Block; secrets must never be read by skills."),
    "SC1": ("Unpinned deps allow supply-chain takeover.", "Pin to exact versions; use hash-locked files."),
    "SC2": ("curl|bash / wget|sh / iiex — remote code exec.", "Replace with pre-audited packages."),
    "SC3": ("Base64 / hex obfuscation around exec.", "Decode and review; reject if payload is opaque."),
    "EA1": ("Unrestricted tool access.", "Whitelist required toolsets; deny by default."),
    "EA2": ("Autonomous destructive actions.", "Require HITL confirmation for delete/transfer/etc."),
    "EA3": ("Scope creep beyond stated purpose.", "Trim to declared functionality."),
    "EA4": ("Unbounded loops / resource use.", "Add rate limits and timeout guards."),
    "OH1": ("Unvalidated LLM output into shell/SQL/HTML.", "Sanitize and validate; never pass output to exec sinks directly."),
    "OH2": ("Output crosses context boundaries.", "Enforce per-tenant redaction."),
    "OH3": ("Unbounded output size.", "Cap response length; truncate gracefully."),
    "P6": ("Directly exposes system prompt.", "Remove all references to hidden/internal prompts."),
    "P7": ("Indirect prompt extraction.", "Block paraphrase/summarize-prompt patterns."),
    "P8": ("Exfiltrates prompt via tool calls.", "Strip tool calls that reference system_prompt / hidden_prompt."),
    "MP1": ("Persists directives into agent memory.", "No skill may write to memory except via audited channel."),
    "MP2": ("Context window stuffing.", "Cap any single write; rate-limit memory writes."),
    "MP3": ("Modifies / deletes memory entries.", "Skills must never call memory.replace/remove."),
    "TM1": ("Abuses tool parameters (shell=True, --force).", "Reject shell=True; use argv lists; require --no-force."),
    "TM3": ("Unsafe defaults (verify=False, etc.).", "Force TLS verify; fail closed."),
    "RA1": ("Skill modifies own code/config at runtime.", "Block; skills must be immutable post-install."),
    "RA2": ("Establishes cross-session persistence (cron, systemd, @reboot).", "Block; skills must not install persistent mechanisms."),
    "TR1": ("Overly broad trigger shadows common commands.", "Tighten trigger specificity."),
    "TT1": ("Data flows from source to sink without validation.", "Insert validation between source and sink."),
    "TT3": ("Credentials flow to a network sink — high-confidence exfil.", "Block; remove code path."),
    "TT4": ("File contents flow to a network sink.", "Remove or require explicit user consent."),
    "TT5": ("External input flows to a code execution sink.", "Block; never eval/network-to-exec."),
}


def _get_line_number(content: str, offset: int) -> int:
    return content[:offset].count("\n") + 1


def _get_context(content: str, match_start: int, context_lines: int = 3) -> str:
    lines = content.splitlines()
    match_line = content[:match_start].count("\n")
    start = max(0, match_line - context_lines)
    end = min(len(lines), match_line + context_lines + 1)
    return "\n".join(lines[start:end])


def _scan_content(content: str, file_path: str, file_type: str) -> list[Finding]:
    findings: list[Finding] = []
    for rule_id, spec in RULE_CATALOG.items():
        for pattern, base_conf in spec["patterns"]:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                start = match.start()
                ctx_text = _get_context(content, start)
                if is_code_example(ctx_text):
                    continue
                conf = base_conf
                if file_type in ("py", "js", "sh", "bash", "shell") and conf < 1.0:
                    conf = min(1.0, conf + 0.1)
                line_num = _get_line_number(content, start)
                matched = match.group(0)[:200]
                expl, rem = REMEDIATIONS.get(rule_id, ("", ""))
                findings.append(Finding(
                    rule_id=rule_id,
                    category=spec["cat"],
                    pattern=spec["name"],
                    message=f"{spec['name']} pattern detected",
                    severity=spec["sev"],
                    confidence=conf,
                    location=Location(file=file_path, start_line=line_num),
                    explanation=expl,
                    remediation=rem,
                    code_snippet=ctx_text[:200],
                    matched_text=matched,
                ))
    return findings


class _TaintedVar(NamedTuple):
    name: str
    source_call: str
    lineno: int


def _scan_python_taint(content: str, file_path: str) -> list[Finding]:
    """AST source->sink data-flow tracker (Phase 1 core)."""
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return []

    lines = content.splitlines()
    findings: list[Finding] = []
    tainted: dict[str, _TaintedVar] = {}
    seen: set[tuple[str, int, str, str]] = set()

    def _resolve(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts: list[str] = [node.attr]
            cur: ast.AST = node.value
            # Descend through Attribute and Call chains (e.g. Path(x).read_text,
            # urllib.request.urlopen(x).read). For Call, only the inner call's
            # resolved name is used — outer attr (e.g. .read) is dropped because
            # taint source is the network call, not the post-process method.
            while isinstance(cur, (ast.Attribute, ast.Call)):
                if isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                else:  # ast.Call
                    inner = cur.func
                    if isinstance(inner, ast.Name):
                        return inner.id
                    if isinstance(inner, ast.Attribute):
                        inner_resolved = _resolve(inner)
                        if inner_resolved:
                            return inner_resolved
                    return None
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                return ".".join(reversed(parts))
        return None

    CRED_SRCS = {"os.environ", "os.environ.get", "os.getenv"}
    FILE_SRCS = {"open", "Path", "Path.read_text", "Path.read_bytes"}
    NET_SRCS = {"requests.get", "requests.post", "requests.put",
                "httpx.get", "httpx.post", "httpx.put",
                "urllib.request.urlopen"}
    NET_SINKS = {"requests.post", "requests.put", "requests.patch",
                 "httpx.post", "httpx.put", "httpx.patch",
                 "urllib.request.urlopen"}
    EXEC_SINKS = {"exec", "eval", "os.system", "os.popen",
                  "subprocess.run", "subprocess.call", "subprocess.Popen"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            src = None
            if isinstance(value, ast.Call):
                src = _resolve(value.func)
                if src not in CRED_SRCS and src not in FILE_SRCS and src not in NET_SRCS:
                    src = None
            elif isinstance(value, ast.Subscript):
                base = _resolve(value.value)
                if base in CRED_SRCS:
                    src = base
            if src:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        tainted[t.id] = _TaintedVar(t.id, src, node.lineno)
            continue

        if not isinstance(node, ast.Call):
            continue
        sink = _resolve(node.func)
        if sink not in NET_SINKS and sink not in EXEC_SINKS:
            continue
        line = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", None)
        for child in ast.walk(node):
            if child is node:
                continue
            var = None
            if isinstance(child, ast.Name):
                var = child.id
            elif isinstance(child, ast.Subscript):
                var = _resolve(child.value)
            if not var or var not in tainted:
                continue
            tv = tainted[var]
            key = (tv.source_call, line, sink, var)
            if key in seen:
                continue
            seen.add(key)
            if tv.source_call in CRED_SRCS and sink in NET_SINKS:
                rule, sev, conf, name = "TT3", Severity.CRITICAL, 0.9, "Credential Exfil Flow"
            elif tv.source_call in FILE_SRCS and sink in NET_SINKS:
                rule, sev, conf, name = "TT4", Severity.HIGH, 0.8, "File Data Exfil Flow"
            elif tv.source_call in NET_SRCS and sink in EXEC_SINKS:
                rule, sev, conf, name = "TT5", Severity.CRITICAL, 0.9, "External Input to Exec"
            elif tv.source_call in (CRED_SRCS | FILE_SRCS) and sink in EXEC_SINKS:
                rule, sev, conf, name = "TT1", Severity.HIGH, 0.8, "Source-to-Exec Flow"
            else:
                rule, sev, conf, name = "TT1", Severity.HIGH, 0.8, "Direct Taint Flow"
            findings.append(Finding(
                rule_id=rule, category="Data Flow", pattern=name,
                message=f"'{tv.name}' (line {tv.lineno}, {tv.source_call}) -> {sink}",
                severity=sev, confidence=conf,
                location=Location(file=file_path, start_line=line, end_line=end),
                explanation=REMEDIATIONS.get(rule, ("", ""))[0],
                remediation=REMEDIATIONS.get(rule, ("", ""))[1],
                code_snippet="\n".join(lines[max(0, line - 2):line + 2])[:200],
                matched_text=tv.name,
            ))
            break
    return findings


_SKILL_EXTS = {".md", ".py", ".sh", ".bash", ".js", ".ts", ".yaml", ".yml", ".json", ".toml"}
_MAX_BYTES = 1_000_000

# For pyproject.toml parsing — duplicated from osv_client because the
# pyproject handler runs on stripped lines, not the full requirements format.
_PYPROJECT_PKG_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*([<>=!~]=?\s*[^\s,#;]+)?\s*[,]?\s*$")


def _iter_files(skill_path: Path) -> list[Path]:
    if skill_path.is_file():
        return [skill_path]
    files: list[Path] = []
    for p in skill_path.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SKILL_EXTS:
            continue
        if any(part.startswith(".") and part not in {".env", ".gitignore"} for part in p.parts):
            continue
        try:
            if p.stat().st_size > _MAX_BYTES:
                continue
        except OSError:
            continue
        files.append(p)
    return files


def _scan_dependencies(p: Path) -> list[Finding]:
    """If a requirements.txt or pyproject.toml is present, query OSV.dev."""
    findings: list[Finding] = []
    req_files: list[tuple[Path, str, str]] = []  # (path, content, ecosystem)
    for name, eco in [("requirements.txt", ECOSYSTEM_PYPI),
                      ("requirements-dev.txt", ECOSYSTEM_PYPI),
                      ("pyproject.toml", ECOSYSTEM_PYPI),
                      ("package.json", ECOSYSTEM_NPM)]:
        for candidate in p.rglob(name) if p.is_dir() else [p] if p.name == name else []:
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            req_files.append((candidate, content, eco))
    if not req_files:
        return findings
    for path, content, eco in req_files:
        if path.name.endswith(".txt"):
            pkgs = parse_requirements_txt(content)
        elif path.name == "pyproject.toml":
            # Minimal: pull out [project.dependencies] lines (best-effort)
            pkgs = []
            in_deps = False
            for line in content.splitlines():
                if line.strip().startswith("[") and line.strip() != "[project.dependencies]":
                    in_deps = False
                if line.strip() == "[project.dependencies]":
                    in_deps = True
                    continue
                if in_deps and line.strip().startswith('"'):
                    m = _PYPROJECT_PKG_RE.match(line.strip().strip('",'))
                    if m:
                        pkgs.append((m.group(1),
                                     re.sub(r"^[<>=!~]+\s*", "", m.group(2) or "").strip() or None))
        else:  # package.json
            pkgs = []
            for m in re.finditer(r'"([A-Za-z0-9_.\-@/]+)":\s*"\s*([~^]?[0-9][^"]*)"', content):
                if not m.group(1).startswith("@"):
                    pkgs.append((m.group(1), m.group(2).lstrip("~^") or None))
        if not pkgs:
            continue
        try:
            results = query_packages(pkgs, ecosystem=eco)
        except Exception as e:
            logger.warning("OSV scan failed: %s", e)
            continue
        for (name, version), vulns in zip(pkgs, results):
            for v in vulns:
                sev = Severity(v.severity) if v.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL") else Severity.HIGH
                findings.append(Finding(
                    rule_id="SC4",
                    category="Supply Chain",
                    pattern="Known Vulnerable Dependency",
                    message=f"{name} {version or ''}: {v.vuln_id} ({v.severity})",
                    severity=sev,
                    confidence=0.95,
                    location=Location(file=str(path), start_line=1),
                    explanation=f"OSV.dev reports {v.vuln_id} affecting {name}. {v.summary[:150]}",
                    remediation=f"Upgrade {name} to a non-vulnerable version.",
                    code_snippet=v.vuln_id,
                    matched_text=", ".join(v.aliases[:3]),
                ))
    return findings


def scan_skill(skill_path: str | Path) -> ScanReport:
    """Scan a skill directory or file. Returns ScanReport with findings + verdict.

    Runs three passes:
      1. Regex-based static patterns (P1..TR1)
      2. AST taint tracker for Python files (TT1..TT5)
      3. YARA malware/hacktool signatures (YR1..YR4, optional)
      4. OSV.dev CVE lookup for declared dependencies (SC4, optional)
    """
    p = Path(skill_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Skill path does not exist: {p}")
    report = ScanReport(skill_path=str(p))
    for file_p in _iter_files(p):
        try:
            content = file_p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("Cannot read %s: %s", file_p, e)
            continue
        ftype = file_p.suffix.lstrip(".").lower()
        report.findings.extend(_scan_content(content, str(file_p), ftype))
        if ftype == "py":
            report.findings.extend(_scan_python_taint(content, str(file_p)))
    # Phase 2 passes
    try:
        from .yara_pass import scan_yara
        report.findings.extend(scan_yara(p))
    except Exception as e:
        logger.debug("YARA pass skipped: %s", e)
    try:
        report.findings.extend(_scan_dependencies(p))
    except Exception as e:
        logger.debug("OSV pass skipped: %s", e)
    return report


def format_report(report: ScanReport) -> str:
    """Human-readable summary for Telegram / CLI."""
    sev_counts = report.by_severity()
    lines = [
        f"Skill: {report.skill_path}",
        f"Verdict: {report.verdict.value.upper()}",
        f"Risk score: {report.risk_score}/100",
        f"Findings: {len(report.findings)} "
        f"(CRIT={sev_counts['CRITICAL']} HIGH={sev_counts['HIGH']} "
        f"MED={sev_counts['MEDIUM']} LOW={sev_counts['LOW']})",
    ]
    if report.findings:
        lines.append("")
        ordered = sorted(report.findings, key=lambda f: (
            {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}[f.severity.value], -f.confidence))
        for f in ordered[:5]:
            lines.append(
                f"  [{f.severity.value}] {f.rule_id} {f.pattern} "
                f"@ {Path(f.location.file).name}:{f.location.start_line} "
                f"(conf={f.confidence:.2f})"
            )
    return "\n".join(lines)
