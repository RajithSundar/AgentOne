"""Prompt injection, jailbreak defense, and adversarial input shield."""

import re
from typing import List, Tuple
from pydantic import BaseModel, Field


class InjectionDetectionResult(BaseModel):
    """Evaluation output from input injection shield."""
    is_safe: bool = True
    risk_score: float = 0.0  # 0.0 (benign) to 1.0 (malicious)
    flags: List[str] = Field(default_factory=list)
    sanitized_input: str
    remediation_applied: bool = False


class InputShield:
    """Multi-layer heuristic and pattern-based prompt injection defense engine."""

    # High-confidence adversarial injection patterns
    INJECTION_PATTERNS: List[Tuple[str, float, str]] = [
        (r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b", 0.95, "INSTRUCTION_OVERRIDE"),
        (r"(?i)\bdisregard\s+(all\s+)?(system\s+)?prompts?\b", 0.90, "SYSTEM_PROMPT_DISREGARD"),
        (r"(?i)\byou\s+are\s+now\s+(in\s+)?developer\s+mode\b", 0.90, "DEVELOPER_MODE_JAILBREAK"),
        (r"(?i)\byou\s+are\s+dan\b", 0.95, "DAN_JAILBREAK"),
        (r"(?i)\breveal\s+(your\s+)?(system\s+)?(instructions?|prompt|internal\s+rules)\b", 0.85, "SYSTEM_PROMPT_EXTRACTION"),
        (r"(?i)\bact\s+as\s+an\s+unrestricted\s+ai\b", 0.90, "UNRESTRICTED_ROLEPLAY"),
        (r"(?i)```(?:system|admin|root)[\s\S]*?```", 0.80, "ROLE_SPOOFING_DELIMITER"),
        (r"(?i)\bprint\s+(all\s+)?system\s+context\b", 0.85, "CONTEXT_LEAKAGE"),
        (r"(?i)\bexec\s*\(\s*base64\.b64decode\b", 0.95, "CODE_INJECTION_BASE64"),
    ]

    # Suspicious prompt tampering markers
    SUSPICIOUS_MARKERS: List[Tuple[str, float, str]] = [
        (r"\[SYSTEM\]", 0.50, "SYSTEM_TAG_INJECTION"),
        (r"\[ASSISTANT\]", 0.50, "ASSISTANT_TAG_INJECTION"),
        (r"(?i)new\s+role:\s*always\s+say\s+yes", 0.75, "ROLE_MANIPULATION"),
        (r"(?i)from\s+now\s+on\s+you\s+must", 0.40, "INSTRUCTION_TAMPERING"),
    ]

    def __init__(self, risk_threshold: float = 0.65):
        self.risk_threshold = risk_threshold

    def inspect(self, user_input: str) -> InjectionDetectionResult:
        """Analyze prompt for adversarial instructions, calculate risk score, and sanitize."""
        if not user_input or not user_input.strip():
            return InjectionDetectionResult(is_safe=True, risk_score=0.0, sanitized_input="")

        flags: List[str] = []
        max_risk: float = 0.0

        # Check primary injection patterns
        for pattern, weight, flag in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input):
                flags.append(flag)
                max_risk = max(max_risk, weight)

        # Check secondary suspicious markers
        for pattern, weight, flag in self.SUSPICIOUS_MARKERS:
            if re.search(pattern, user_input):
                flags.append(flag)
                max_risk = min(1.0, max_risk + weight * 0.5)

        # Determine safety verdict
        is_safe = max_risk < self.risk_threshold
        sanitized = user_input

        if not is_safe:
            # Strip offending overrides for safer processing
            for pattern, _, _ in self.INJECTION_PATTERNS:
                sanitized = re.sub(pattern, "[REDACTED_ADVERSARIAL_INSTRUCTION]", sanitized)

        return InjectionDetectionResult(
            is_safe=is_safe,
            risk_score=round(max_risk, 4),
            flags=flags,
            sanitized_input=sanitized.strip(),
            remediation_applied=not is_safe,
        )
