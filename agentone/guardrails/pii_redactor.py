"""Personally Identifiable Information (PII) redaction and token restoration."""

import re
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field


class RedactionResult(BaseModel):
    """Result of PII masking operation with bidirectional entity map."""
    original_text: str
    redacted_text: str
    pii_found: bool = False
    entity_counts: Dict[str, int] = Field(default_factory=dict)
    token_map: Dict[str, str] = Field(default_factory=dict)  # [TOKEN_ID] -> original_value


class PIIRedactor:
    """Detects and masks sensitive personal data before ingestion into LLM context."""

    PATTERNS: List[Tuple[str, str]] = [
        ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b"),
        ("SSN", r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b"),
        ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        ("PHONE", r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ("API_KEY", r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36}|AIza[0-9A-Za-z-_]{35})\b"),
    ]

    def redact(self, text: str) -> RedactionResult:
        """Scan text, replace sensitive entities with placeholders, and maintain token map."""
        if not text:
            return RedactionResult(original_text="", redacted_text="", pii_found=False)

        redacted = text
        token_map: Dict[str, str] = {}
        entity_counts: Dict[str, int] = {}
        token_counters: Dict[str, int] = {}

        for entity_type, regex in self.PATTERNS:
            matches = list(re.finditer(regex, redacted))
            if not matches:
                continue

            entity_counts[entity_type] = 0
            # Reverse order to preserve match spans
            for match in reversed(matches):
                val = match.group(0).strip()
                # Basic sanity check to avoid masking pure short numbers
                if entity_type in ["CREDIT_CARD", "SSN"] and len(re.sub(r"\D", "", val)) < 9:
                    continue

                token_counters[entity_type] = token_counters.get(entity_type, 0) + 1
                token_id = f"[{entity_type}_{token_counters[entity_type]}]"
                
                token_map[token_id] = val
                entity_counts[entity_type] += 1
                
                start, end = match.span()
                redacted = redacted[:start] + token_id + redacted[end:]

        pii_found = bool(token_map)
        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            pii_found=pii_found,
            entity_counts=entity_counts,
            token_map=token_map,
        )

    def restore(self, redacted_text: str, token_map: Dict[str, str]) -> str:
        """Reconstruct original values from token map into generated text."""
        if not redacted_text or not token_map:
            return redacted_text

        restored = redacted_text
        for token_id, original_val in token_map.items():
            restored = restored.replace(token_id, original_val)
        return restored
