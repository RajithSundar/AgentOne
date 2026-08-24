"""Output verification, hallucination detection, and schema compliance checking."""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from agentone.core.state import DocumentChunk


class VerificationResult(BaseModel):
    """Output validation and compliance verdict."""
    is_valid: bool = True
    faithfulness_score: float = 1.0  # 0.0 (complete hallucination) to 1.0 (fully grounded)
    hallucination_detected: bool = False
    policy_compliant: bool = True
    violations: List[str] = Field(default_factory=list)
    sanitized_output: str


class OutputVerifier:
    """Validates model responses against retrieved ground truth and governance policies."""

    BANNED_PHRASES: List[str] = [
        "as an ai language model",
        "i don't have access to real-time information",
        "i cannot fulfill this request because i am an ai",
    ]

    def verify(
        self,
        generated_output: str,
        retrieved_documents: Optional[List[DocumentChunk]] = None,
        min_faithfulness_threshold: float = 0.50,
    ) -> VerificationResult:
        """Verify grounding against retrieved context and check policy rules."""
        if not generated_output or not generated_output.strip():
            return VerificationResult(
                is_valid=False,
                faithfulness_score=0.0,
                violations=["Empty response generated."],
                sanitized_output="",
            )

        violations: List[str] = []
        gen_lower = generated_output.lower()

        # Check banned generic AI filler phrases
        for phrase in self.BANNED_PHRASES:
            if phrase in gen_lower:
                violations.append(f"Contains generic AI boilerplate: '{phrase}'")

        # Groundedness / Faithfulness evaluation vs retrieved context
        faithfulness = 1.0
        hallucination = False

        if retrieved_documents:
            context_text = " ".join(doc.content.lower() for doc in retrieved_documents)
            # Extract key nouns and numerical facts from generated text
            gen_words = set(re.findall(r"\b[a-z0-9\$\%]{4,}\b", gen_lower))
            stop_words = {"this", "that", "with", "from", "have", "been", "under", "about", "which", "there", "their"}
            content_words = gen_words - stop_words

            if content_words:
                found_in_context = sum(1 for word in content_words if word in context_text)
                grounded_ratio = found_in_context / float(len(content_words))
                faithfulness = round(min(1.0, grounded_ratio * 1.35), 4)

                if faithfulness < min_faithfulness_threshold:
                    hallucination = True
                    violations.append(
                        f"Low groundedness score ({faithfulness:.2f} < {min_faithfulness_threshold:.2f}). "
                        "Response contains claims not directly supported by retrieved runbooks."
                    )

        is_valid = len(violations) == 0 or (not hallucination and faithfulness >= min_faithfulness_threshold)

        return VerificationResult(
            is_valid=is_valid,
            faithfulness_score=faithfulness,
            hallucination_detected=hallucination,
            policy_compliant=len(violations) == 0,
            violations=violations,
            sanitized_output=generated_output.strip(),
        )
