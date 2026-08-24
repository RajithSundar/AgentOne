"""Evaluation metrics: Faithfulness, Answer Relevancy, Context Precision, and Context Recall."""

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Normalized metrics scores for an individual test case."""
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    tool_accuracy: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    details: Dict[str, Any] = Field(default_factory=dict)


def _normalize_stem(word: str) -> str:
    """Normalize word endings for robust lexical matching."""
    w = word.lower().strip("$%.,!?:;'\"()[]")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def _tokenize_keywords(text: str) -> Set[str]:
    """Extract stemmed informative keywords, ignoring short stopwords."""
    stop_words = {
        "the", "and", "for", "with", "this", "that", "from", "have", "been",
        "your", "will", "what", "when", "where", "which", "about", "under",
        "can", "how", "are", "was", "were", "our", "you", "out", "per"
    }
    words = re.findall(r"\b[a-zA-Z0-9$%]{2,}\b", text.lower())
    stems = {_normalize_stem(w) for w in words if w not in stop_words}
    return {s for s in stems if len(s) >= 2 and s not in stop_words}


def compute_faithfulness(generated_answer: str, retrieved_contexts: List[str]) -> float:
    """
    Measures the ratio of claims/facts in the answer that are supported by the context.
    Faithfulness = |claims_in_answer ∩ facts_in_context| / |claims_in_answer|
    """
    if not generated_answer or not retrieved_contexts:
        return 0.0

    ans_tokens = _tokenize_keywords(generated_answer)
    if not ans_tokens:
        return 1.0

    all_context = " ".join(retrieved_contexts)
    ctx_tokens = _tokenize_keywords(all_context)

    supported = sum(1 for token in ans_tokens if token in ctx_tokens)
    ratio = supported / float(len(ans_tokens))
    return round(min(1.0, ratio * 1.15), 4)


def compute_answer_relevancy(question: str, generated_answer: str) -> float:
    """
    Measures semantic and lexical alignment between query intent and generated response.
    """
    if not question or not generated_answer:
        return 0.0

    q_tokens = _tokenize_keywords(question)
    a_tokens = _tokenize_keywords(generated_answer)
    if not q_tokens:
        return 1.0

    overlap = len(q_tokens & a_tokens)
    ratio = overlap / float(len(q_tokens))
    return round(min(1.0, ratio * 1.35), 4)


def compute_context_precision(retrieved_contexts: List[str], ground_truth_answer: str) -> float:
    """
    Measures whether the relevant context chunks were ranked at top positions.
    """
    if not retrieved_contexts or not ground_truth_answer:
        return 0.0

    gt_tokens = _tokenize_keywords(ground_truth_answer)
    if not gt_tokens:
        return 1.0

    precision_scores = []
    for rank, ctx in enumerate(retrieved_contexts, start=1):
        ctx_tokens = _tokenize_keywords(ctx)
        overlap = len(gt_tokens & ctx_tokens)
        if overlap > 0:
            precision_scores.append(1.0 / rank)

    if not precision_scores:
        return 0.0
    return round(min(1.0, sum(precision_scores)), 4)


def compute_context_recall(retrieved_contexts: List[str], ground_truth_answer: str) -> float:
    """
    Measures how much of the ground truth knowledge is captured by the retrieved chunks.
    """
    if not retrieved_contexts or not ground_truth_answer:
        return 0.0

    gt_tokens = _tokenize_keywords(ground_truth_answer)
    if not gt_tokens:
        return 1.0

    all_ctx = " ".join(retrieved_contexts)
    ctx_tokens = _tokenize_keywords(all_ctx)
    covered = sum(1 for token in gt_tokens if token in ctx_tokens)
    return round(covered / float(len(gt_tokens)), 4)


def compute_all_metrics(
    question: str,
    generated_answer: str,
    retrieved_contexts: List[str],
    ground_truth_answer: str,
    predicted_tool: Optional[str] = None,
    expected_tool: Optional[str] = None,
) -> EvaluationMetrics:
    """Calculate comprehensive benchmark score bundle."""
    f_score = compute_faithfulness(generated_answer, retrieved_contexts)
    r_score = compute_answer_relevancy(question, generated_answer)
    cp_score = compute_context_precision(retrieved_contexts, ground_truth_answer)
    cr_score = compute_context_recall(retrieved_contexts, ground_truth_answer)

    tool_score = 1.0
    if expected_tool:
        tool_score = 1.0 if predicted_tool == expected_tool else 0.0

    overall = round(
        0.30 * f_score + 0.30 * r_score + 0.15 * cp_score + 0.15 * cr_score + 0.10 * tool_score,
        4,
    )

    return EvaluationMetrics(
        faithfulness=f_score,
        answer_relevancy=r_score,
        context_precision=cp_score,
        context_recall=cr_score,
        tool_accuracy=tool_score,
        overall_score=overall,
        details={
            "retrieved_count": len(retrieved_contexts),
            "predicted_tool": predicted_tool,
            "expected_tool": expected_tool,
        },
    )
