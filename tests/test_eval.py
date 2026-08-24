"""Unit tests for Evaluation metrics & Benchmark Harness."""

import pytest
from agentone.eval.metrics import (
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    compute_all_metrics,
)
from agentone.eval.benchmark_runner import BenchmarkRunner


def test_metrics_faithfulness():
    gen = "Enterprise SLA guarantees 99.95% monthly uptime."
    ctx = ["Our Enterprise plan SLA provides 99.95% uptime with dedicated failover."]
    score = compute_faithfulness(gen, ctx)
    assert score >= 0.70


def test_metrics_relevancy():
    q = "What is the uptime SLA for Enterprise tier?"
    ans = "The Enterprise tier offers a 99.95% uptime SLA guarantee."
    score = compute_answer_relevancy(q, ans)
    assert score >= 0.70


def test_compute_all_metrics_bundle():
    res = compute_all_metrics(
        question="What is the refund limit?",
        generated_answer="Refunds under $500 are processed automatically.",
        retrieved_contexts=["Refunds under $500.00 USD requested within 30 days are processed automatically."],
        ground_truth_answer="Transactions under $500 are eligible for automated refund.",
        predicted_tool="refund_process",
        expected_tool="refund_process",
    )
    assert res.overall_score >= 0.65
    assert res.tool_accuracy == 1.0


@pytest.mark.asyncio
async def test_benchmark_runner_execution():
    runner = BenchmarkRunner()
    res = await runner.run()
    assert res.total_cases > 0
    assert res.overall_system_score > 0.50
    assert res.avg_faithfulness > 0.50
