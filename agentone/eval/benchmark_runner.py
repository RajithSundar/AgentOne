"""Benchmark execution runner for end-to-end multi-agent evaluation."""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agentone.agents.graph import compile_agent_graph
from agentone.eval.metrics import EvaluationMetrics, compute_all_metrics


class BenchmarkResult(BaseModel):
    """Evaluation summary over a dataset."""
    total_cases: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_tool_accuracy: float
    overall_system_score: float
    avg_latency_ms: float
    case_results: List[Dict[str, Any]] = Field(default_factory=list)


class BenchmarkRunner:
    """Automated benchmark harness."""

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or "agentone/eval/benchmark_dataset.json"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load JSON benchmark test cases."""
        path = Path(self.dataset_path)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    async def run(self) -> BenchmarkResult:
        """Run all test cases through LangGraph and compute benchmark metrics."""
        cases = self.load_dataset()
        if not cases:
            return BenchmarkResult(
                total_cases=0,
                avg_faithfulness=0.0,
                avg_answer_relevancy=0.0,
                avg_context_precision=0.0,
                avg_context_recall=0.0,
                avg_tool_accuracy=0.0,
                overall_system_score=0.0,
                avg_latency_ms=0.0,
            )

        app = compile_agent_graph()
        case_results: List[Dict[str, Any]] = []

        total_f = 0.0
        total_r = 0.0
        total_cp = 0.0
        total_cr = 0.0
        total_ta = 0.0
        total_latency = 0.0

        for case in cases:
            q = case["question"]
            gt = case["ground_truth_answer"]
            expected_tool = case.get("expected_tool")

            start_t = time.perf_counter()

            initial_state = {
                "thread_id": f"eval-{case['id']}",
                "original_query": q,
                "sanitized_query": q,
                "messages": [{"role": "user", "content": q}],
                "current_agent": "supervisor",
                "next_agent": None,
                "iteration_count": 0,
                "is_complete": False,
                "status": "running",
                "intent": None,
                "priority": None,
                "customer_id": None,
                "sentiment_score": None,
                "retrieved_documents": [],
                "knowledge_gap_detected": False,
                "web_search_needed": False,
                "pending_actions": [],
                "executed_actions": [],
                "approval_requests": [],
                "requires_human_approval": False,
                "guardrail_verdicts": [],
                "critic_notes": None,
                "critic_approved": False,
                "final_response": None,
                "structured_output": None,
                "audit_trail": [],
                "telemetry": {},
            }

            result_state = await app.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"eval-{case['id']}"}},
            )

            latency = round((time.perf_counter() - start_t) * 1000.0, 2)
            total_latency += latency

            gen_ans = result_state.get("final_response", "")
            retrieved = [
                d.get("content", "") for d in result_state.get("retrieved_documents", [])
            ]

            executed_tools = [
                a.get("tool_name") for a in result_state.get("executed_actions", [])
            ] + [
                a.get("tool_name") for a in result_state.get("pending_actions", [])
            ]
            predicted_tool = executed_tools[0] if executed_tools else None

            metrics = compute_all_metrics(
                question=q,
                generated_answer=gen_ans,
                retrieved_contexts=retrieved,
                ground_truth_answer=gt,
                predicted_tool=predicted_tool,
                expected_tool=expected_tool,
            )

            total_f += metrics.faithfulness
            total_r += metrics.answer_relevancy
            total_cp += metrics.context_precision
            total_cr += metrics.context_recall
            total_ta += metrics.tool_accuracy

            case_results.append({
                "case_id": case["id"],
                "category": case.get("category"),
                "question": q,
                "generated_answer": gen_ans,
                "metrics": metrics.model_dump(),
                "latency_ms": latency,
            })

        n = len(cases)
        return BenchmarkResult(
            total_cases=n,
            avg_faithfulness=round(total_f / n, 4),
            avg_answer_relevancy=round(total_r / n, 4),
            avg_context_precision=round(total_cp / n, 4),
            avg_context_recall=round(total_cr / n, 4),
            avg_tool_accuracy=round(total_ta / n, 4),
            overall_system_score=round(
                (0.30 * total_f + 0.30 * total_r + 0.15 * total_cp + 0.15 * total_cr + 0.10 * total_ta) / n,
                4,
            ),
            avg_latency_ms=round(total_latency / n, 2),
            case_results=case_results,
        )


async def run_standard_benchmark() -> BenchmarkResult:
    """Convenience entry point for running standard benchmarks."""
    runner = BenchmarkRunner()
    return await runner.run()


if __name__ == "__main__":
    res = asyncio.run(run_standard_benchmark())
    print(json.dumps(res.model_dump(), indent=2))
