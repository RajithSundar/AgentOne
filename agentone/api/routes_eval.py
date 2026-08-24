"""Evaluation and benchmark API routes."""

from typing import Optional
from fastapi import APIRouter, HTTPException
from agentone.eval.benchmark_runner import BenchmarkRunner, BenchmarkResult

router = APIRouter(prefix="/api/eval", tags=["Evaluation & Benchmarks"])

_LATEST_BENCHMARK_RESULT: Optional[BenchmarkResult] = None


@router.post("/run", response_model=BenchmarkResult)
async def run_evaluation_benchmark():
    """Execute end-to-end evaluation harness across all standard test cases."""
    global _LATEST_BENCHMARK_RESULT
    try:
        runner = BenchmarkRunner()
        result = await runner.run()
        _LATEST_BENCHMARK_RESULT = result
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {str(e)}")


@router.get("/latest")
async def get_latest_benchmark():
    """Retrieve the most recent benchmark scores."""
    global _LATEST_BENCHMARK_RESULT
    if _LATEST_BENCHMARK_RESULT is None:
        return {
            "status": "not_run",
            "message": "No benchmark run completed yet. Trigger POST /api/eval/run.",
        }
    return _LATEST_BENCHMARK_RESULT
