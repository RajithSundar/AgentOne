"""Integration tests for FastAPI REST endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from agentone.api.server import app


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_agent_execute_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "query": "What is the SLA for enterprise uptime?",
            "thread_id": "test-api-01",
        }
        res = await client.post("/api/agents/execute", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["thread_id"] == "test-api-01"
        assert len(data["final_response"]) > 0


@pytest.mark.asyncio
async def test_rag_search_and_ingest():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test Ingest
        ingest_payload = {
            "title": "custom_runbook.md",
            "content": "# Custom Policy\nCustom SLA response is 5 minutes.",
            "category": "custom",
        }
        res_ingest = await client.post("/api/rag/ingest", json=ingest_payload)
        assert res_ingest.status_code == 200
        assert res_ingest.json()["chunks_created"] >= 1

        # Test Search
        res_search = await client.get("/api/rag/search?query=Custom+SLA+5+minutes&top_k=2")
        assert res_search.status_code == 200
        assert res_search.json()["results_count"] >= 1


@pytest.mark.asyncio
async def test_hitl_pending_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/api/hitl/pending")
        assert res.status_code == 200
        assert isinstance(res.json(), list)


@pytest.mark.asyncio
async def test_eval_benchmark_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post("/api/eval/run")
        assert res.status_code == 200
        data = res.json()
        assert data["total_cases"] > 0
        assert data["overall_system_score"] > 0.50
