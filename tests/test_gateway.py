"""Integration tests for FastAPI endpoints: completions, metrics, logs, health."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app
from app.database import db_manager


@pytest.fixture(autouse=True)
async def setup_db():
    await db_manager.init_db()


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "cheap_model" in data


@pytest.mark.asyncio
async def test_chat_completions_auto_routing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "use_cache": False,
        }
        res = await client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        meta = data.get("gateway_metadata", {})
        assert meta["cost_usd"] >= 0.0
        assert "route_reason" in meta


@pytest.mark.asyncio
async def test_caching_behavior():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Tell me a unique fact about Neptune."}
            ],
            "use_cache": True,
        }
        # First request -> cache miss
        res1 = await client.post("/v1/chat/completions", json=payload)
        assert res1.status_code == 200
        assert res1.json()["gateway_metadata"]["cache_hit"] is False

        # Second request -> cache hit (< 10ms, cost = $0.0)
        res2 = await client.post("/v1/chat/completions", json=payload)
        assert res2.status_code == 200
        assert res2.json()["gateway_metadata"]["cache_hit"] is True
        assert res2.json()["gateway_metadata"]["cost_usd"] == 0.0


@pytest.mark.asyncio
async def test_metrics_and_logs():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Trigger a completion
        await client.post(
            "/v1/chat",
            json={"prompt": "Hello test prompt", "model": "auto"},
        )

        # Query metrics
        m_res = await client.get("/v1/metrics")
        assert m_res.status_code == 200
        metrics = m_res.json()
        assert metrics["total_requests"] >= 1

        # Query logs
        l_res = await client.get("/v1/logs?limit=5")
        assert l_res.status_code == 200
        logs = l_res.json()
        assert len(logs) >= 1
        assert logs[0]["endpoint"] in ("/v1/chat", "/v1/chat/completions")

