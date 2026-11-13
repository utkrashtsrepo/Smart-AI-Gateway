"""FastAPI Smart AI Gateway Application.

Combines:
1. Auto-Routing: Dynamic model selection based on task complexity.
2. LLMOps: Operational request logging, token accounting, cost tracking, and telemetry.
3. Reliability: In-memory response caching, retries with exponential backoff, and automatic fallback.
"""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.cache import response_cache
from app.config import get_settings
from app.database import db_manager
from app.logger import LLMOpsLogger
from app.providers import get_provider
from app.reliability import reliability_manager
from app.router import router
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    GatewayMetricsResponse,
    RequestLogEntry,
    SimplifiedChatRequest,
    SimplifiedChatResponse,
    Usage,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database on startup."""
    await db_manager.init_db()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Production-grade AI Gateway combining LLMOps, Auto-Routing, and Agent Evals",
    lifespan=lifespan,
)

# Enable CORS for frontend clients or dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & Web UI Dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the Web UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Smart AI Gateway API is running. Visit /docs for Swagger UI."}


@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health and readiness check."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "provider_mode": "mock" if settings.is_mock_mode else "groq",
        "cheap_model": settings.CHEAP_MODEL,
        "strong_model": settings.STRONG_MODEL,
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["Chat"])
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint enriched with Gateway routing and LLMOps."""
    start_time = time.perf_counter()
    endpoint = "/v1/chat/completions"

    # Step 1: Auto-Routing Analysis
    route_decision = router.route(
        messages=request.messages,
        requested_model=request.model,
        policy_override=request.policy,
    )
    routed_model = route_decision.routed_model

    # Step 2: Response Cache Check
    cache_key = None
    if request.use_cache and settings.CACHE_ENABLED:
        msg_dicts = [m.model_dump() for m in request.messages]
        cache_key = response_cache.generate_key(
            model=routed_model,
            messages=msg_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        cached_result = await response_cache.get(cache_key)
        if cached_result:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            # Log cache hit ($0.00 cost)
            await LLMOpsLogger.log_completion(
                endpoint=endpoint,
                requested_model=request.model,
                routed_model=routed_model,
                route_reason=f"CACHE HIT; {route_decision.route_reason}",
                complexity_score=route_decision.complexity_score,
                input_tokens=cached_result["usage"]["prompt_tokens"],
                output_tokens=cached_result["usage"]["completion_tokens"],
                latency_ms=latency_ms,
                cache_hit=True,
                status="success",
                tags=request.tags,
            )

            # Return cached response with updated gateway metadata
            return ChatCompletionResponse(
                id=f"cache-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=routed_model,
                choices=[
                    Choice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=cached_result["choices"][0]["message"]["content"],
                        ),
                    )
                ],
                usage=Usage(**cached_result["usage"]),
                gateway_metadata={
                    "routed_model": routed_model,
                    "complexity_score": route_decision.complexity_score,
                    "route_reason": route_decision.route_reason,
                    "latency_ms": latency_ms,
                    "cost_usd": 0.0,
                    "cache_hit": True,
                    "retries": 0,
                    "fallback_used": False,
                },
            )

    # Step 3: Provider Execution with Reliability (Retries + Fallback)
    provider = get_provider()
    msg_dicts = [m.model_dump() for m in request.messages]

    try:
        provider_resp, retries, fallback_used = await reliability_manager.execute_with_resilience(
            provider=provider,
            primary_model=routed_model,
            messages=msg_dicts,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        # Log failure
        await LLMOpsLogger.log_completion(
            endpoint=endpoint,
            requested_model=request.model,
            routed_model=routed_model,
            route_reason=route_decision.route_reason,
            complexity_score=route_decision.complexity_score,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            cache_hit=False,
            status="error",
            error_message=str(exc),
            tags=request.tags,
        )
        raise HTTPException(status_code=502, detail=f"Model provider error: {exc}")

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    effective_model = provider_resp.model_name

    # Step 4: LLMOps Logging & Cost Accounting
    cost_usd = await LLMOpsLogger.log_completion(
        endpoint=endpoint,
        requested_model=request.model,
        routed_model=effective_model,
        route_reason=route_decision.route_reason,
        complexity_score=route_decision.complexity_score,
        input_tokens=provider_resp.prompt_tokens,
        output_tokens=provider_resp.completion_tokens,
        latency_ms=latency_ms,
        cache_hit=False,
        status="success",
        retries=retries,
        fallback_used=fallback_used,
        tags=request.tags,
    )

    # Step 5: Save into Response Cache
    response_payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "created": int(time.time()),
        "model": effective_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": provider_resp.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": provider_resp.prompt_tokens,
            "completion_tokens": provider_resp.completion_tokens,
            "total_tokens": provider_resp.total_tokens,
        },
    }

    if cache_key and request.use_cache and settings.CACHE_ENABLED:
        await response_cache.set(cache_key, response_payload)

    return ChatCompletionResponse(
        id=response_payload["id"],
        created=response_payload["created"],
        model=effective_model,
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=provider_resp.content),
                finish_reason="stop",
            )
        ],
        usage=Usage(**response_payload["usage"]),
        gateway_metadata={
            "routed_model": effective_model,
            "complexity_score": route_decision.complexity_score,
            "route_reason": route_decision.route_reason,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "cache_hit": False,
            "retries": retries,
            "fallback_used": fallback_used,
        },
    )


@app.post("/v1/chat", response_model=SimplifiedChatResponse, tags=["Chat"])
async def simplified_chat(request: SimplifiedChatRequest) -> SimplifiedChatResponse:
    """Convenience single-prompt endpoint."""
    cmpl_req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=request.prompt)],
        model=request.model,
        temperature=request.temperature,
        policy=request.policy,
    )
    res = await chat_completions(cmpl_req)
    reply = res.choices[0].message.content if res.choices else ""
    meta = res.gateway_metadata
    return SimplifiedChatResponse(
        reply=reply,
        model_used=meta.get("routed_model", res.model),
        latency_ms=meta.get("latency_ms", 0.0),
        cost_usd=meta.get("cost_usd", 0.0),
        cache_hit=meta.get("cache_hit", False),
        route_reason=meta.get("route_reason", ""),
    )


@app.get("/v1/metrics", response_model=GatewayMetricsResponse, tags=["LLMOps"])
async def get_metrics() -> GatewayMetricsResponse:
    """Fetch aggregated LLMOps operational metrics (costs, latencies, cache rates, routing split)."""
    return await db_manager.get_metrics()


@app.get("/v1/logs", response_model=List[RequestLogEntry], tags=["LLMOps"])
async def get_logs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[RequestLogEntry]:
    """Retrieve detailed recent request log rows."""
    return await db_manager.get_logs(limit=limit, offset=offset)


@app.post("/v1/cache/clear", tags=["Reliability"])
async def clear_cache() -> Dict[str, str]:
    """Manually invalidate and clear the in-memory response cache."""
    await response_cache.clear()
    return {"message": "Response cache cleared successfully."}

