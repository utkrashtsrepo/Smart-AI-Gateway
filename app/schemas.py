"""Pydantic schemas for Smart AI Gateway requests, responses, and metrics."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with Gateway extension parameters."""
    messages: List[ChatMessage]
    model: str = Field(default="auto", description="Model name or 'auto', 'cheap', 'strong'")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=1024, ge=1)
    
    # Custom Gateway Controls
    policy: Optional[Literal["auto", "always_cheap", "always_strong"]] = Field(
        default=None,
        description="Routing policy override. If provided, overrides model='auto'."
    )
    use_cache: bool = Field(default=True, description="Enable response caching")
    tags: Optional[List[str]] = Field(default=None, description="Optional metadata tags for analytics")


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RouteDecision(BaseModel):
    routed_model: str
    policy: str
    complexity_score: float
    route_reason: str
    signals: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response enriched with LLMOps gateway metadata."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

    # Gateway LLMOps Metadata
    gateway_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metrics: latency_ms, cost_usd, cache_hit, retries, route_reason"
    )


class SimplifiedChatRequest(BaseModel):
    """Single-prompt convenience request."""
    prompt: str
    model: str = "auto"
    temperature: float = 0.7
    policy: Optional[Literal["auto", "always_cheap", "always_strong"]] = None


class SimplifiedChatResponse(BaseModel):
    """Single-prompt convenience response."""
    reply: str
    model_used: str
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    route_reason: str


class RequestLogEntry(BaseModel):
    id: int
    timestamp: str
    endpoint: str
    requested_model: str
    routed_model: str
    route_reason: str
    complexity_score: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    status: str
    error_message: Optional[str] = None
    retries: int
    fallback_used: bool


class GatewayMetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate_pct: float
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    cache_hits: int
    cache_hit_rate_pct: float
    route_distribution: Dict[str, int]
    cost_by_model: Dict[str, float]

