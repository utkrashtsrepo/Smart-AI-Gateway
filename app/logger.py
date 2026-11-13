"""LLMOps logging service: calculates costs, tracks latency, and records request telemetry."""

import logging
from typing import Any, Dict, List, Optional
from app.config import get_model_pricing
from app.database import db_manager

logger = logging.getLogger("gateway.llmops")


class LLMOpsLogger:
    """Handles operational logging and cost accounting for every gateway request."""

    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate total USD cost based on catalog pricing per 1M tokens."""
        pricing = get_model_pricing(model_name)
        return pricing.calculate_cost(input_tokens, output_tokens)

    @classmethod
    async def log_completion(
        cls,
        endpoint: str,
        requested_model: str,
        routed_model: str,
        route_reason: str,
        complexity_score: float,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cache_hit: bool,
        status: str = "success",
        error_message: Optional[str] = None,
        retries: int = 0,
        fallback_used: bool = False,
        tags: Optional[List[str]] = None,
    ) -> float:
        """Compute cost and persist log row asynchronously. Returns calculated cost."""
        # Cache hits incur $0.00 model cost
        cost_usd = 0.0 if cache_hit else cls.calculate_cost(routed_model, input_tokens, output_tokens)

        try:
            await db_manager.log_request(
                endpoint=endpoint,
                requested_model=requested_model,
                routed_model=routed_model,
                route_reason=route_reason,
                complexity_score=complexity_score,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                cache_hit=cache_hit,
                status=status,
                error_message=error_message,
                retries=retries,
                fallback_used=fallback_used,
                tags=tags,
            )
        except Exception as e:
            logger.error(f"Failed to record request log in SQLite: {e}")

        return cost_usd

