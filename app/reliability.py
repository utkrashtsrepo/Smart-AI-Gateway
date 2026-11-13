"""Reliability layer: Retries with exponential backoff and automatic provider/model fallback."""

import asyncio
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import get_settings
from app.providers.base import BaseProvider, ProviderResponse

logger = logging.getLogger("gateway.reliability")


def is_retryable_error(exc: Exception) -> bool:
    """Classify whether an exception is transient and worth retrying."""
    msg = str(exc).lower()
    retryable_markers = [
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "connection",
        "service unavailable",
    ]
    return any(marker in msg for marker in retryable_markers)


class ReliabilityManager:
    """Executes requests with jittered exponential backoff and fallback failover."""

    def __init__(
        self,
        max_retries: Optional[int] = None,
        base_backoff: Optional[float] = None,
        max_backoff: Optional[float] = None,
    ):
        settings = get_settings()
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
        self.base_backoff = base_backoff if base_backoff is not None else settings.BACKOFF_BASE_SECONDS
        self.max_backoff = max_backoff if max_backoff is not None else settings.BACKOFF_MAX_SECONDS
        self.fallback_model = settings.FALLBACK_MODEL

    async def execute_with_resilience(
        self,
        provider: BaseProvider,
        primary_model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        fallback_model: Optional[str] = None,
    ) -> Tuple[ProviderResponse, int, bool]:
        """Execute request with retries on primary model, failing over to backup model if needed.
        
        Returns:
            Tuple[ProviderResponse, retry_count, fallback_used]
        """
        target_fallback = fallback_model or self.fallback_model
        retries_spent = 0
        last_exception: Optional[Exception] = None

        # Phase 1: Try Primary Model with Exponential Backoff
        for attempt in range(self.max_retries + 1):
            try:
                response = await provider.chat_completion(
                    model=primary_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response, retries_spent, False
            except Exception as e:
                last_exception = e
                retries_spent += 1
                if attempt < self.max_retries and is_retryable_error(e):
                    # Check for rate limit wait hints (e.g. Groq 429 "try again in Xs")
                    wait_hint = 0.0
                    msg_lower = str(e).lower()
                    m_sec = re.search(r"try again in ([\d\.]+)s", msg_lower)
                    m_ms = re.search(r"try again in ([\d\.]+)ms", msg_lower)
                    if m_sec:
                        wait_hint = float(m_sec.group(1)) + 0.5
                    elif m_ms:
                        wait_hint = (float(m_ms.group(1)) / 1000.0) + 0.3

                    # Exponential backoff with random jitter or rate limit hint
                    calc_backoff = min(self.max_backoff, self.base_backoff * (2**attempt))
                    delay = max(calc_backoff, wait_hint)
                    jitter = random.uniform(0.1, 0.4)
                    total_delay = delay + jitter
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {primary_model}: {e}. Retrying in {total_delay:.2f}s..."
                    )
                    await asyncio.sleep(total_delay)
                else:
                    logger.error(
                        f"Primary model {primary_model} exhausted retries or encountered non-retryable error: {e}"
                    )
                    break

        # Phase 2: Automatic Fallback to Secondary Model
        if target_fallback and target_fallback.lower() != primary_model.lower():
            logger.info(
                f"Triggering automatic fallback from {primary_model} -> {target_fallback}"
            )
            try:
                fallback_response = await provider.chat_completion(
                    model=target_fallback,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return fallback_response, retries_spent, True
            except Exception as fb_err:
                logger.error(f"Fallback model {target_fallback} also failed: {fb_err}")
                raise RuntimeError(
                    f"All attempts failed. Primary error: {last_exception}. Fallback error: {fb_err}"
                ) from fb_err

        # If no fallback or fallback identical to primary, raise original exception
        raise last_exception or RuntimeError("Unknown gateway execution failure")


reliability_manager = ReliabilityManager()

