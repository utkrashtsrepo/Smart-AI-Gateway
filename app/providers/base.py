"""Abstract base class and response dataclass for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ProviderResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Dict[str, Any]
    model_name: str


class BaseProvider(ABC):
    """Interface that every LLM provider backend must implement."""

    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ProviderResponse:
        """Issue an asynchronous chat completion request."""
        pass

