"""Provider exports and factory."""

from app.config import get_settings
from app.providers.base import BaseProvider, ProviderResponse
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockProvider


def get_provider(force_mock: bool = False) -> BaseProvider:
    """Provider factory returning either GroqProvider or MockProvider."""
    settings = get_settings()
    if force_mock or settings.is_mock_mode:
        return MockProvider()
    return GroqProvider()


__all__ = ["BaseProvider", "ProviderResponse", "GroqProvider", "MockProvider", "get_provider"]

