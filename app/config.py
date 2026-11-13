"""Configuration settings and pricing table for Smart AI Gateway."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing:
    """Token pricing in USD per 1 Million tokens."""
    def __init__(self, input_per_1m: float, output_per_1m: float):
        self.input_per_1m = input_per_1m
        self.output_per_1m = output_per_1m

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000.0) * self.input_per_1m
        output_cost = (output_tokens / 1_000_000.0) * self.output_per_1m
        return round(input_cost + output_cost, 7)


# Pricing table aligned with standard Groq / Open Source frontier API rates
PRICING_CATALOG: Dict[str, ModelPricing] = {
    # Cheap tier ($0.05 / $0.08 per 1M tokens)
    "openai/gpt-oss-20b": ModelPricing(input_per_1m=0.05, output_per_1m=0.08),
    "llama-3.1-8b-instant": ModelPricing(input_per_1m=0.05, output_per_1m=0.08),
    "llama3-8b-8192": ModelPricing(input_per_1m=0.05, output_per_1m=0.08),
    "mock-cheap": ModelPricing(input_per_1m=0.05, output_per_1m=0.08),

    # Strong tier ($0.59 / $0.79 per 1M tokens)
    "openai/gpt-oss-120b": ModelPricing(input_per_1m=0.59, output_per_1m=0.79),
    "llama-3.3-70b-versatile": ModelPricing(input_per_1m=0.59, output_per_1m=0.79),
    "llama3-70b-8192": ModelPricing(input_per_1m=0.59, output_per_1m=0.79),
    "mock-strong": ModelPricing(input_per_1m=0.59, output_per_1m=0.79),

    # Fallback default for unknown models
    "default": ModelPricing(input_per_1m=0.20, output_per_1m=0.30),
}


class Settings(BaseSettings):
    """Gateway settings loaded from environment or .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    APP_ENV: Literal["development", "testing", "production"] = "development"
    APP_TITLE: str = "Smart AI Gateway"
    APP_VERSION: str = "0.1.0"

    # API Keys
    GROQ_API_KEY: str = ""
    USE_MOCK_PROVIDER: bool = False

    # Models
    CHEAP_MODEL: str = "openai/gpt-oss-20b"
    STRONG_MODEL: str = "openai/gpt-oss-120b"
    FALLBACK_MODEL: str = "openai/gpt-oss-20b"

    # Routing Thresholds
    ROUTER_COMPLEXITY_THRESHOLD: float = 0.40

    # Caching
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_ENTRIES: int = 1000

    # Reliability
    MAX_RETRIES: int = 3
    BACKOFF_BASE_SECONDS: float = 0.5
    BACKOFF_MAX_SECONDS: float = 4.0
    REQUEST_TIMEOUT_SECONDS: float = 30.0

    # Database
    DATABASE_PATH: str = "data/gateway_logs.db"

    @property
    def is_mock_mode(self) -> bool:
        """Determines whether mock provider should be used."""
        return self.USE_MOCK_PROVIDER or not bool(self.GROQ_API_KEY.strip())


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    # Ensure data folder exists
    db_path = Path(settings.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


def get_model_pricing(model_name: str) -> ModelPricing:
    """Retrieve pricing object for a given model name."""
    clean_name = model_name.strip().lower()
    return PRICING_CATALOG.get(clean_name, PRICING_CATALOG["default"])

