"""
Centralized LLM Factory
========================
Provides a unified interface for creating LLM instances
across different providers (Ollama local/cloud, OpenAI, Anthropic).
"""

import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential

from core.constants import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_GPT54_CONTEXT_WINDOW,
    DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_CLOUD_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
)

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA_LOCAL = "ollama_local"
    OLLAMA_CLOUD = "ollama_cloud"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class ModelContextPolicy:
    """Model-aware default context window policy."""

    model_pattern: str
    default_context_window: int

    def matches(self, model_name: str) -> bool:
        normalized = (model_name or "").strip().lower()
        return normalized.startswith(self.model_pattern.lower())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMFactory:
    """Factory for creating LLM instances across providers.

    Supports Ollama (local/cloud), OpenAI, and Anthropic with
    retry logic for transient failures.
    """

    _DEFAULT_MODELS = {
        LLMProvider.OLLAMA_LOCAL: DEFAULT_OLLAMA_MODEL,
        LLMProvider.OLLAMA_CLOUD: DEFAULT_OLLAMA_MODEL,
        LLMProvider.OPENAI: DEFAULT_OPENAI_MODEL,
        LLMProvider.ANTHROPIC: DEFAULT_ANTHROPIC_MODEL,
    }
    _MODEL_CONTEXT_POLICIES = (
        ModelContextPolicy(
            model_pattern="gpt-5.4",
            default_context_window=DEFAULT_GPT54_CONTEXT_WINDOW,
        ),
    )
    _ANTHROPIC_MODELS_WITHOUT_TEMPERATURE = (
        "claude-opus-4-7",
    )

    @staticmethod
    def _request_timeout() -> Optional[float]:
        if DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS <= 0:
            return None
        return DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS

    @classmethod
    def resolve_context_window(
        cls,
        model_name: str,
        explicit_context_window: Optional[int] = None,
    ) -> int:
        """Resolve the effective context window for a model.

        Explicit CLI/runtime overrides always win. Otherwise fall back to the
        first matching model policy, then the global default.
        """
        if explicit_context_window is not None:
            return explicit_context_window
        for policy in cls._MODEL_CONTEXT_POLICIES:
            if policy.matches(model_name):
                return policy.default_context_window
        return DEFAULT_CONTEXT_WINDOW

    @classmethod
    def get_context_policy(cls, model_name: str) -> ModelContextPolicy:
        """Return the matching context policy for a model."""
        for policy in cls._MODEL_CONTEXT_POLICIES:
            if policy.matches(model_name):
                return policy
        return ModelContextPolicy(
            model_pattern="default",
            default_context_window=DEFAULT_CONTEXT_WINDOW,
        )

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def create(
        provider: LLMProvider,
        model_name: str,
        temperature: float = DEFAULT_TEMPERATURE,
        context_window: Optional[int] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an LLM instance for the specified provider.

        Args:
            provider: The LLM provider to use.
            model_name: Name/ID of the model.
            temperature: Sampling temperature (0-1).
            context_window: Context window size in tokens.
            max_tokens: Maximum output tokens.
            base_url: Base URL for API (required for Ollama cloud).
            api_key: API key for OpenAI/Anthropic (or use env vars).
            **kwargs: Additional provider-specific arguments.

        Returns:
            Configured BaseChatModel instance.

        Raises:
            ValueError: If provider is unsupported or required args missing.
        """
        resolved_context_window = LLMFactory.resolve_context_window(
            model_name=model_name,
            explicit_context_window=context_window,
        )
        if provider in (LLMProvider.OLLAMA_LOCAL, LLMProvider.OLLAMA_CLOUD):
            if provider == LLMProvider.OLLAMA_CLOUD:
                resolved_base = base_url or DEFAULT_OLLAMA_CLOUD_BASE_URL
                if not resolved_base:
                    raise ValueError("base_url required for Ollama cloud provider")
                resolved_key = api_key or os.environ.get("OLLAMA_CLOUD_API_KEY") or os.environ.get("OLLAMA_API_KEY")
            else:
                resolved_base = base_url or DEFAULT_OLLAMA_BASE_URL
                resolved_key = api_key or os.environ.get("OLLAMA_API_KEY")
            return LLMFactory._create_ollama(
                model_name=model_name,
                base_url=resolved_base,
                temperature=temperature,
                context_window=resolved_context_window,
                max_tokens=max_tokens,
                api_key=resolved_key,
                **kwargs,
            )
        elif provider == LLMProvider.OPENAI:
            return LLMFactory._create_openai(
                model_name=model_name or DEFAULT_OPENAI_MODEL,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        elif provider == LLMProvider.ANTHROPIC:
            return LLMFactory._create_anthropic(
                model_name=model_name or DEFAULT_ANTHROPIC_MODEL,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _create_ollama(
        model_name: str,
        base_url: str,
        temperature: float,
        context_window: int,
        max_tokens: int,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an Ollama LLM instance."""
        from langchain_ollama import ChatOllama

        extra = {k: v for k, v in kwargs.items() if k != "verbose"}
        ollama_kwargs = dict(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=context_window,
            num_predict=max_tokens,
            verbose=kwargs.get("verbose", True),
            **extra,
        )
        if api_key:
            ollama_kwargs["client_kwargs"] = {
                "headers": {"Authorization": f"Bearer {api_key}"}
            }
        return ChatOllama(**ollama_kwargs)

    @staticmethod
    def _create_openai(
        model_name: str,
        api_key: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an OpenAI LLM instance."""
        from langchain_openai import ChatOpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )
        return ChatOpenAI(
            model=model_name,
            api_key=resolved_key,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=LLMFactory._request_timeout(),
            **kwargs,
        )

    @staticmethod
    def _create_anthropic(
        model_name: str,
        api_key: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs: Any,
    ) -> BaseChatModel:
        """Create an Anthropic LLM instance."""
        from langchain_anthropic import ChatAnthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Provide via api_key parameter "
                "or set ANTHROPIC_API_KEY environment variable."
            )
        anthropic_kwargs = dict(
            model=model_name,
            api_key=resolved_key,
            max_tokens=max_tokens,
            default_request_timeout=LLMFactory._request_timeout(),
            **kwargs,
        )
        normalized_model = (model_name or "").strip().lower()
        if not normalized_model.startswith(LLMFactory._ANTHROPIC_MODELS_WITHOUT_TEMPERATURE):
            anthropic_kwargs["temperature"] = temperature
        return ChatAnthropic(**anthropic_kwargs)

    @classmethod
    def get_default_model(cls, provider: LLMProvider) -> str:
        """Get the default model name for a provider."""
        return cls._DEFAULT_MODELS.get(provider, "unknown")

    @staticmethod
    def list_providers() -> list:
        """List all available provider values."""
        return [p.value for p in LLMProvider]
