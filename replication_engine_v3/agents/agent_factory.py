"""
Shared LangChain v1 agent factory with middleware and optional checkpointing.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from core.run_context import RunContext

logger = logging.getLogger(__name__)

try:
    from langchain.agents.middleware import AgentMiddleware
except ImportError:  # pragma: no cover - depends on installed LangChain version
    AgentMiddleware = object  # type: ignore[assignment]

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover - depends on installed LangGraph version
    InMemorySaver = None  # type: ignore[assignment]

try:
    from langgraph.store.memory import InMemoryStore
except ImportError:  # pragma: no cover - depends on installed LangGraph version
    InMemoryStore = None  # type: ignore[assignment]


class ToolErrorMiddleware(AgentMiddleware):  # type: ignore[misc]
    """Convert tool exceptions into explicit tool messages."""

    def wrap_tool_call(
        self,
        request: Any,
        handler: Any,
    ) -> Any:
        try:
            return handler(request)
        except Exception as exc:  # pragma: no cover - defensive middleware
            logger.exception("Tool call failed")
            tool_call = getattr(request, "tool_call", {}) or {}
            return ToolMessage(
                content=f"Tool error: {exc}",
                tool_call_id=tool_call.get("id", "tool-error"),
            )


def create_replication_agent(
    model: Any,
    tools: Iterable[BaseTool],
    system_prompt: str,
    middleware: Optional[List[Any]] = None,
    context_schema: type[BaseModel] | type[RunContext] = RunContext,
    checkpointer: Any = None,
    store: Any = None,
) -> Any:
    """Create a LangChain v1 agent with sensible defaults for this project."""
    resolved_middleware = [ToolErrorMiddleware()]
    if middleware:
        resolved_middleware.extend(middleware)

    if checkpointer is None and InMemorySaver is not None:
        checkpointer = InMemorySaver()
    if store is None and InMemoryStore is not None:
        store = InMemoryStore()

    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=resolved_middleware,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
    )
