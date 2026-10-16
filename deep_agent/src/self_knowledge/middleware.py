"""Put the agent's own description in front of the model.

Replaces the MCP status middleware rather than sitting beside it.
Two middlewares each appending their own block left the model with
two descriptions of itself to reconcile, and no way to tell that
"datadog is not connected" and "you may delegate to investigator,
which reaches datadog" were about the same thing.

Added to the *request*, never to state. Returning it from
`before_model` would append a copy per turn through the
`add_messages` reducer, and the thread would carry one for every
exchange, forever.
"""

from __future__ import annotations

import time
from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

# Connection state changes only when somebody completes an OAuth
# flow. Asking per model call would put a round trip in front of
# every turn.
_CACHE_SECONDS = 60


def build_self_knowledge_middleware(
    user_id: str | None = None, model_name: str = ""
) -> Any:
    """Middleware describing the agent to itself, or None if there is nothing to say."""
    from deep_agent.src.self_knowledge.describe import build_self_description

    try:
        from langchain.agents.middleware.types import AgentMiddleware
        from langchain_core.messages import SystemMessage
    except ImportError:
        logger.warning("AgentMiddleware unavailable — self-knowledge disabled")
        return None

    # The config-derived half never changes within a process, so it
    # is built once; only the connection state is per-user and
    # refreshed.
    cache: dict[str, tuple[float, str]] = {}

    def _notice(connected: list[str], unconnected: list[str]) -> str:
        return build_self_description(model_name, connected, unconnected)

    static_notice = _notice([], [])

    class SelfKnowledgeMiddleware(AgentMiddleware):
        """Tell the model what it is, what it can reach, and what is refused."""

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            return await handler(_apply(request, await self._notice(), SystemMessage))

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            # The sync path cannot await the connection lookup, and
            # blocking the event loop to find out would cost more
            # than the integration list is worth. It uses whatever
            # the async path cached, falling back to the half that
            # needs no lookup.
            hit = cache.get(user_id or "")
            return handler(
                _apply(request, hit[1] if hit else static_notice, SystemMessage)
            )

        async def _notice(self) -> str:
            key = user_id or ""
            hit = cache.get(key)
            now = time.monotonic()
            if hit and now - hit[0] < _CACHE_SECONDS:
                return hit[1]

            connected: list[str] = []
            unconnected: list[str] = []
            if user_id:
                # Without an identity the connection state is not
                # knowable, and guessing produces the exact wrong
                # answer: "nothing is connected".
                try:
                    from deep_agent.src.mcp_status.middleware import connection_state

                    connected, unconnected = await connection_state(user_id)
                except Exception:
                    logger.debug("Could not read MCP connection state", exc_info=True)

            notice = _notice(connected, unconnected)
            cache[key] = (now, notice)
            return notice

    if not static_notice:
        logger.info("Self-knowledge: nothing to announce")
        return None

    logger.info("Self-knowledge active for user=%s model=%s", user_id, model_name)
    return SelfKnowledgeMiddleware()


def _apply(request: Any, notice: str, system_message_cls: Any) -> Any:
    """Append the notice to the request's system message."""
    if not notice:
        return request
    try:
        existing = getattr(request, "system_message", None)
        if existing is None:
            updated = system_message_cls(content=notice)
        elif isinstance(getattr(existing, "content", None), str):
            updated = existing.model_copy(
                update={"content": f"{existing.content}\n\n{notice}".strip()}
            )
        else:
            blocks = list(existing.content or [])
            blocks.append({"type": "text", "text": notice})
            updated = existing.model_copy(update={"content": blocks})
        return request.override(system_message=updated)
    except Exception:
        # A missing notice costs the model some context; failing
        # here would cost the answer.
        logger.debug("Could not attach the self-knowledge notice", exc_info=True)
        return request
