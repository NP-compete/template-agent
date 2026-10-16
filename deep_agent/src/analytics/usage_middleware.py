"""Record token usage where the model call actually happens.

This replaces a LangChain callback registered through
`register_configure_hook`. The hook registered successfully, logged
that it had, and was never invoked for a single model call the agent
made — so every run recorded zero tokens while the Usage tab
reported them as fact.

The callback worked in isolation: invoking a chat model directly
runs the configure hooks and the handler fires. Inside the graph it
does not, and a mechanism that works in a probe and not in
production is not a mechanism. Middleware is the agent's own
extension point, already proven on this exact path by the audit
middleware, and it holds the response rather than hoping a global
hook is consulted.

The write is awaited, not fired and forgotten. The first version
scheduled it with `create_task` so the reply would not wait on a
database round trip — and nothing was ever written, because the run
finishes immediately afterwards and the task was collected before it
ran. A few milliseconds on a pooled connection is a worse trade than
a dashboard that reads zero forever.

Failures are swallowed inside `record_llm_usage`: a write that does
not land must cost a number on a dashboard, never the reply.
"""

from __future__ import annotations

from typing import Any

from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()


def _run_id() -> str | None:
    """The Aegra run this call belongs to.

    `create_run_config` pins `configurable.run_id`; the same place
    `token_budget.identity` reads thread and user from.
    """
    try:
        from langgraph.config import get_config

        configurable = (get_config() or {}).get("configurable") or {}
    except Exception:
        return None
    value = configurable.get("run_id")
    return str(value) if value else None


def _messages_of(response: Any) -> list[Any]:
    result = getattr(response, "result", None)
    if isinstance(result, list):
        return result
    return [response] if response is not None else []


def _model_of(messages: list[Any]) -> str | None:
    for message in messages:
        meta = getattr(message, "response_metadata", None)
        if isinstance(meta, dict):
            for key in ("model_name", "model", "model_version"):
                value = meta.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _tokens_of(messages: list[Any]) -> tuple[int, int]:
    from deep_agent.src.token_budget.service import extract_tokens_from_message

    input_tokens = 0
    output_tokens = 0
    for message in messages:
        try:
            in_t, out_t = extract_tokens_from_message(message)
        except Exception:
            continue
        input_tokens += in_t
        output_tokens += out_t
    return input_tokens, output_tokens


def build_usage_middleware() -> Any:
    """Middleware recording tokens per model call, or None if unavailable."""
    try:
        from langchain.agents.middleware.types import AgentMiddleware
    except ImportError:
        logger.warning("AgentMiddleware unavailable — usage recording disabled")
        return None

    class UsageRecordingMiddleware(AgentMiddleware):
        """Add each model call's tokens to its run."""

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            # The sync path cannot await the write and must not block
            # a thread on it. Every real call goes through the async
            # path; this exists so the sync one still answers.
            return handler(request)

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            response = await handler(request)
            await _record(response)
            return response

    return UsageRecordingMiddleware()


async def _record(response: Any) -> None:
    """Add this call's tokens to its run. Never raises."""
    try:
        run_id = _run_id()
        if not run_id:
            # Evals, warmups and anything outside a request have no
            # run row to attribute to. Not an error.
            logger.debug("usage_middleware: no run id")
            return

        messages = _messages_of(response)
        input_tokens, output_tokens = _tokens_of(messages)
        model = _model_of(messages)

        from deep_agent.src.analytics.usage import record_llm_usage

        await record_llm_usage(run_id, model, input_tokens, output_tokens)
        logger.debug(
            "usage_middleware: recorded run=%s model=%s in=%d out=%d",
            run_id,
            model,
            input_tokens,
            output_tokens,
        )
    except Exception:
        # The reply is worth more than the measurement of it.
        logger.debug("usage_middleware: could not record usage", exc_info=True)
