"""The middleware that puts the self-description in front of the model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.src.self_knowledge import middleware as mw


def _request(system_message=None):
    request = MagicMock()
    request.system_message = system_message
    request.override = MagicMock(return_value="overridden")
    return request


def _build(notice="## What you cannot do\n\n- nothing", **kwargs):
    with patch(
        "deep_agent.src.self_knowledge.describe.build_self_description",
        return_value=notice,
    ):
        return mw.build_self_knowledge_middleware(**kwargs)


class TestBuild:
    def test_none_when_there_is_nothing_to_say(self):
        assert _build(notice="") is None

    def test_built_when_there_is(self):
        assert _build() is not None


class TestApply:
    @pytest.mark.asyncio
    async def test_attaches_to_an_absent_system_message(self):
        built = _build()
        request = _request()
        with patch.object(mw, "_apply", wraps=mw._apply):
            await built.awrap_model_call(request, AsyncMock(return_value="answered"))
        request.override.assert_called_once()

    @pytest.mark.asyncio
    async def test_appends_to_an_existing_system_message(self):
        from langchain_core.messages import SystemMessage

        built = _build()
        request = _request(SystemMessage(content="You are an agent."))
        await built.awrap_model_call(request, AsyncMock())
        attached = request.override.call_args.kwargs["system_message"]
        assert attached.content.startswith("You are an agent.")
        assert "cannot do" in attached.content

    @pytest.mark.asyncio
    async def test_appends_a_block_to_structured_content(self):
        from langchain_core.messages import SystemMessage

        built = _build()
        request = _request(SystemMessage(content=[{"type": "text", "text": "base"}]))
        await built.awrap_model_call(request, AsyncMock())
        blocks = request.override.call_args.kwargs["system_message"].content
        assert blocks[-1]["text"].startswith("## What")

    @pytest.mark.asyncio
    async def test_an_unattachable_request_still_answers(self):
        # A missing notice costs the model some context; failing
        # here would cost the answer.
        built = _build()
        request = MagicMock()
        request.system_message = SimpleNamespace(content=object())
        request.override.side_effect = RuntimeError("cannot override")
        result = await built.awrap_model_call(request, AsyncMock(return_value="ok"))
        assert result == "ok"


class TestConnectionState:
    @pytest.mark.asyncio
    async def test_asks_for_connection_state_once_and_caches_it(self):
        state = AsyncMock(return_value=(["gitlab"], ["slack"]))
        with (
            patch("deep_agent.src.mcp_status.middleware.connection_state", new=state),
            patch(
                "deep_agent.src.self_knowledge.describe.build_self_description",
                return_value="notice",
            ),
        ):
            built = mw.build_self_knowledge_middleware("u1")
            await built.awrap_model_call(_request(), AsyncMock())
            await built.awrap_model_call(_request(), AsyncMock())
        assert state.await_count == 1

    @pytest.mark.asyncio
    async def test_no_identity_means_no_lookup(self):
        # Without a user the state is not knowable, and guessing
        # produces the exact wrong answer: "nothing is connected".
        state = AsyncMock()
        with (
            patch("deep_agent.src.mcp_status.middleware.connection_state", new=state),
            patch(
                "deep_agent.src.self_knowledge.describe.build_self_description",
                return_value="notice",
            ),
        ):
            built = mw.build_self_knowledge_middleware(None)
            await built.awrap_model_call(_request(), AsyncMock())
        state.assert_not_awaited()

    def test_sync_path_uses_what_the_async_path_cached(self):
        built = _build()
        request = _request()
        assert built.wrap_model_call(request, MagicMock(return_value="ok")) == "ok"
