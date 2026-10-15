"""The note telling the model which integrations it can use."""

from __future__ import annotations

import pytest

from deep_agent.src.mcp_status.middleware import build_notice, connection_state


class TestNotice:
    def test_names_what_is_connected(self):
        notice = build_notice(["atlan", "miro"], [])
        assert "atlan" in notice and "miro" in notice
        assert "Connected and usable" in notice

    def test_names_what_is_not_and_says_what_to_do(self):
        # Knowing a tool will fail is only half of it — the model has
        # to be able to tell the user how to fix it.
        notice = build_notice([], ["slack"])
        assert "slack" in notice
        assert "Settings" in notice

    def test_is_ordered_so_it_does_not_churn_the_prompt_cache(self):
        assert build_notice(["b", "a"], []) == build_notice(["a", "b"], [])

    def test_is_empty_when_nothing_is_configured(self):
        assert build_notice([], []) == ""


class TestConnectionState:
    @pytest.mark.asyncio
    async def test_a_failure_reports_nothing_rather_than_guessing(self, monkeypatch):
        # A wrong list is worse than none: telling the model an
        # integration is unavailable when it works would stop it
        # using a tool the user is waiting on.
        async def boom(_user_id):
            raise RuntimeError("redis down")

        monkeypatch.setattr(
            "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections", boom
        )
        assert await connection_state("alice") == ([], [])

    @pytest.mark.asyncio
    async def test_splits_connected_from_unconnected(self, monkeypatch):
        async def connections(_user_id):
            return {
                "connections": [
                    {"mcp_name": "atlan", "connected": True},
                    {"mcp_name": "slack", "connected": False},
                ]
            }

        monkeypatch.setattr(
            "deep_agent.aegra.mcp_oauth_handlers.handle_mcp_connections", connections
        )
        assert await connection_state("alice") == (["atlan"], ["slack"])


class TestRequestOverride:
    def test_the_notice_never_enters_the_transcript(self):
        # `messages` reduces with add_messages, which appends — a
        # notice put there accumulates one copy per turn and is
        # stored in the thread forever. It goes on the request's
        # system message instead.
        import inspect

        from deep_agent.src.mcp_status import middleware

        source = inspect.getsource(middleware)
        # The comment explaining why names it, so look for a hook,
        # not a mention.
        assert "def before_model" not in source
        assert "def wrap_model_call" in source
        assert "def awrap_model_call" in source


class TestIdentity:
    def test_the_supplied_identity_is_used(self):
        # Regression: assigning to `user_id` inside `_user_id` made it
        # a local, shadowing the closure argument, and every model
        # call died with UnboundLocalError.
        from deep_agent.src.mcp_status.middleware import (
            build_mcp_status_middleware,
        )

        mw = build_mcp_status_middleware("alice")
        assert mw._user_id(object()) == "alice"

    def test_no_identity_is_not_an_error(self):
        from deep_agent.src.mcp_status.middleware import (
            build_mcp_status_middleware,
        )

        mw = build_mcp_status_middleware(None)
        assert mw._user_id(object()) == ""
