"""One MCP session per server per turn, not one per call."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_agent.aegra import mcp_host


@pytest.fixture(autouse=True)
def _clean_scope():
    mcp_host._open_sessions.set(None)
    mcp_host._session_stacks.clear()
    yield
    mcp_host._open_sessions.set(None)
    mcp_host._session_stacks.clear()


def _patched(session_obj, opened):
    """Patch everything mcp_session touches, counting handshakes."""

    @asynccontextmanager
    async def fake_session(_name):
        opened.append(1)
        yield session_obj

    client = MagicMock()
    client.session = fake_session
    return (
        patch.object(mcp_host, "ensure_mcp_apps_capability_advertised", MagicMock()),
        patch.object(mcp_host, "_get_enabled_server", MagicMock(return_value={})),
        patch.object(mcp_host, "_resolve_bearer", AsyncMock(return_value="tok")),
        patch.object(mcp_host, "_build_server_config", MagicMock(return_value={})),
        patch.object(mcp_host, "MultiServerMCPClient", MagicMock(return_value=client)),
    )


class TestSessionReuse:
    async def test_without_a_scope_every_call_opens_its_own(self):
        """The safe default: a session must not outlive its request."""
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                pass
            async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                pass
        assert len(opened) == 2

    async def test_inside_a_scope_the_session_is_reused(self):
        """Two listings used to mean two `initialize` handshakes.

        Measured at roughly twelve seconds against a server that
        answers each call in well under one, run back to back —
        which is what "things happen sequentially" looked like.
        """
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session(
                    "dv", user_id="u", sso_token=None
                ) as s1:
                    pass
                async with mcp_host.mcp_session(
                    "dv", user_id="u", sso_token=None
                ) as s2:
                    pass
        assert len(opened) == 1
        assert s1 is s2

    async def test_a_different_server_gets_its_own_session(self):
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                    pass
                async with mcp_host.mcp_session("atlan", user_id="u", sso_token=None):
                    pass
        assert len(opened) == 2

    async def test_a_different_user_never_shares_a_session(self):
        """A session is established with one user's bearer.

        Handing it to another user would send their request under
        someone else's authorization.
        """
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session("dv", user_id="alice", sso_token=None):
                    pass
                async with mcp_host.mcp_session("dv", user_id="bob", sso_token=None):
                    pass
        assert len(opened) == 2

    async def test_the_scope_closes_what_it_opened(self):
        opened: list[int] = []
        a, b, c, d, e = _patched(MagicMock(), opened)
        with a, b, c, d, e:
            async with mcp_host.mcp_session_scope():
                async with mcp_host.mcp_session("dv", user_id="u", sso_token=None):
                    pass
        assert mcp_host._open_sessions.get() is None
        assert mcp_host._session_stacks == {}
