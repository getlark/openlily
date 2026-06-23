"""Tests for the browser tool's MCP launch args (``tools/browser/config.py``).

Pure env-driven argument building -- no MCP server is started and no browser is
launched.
"""

from __future__ import annotations

import pytest

from tools.browser.config import (
    BROWSER_CDP_ENDPOINT_ENV,
    BROWSER_MCP_BASE_ARGS,
    build_browser_mcp_args,
)


def test_appends_cdp_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BROWSER_CDP_ENDPOINT_ENV, "http://localhost:9222")
    args = build_browser_mcp_args()
    assert args[: len(BROWSER_MCP_BASE_ARGS)] == BROWSER_MCP_BASE_ARGS
    assert args[-2:] == ["--cdp-endpoint", "http://localhost:9222"]


def test_raises_when_endpoint_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The caller (setup_browser_tools) guards on this, so building args without
    # the endpoint set is a programming error -- surface it loudly.
    monkeypatch.delenv(BROWSER_CDP_ENDPOINT_ENV, raising=False)
    with pytest.raises(KeyError):
        build_browser_mcp_args()
