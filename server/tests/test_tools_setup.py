"""Tests for tool setup graceful-degradation (``tools/web``, ``tools/email``).

These verify the "skip with a warning, don't crash" contract when credentials
are missing, and that a configured provider's tools get wired in. No provider
SDK or network calls happen -- providers are stubbed.
"""

from __future__ import annotations

import pytest

import tools.email as email_pkg
import tools.runtime as tools_runtime
import tools.web as web_pkg
from tools.bundle import ToolBundle
from tools.contracts import ToolActivation, ToolBackend, ToolId, ToolName, ToolSpec
from tools.email import setup_email_tools
from tools.email.config import USER_EMAIL_ENV
from tools.email.resend_provider import ResendProvider
from tools.runtime import setup_tools
from tools.web import WEB_SEARCH_INSTRUCTION, setup_web_tools
from tools.web.exa import ExaProvider


async def _noop() -> None:  # a stand-in "tool" callable
    return None


def test_web_unconfigured_returns_empty_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ExaProvider, "is_configured", classmethod(lambda cls: False))
    bundle = setup_web_tools()
    assert bundle.standard_tools == []
    assert bundle.instructions == []


def test_web_configured_wires_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ExaProvider, "is_configured", classmethod(lambda cls: True))
    monkeypatch.setattr(ExaProvider, "__init__", lambda self: None)
    monkeypatch.setattr(ExaProvider, "create_tools", lambda self: [_noop])
    bundle = setup_web_tools()
    assert bundle.standard_tools == [_noop]
    assert bundle.instructions == [WEB_SEARCH_INSTRUCTION]


def test_web_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_pkg, "WEB_SEARCH_PROVIDER", "nope")
    with pytest.raises(ValueError):
        setup_web_tools()


async def test_email_unconfigured_returns_empty_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Recipient present, but provider credentials missing -> still skipped.
    monkeypatch.setenv(USER_EMAIL_ENV, "me@example.com")
    monkeypatch.setattr(ResendProvider, "is_configured", classmethod(lambda cls: False))
    bundle = await setup_email_tools()
    assert bundle.standard_tools == []
    assert bundle.instructions == []


async def test_email_missing_recipient_returns_empty_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Provider configured, but no recipient -> skipped.
    monkeypatch.delenv(USER_EMAIL_ENV, raising=False)
    monkeypatch.setattr(ResendProvider, "is_configured", classmethod(lambda cls: True))
    bundle = await setup_email_tools()
    assert bundle.standard_tools == []
    assert bundle.instructions == []


async def test_email_configured_wires_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_EMAIL_ENV, "me@example.com")
    monkeypatch.setattr(ResendProvider, "is_configured", classmethod(lambda cls: True))
    monkeypatch.setattr(ResendProvider, "__init__", lambda self, recipient: None)
    monkeypatch.setattr(ResendProvider, "create_tools", lambda self: [_noop])
    bundle = await setup_email_tools()
    assert bundle.standard_tools == [_noop]
    assert bundle.instructions  # the email instruction snippet is attached


async def test_email_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(email_pkg, "EMAIL_PROVIDER", "nope")
    with pytest.raises(ValueError):
        await setup_email_tools()


async def test_generic_tools_always_includes_session(monkeypatch: pytest.MonkeyPatch) -> None:
    # No optional tools enabled -> the always-on session tool is still wired in.
    monkeypatch.setattr(tools_runtime, "get_enabled_tools", lambda: [])
    bundle = await setup_tools()
    assert bundle.standard_tools  # session's end_session tool
    assert bundle.instructions


async def test_generic_tools_enabled_and_configured_wires(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_setup() -> ToolBundle:
        return ToolBundle(standard_tools=[_noop], instructions=["X capability"])

    fake_spec = ToolSpec(
        id=ToolId.EMAIL,
        activation=ToolActivation.CONFIGURED,
        backend=ToolBackend.LOCAL,
        setup=fake_setup,
        configurable_name=ToolName.EMAIL,
        is_configured=lambda: True,
        requirement="email credentials",
    )
    monkeypatch.setattr(tools_runtime, "get_enabled_tools", lambda: [ToolName.EMAIL])
    monkeypatch.setattr(tools_runtime, "get_configurable_tool", lambda name: fake_spec)

    bundle = await setup_tools()

    assert _noop in bundle.standard_tools  # the enabled tool's tool
    assert "X capability" in bundle.instructions


async def test_generic_tools_enabled_but_unconfigured_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_setup() -> ToolBundle:  # must never run when unconfigured
        raise AssertionError("setup should not be called for an unconfigured tool")

    fake_spec = ToolSpec(
        id=ToolId.EMAIL,
        activation=ToolActivation.CONFIGURED,
        backend=ToolBackend.LOCAL,
        setup=fail_setup,
        configurable_name=ToolName.EMAIL,
        is_configured=lambda: False,
        requirement="EMAIL_TEST_CREDENTIALS",
    )
    monkeypatch.setattr(tools_runtime, "get_enabled_tools", lambda: [ToolName.EMAIL])
    monkeypatch.setattr(tools_runtime, "get_configurable_tool", lambda name: fake_spec)

    with pytest.raises(RuntimeError) as exc:
        await setup_tools()
    assert "EMAIL_TEST_CREDENTIALS" in str(exc.value)


async def test_brain_declared_tool_is_resolved_from_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_setup() -> ToolBundle:
        return ToolBundle(standard_tools=[_noop], instructions=["Hosted web"])

    fake_spec = ToolSpec(
        id=ToolId.WEB_HOSTED,
        activation=ToolActivation.BRAIN,
        backend=ToolBackend.HOSTED,
        setup=fake_setup,
    )
    monkeypatch.setattr(tools_runtime, "always_on_tools", lambda: ())
    monkeypatch.setattr(tools_runtime, "get_enabled_tools", lambda: [])
    monkeypatch.setattr(tools_runtime, "get_tool_spec", lambda tool_id: fake_spec)

    bundle = await tools_runtime.setup_tools((ToolId.WEB_HOSTED,))

    assert bundle.standard_tools == [_noop]
    assert bundle.instructions == ["Hosted web"]


async def test_enabled_mcp_tool_requires_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unused_setup() -> ToolBundle:
        raise AssertionError("MCP setup must use the pool")

    async def unused_connect():
        raise AssertionError("MCP connect must happen during warmup")

    fake_spec = ToolSpec(
        id=ToolId.X,
        activation=ToolActivation.CONFIGURED,
        backend=ToolBackend.MCP,
        setup=unused_setup,
        configurable_name=ToolName.X,
        is_configured=lambda: True,
        requirement="X_APP_BEARER_TOKEN",
        mcp_connect=unused_connect,
        mcp_instructions=lambda: ["X capability"],
        warmup_failure_hint="Check X credentials.",
    )

    class _ColdPool:
        def is_ready(self, tool_id: ToolId) -> bool:
            return False

    monkeypatch.setattr(tools_runtime, "always_on_tools", lambda: ())
    monkeypatch.setattr(tools_runtime, "get_enabled_tools", lambda: [ToolName.X])
    monkeypatch.setattr(tools_runtime, "get_configurable_tool", lambda name: fake_spec)
    monkeypatch.setattr(
        tools_runtime.MCPToolsPool,
        "get",
        classmethod(lambda cls: _ColdPool()),
    )

    with pytest.raises(RuntimeError, match="warmup_tools"):
        await tools_runtime.setup_tools()


async def test_brain_declared_mcp_tool_is_warmed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unused_setup() -> ToolBundle:
        return ToolBundle()

    async def unused_connect():
        raise AssertionError("The fake pool should not invoke the connector")

    fake_spec = ToolSpec(
        id=ToolId.WEB_EXA,
        activation=ToolActivation.BRAIN,
        backend=ToolBackend.MCP,
        setup=unused_setup,
        mcp_connect=unused_connect,
        mcp_instructions=lambda: ["Brain MCP"],
        warmup_failure_hint="Check brain MCP.",
    )

    class _RecordingPool:
        specs: list[ToolSpec] = []

        async def warmup(self, specs: list[ToolSpec]) -> None:
            self.specs = specs

    pool = _RecordingPool()
    monkeypatch.setattr(tools_runtime, "get_enabled_tools", lambda: [])
    monkeypatch.setattr(tools_runtime, "get_tool_spec", lambda tool_id: fake_spec)
    monkeypatch.setattr(
        tools_runtime.MCPToolsPool,
        "get",
        classmethod(lambda cls: pool),
    )

    await tools_runtime.warmup_tools((ToolId.WEB_EXA,))

    assert pool.specs == [fake_spec]
