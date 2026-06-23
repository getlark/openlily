"""Tests for tool setup graceful-degradation (``tools/web``, ``tools/email``).

These verify the "skip with a warning, don't crash" contract when credentials
are missing, and that a configured provider's tools get wired in. No provider
SDK or network calls happen -- providers are stubbed.
"""

from __future__ import annotations

import pytest

import tools.email as email_pkg
import tools.web as web_pkg
from tools.email import setup_email_tools
from tools.email.config import USER_EMAIL_ENV
from tools.email.resend_provider import ResendProvider
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
