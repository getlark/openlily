"""Generic email tool, organized by provider.

Like ``tools/browser/``, this is brain-agnostic: it's layered onto every brain
centrally in ``bot.py`` via ``setup_generic_tools``. It mirrors the LiveKit
agent's ``send_email_to_user`` tool -- a single ``send_email_to_user`` function
that sends plain-text email to the user's own configured address (this is a
single-user assistant, so there's no free-form recipient).

The implementation is organized like ``tools/web/``: a registry of providers
(Resend today; SendGrid, SES, ... could follow), one selected via
``config.EMAIL_PROVIDER``. Each provider owns its own credentials and tool
schema. Adding a provider is a new module under ``tools/email/`` plus a registry
entry here and (optionally) selecting it in ``config.py``.

The tool is wired in only when both the recipient (``USER_EMAIL``) and the
selected provider's credentials are configured; otherwise ``setup_email_tools``
logs a warning and returns an empty bundle, so the session runs without an email
capability -- the same graceful degradation the browser tool uses.
"""

from __future__ import annotations

from loguru import logger

from brains.base import ToolBundle

from .base import EmailProvider
from .config import EMAIL_PROVIDER, USER_EMAIL_ENV, get_user_email
from .resend_provider import ResendProvider

# Prompt snippet describing the email capability. Attached to the bundle so the
# system prompt mentions email only when the tool is actually wired in. Mirrors
# the LiveKit agent's prompt guidance. Provider-agnostic: the model just needs
# to know the capability exists.
EMAIL_INSTRUCTION = (
    "You can email the user; emails always go to the user's own address. Use it "
    "when the user asks to be emailed something, such as a summary or notes. "
    "Format the email nicely when possible (like in markdown format)."
)

# Registry of available providers. Add new ones here; select via config.py.
_PROVIDERS: dict[str, type[EmailProvider]] = {
    "resend": ResendProvider,
}


def _provider_cls() -> type[EmailProvider]:
    try:
        return _PROVIDERS[EMAIL_PROVIDER]
    except KeyError:
        raise ValueError(
            f"Unknown email provider {EMAIL_PROVIDER!r}; available: {sorted(_PROVIDERS)}"
        )


def email_is_configured() -> bool:
    """Whether the email tool is fully configured: a recipient *and* the
    selected provider's credentials are present.

    Used to fail fast when the tool is enabled in ``brains.yaml`` but something
    it needs (``USER_EMAIL`` or the provider's key/sender) is missing.
    """
    return get_user_email() is not None and _provider_cls().is_configured()


async def setup_email_tools() -> ToolBundle:
    """Build the email tool from the selected provider, if it's configured.

    Returns an empty ``ToolBundle`` (so the session continues without email) when
    either the recipient (``USER_EMAIL``) or the selected provider's credentials
    are missing, logging a warning that says what to set.
    """
    provider_cls = _provider_cls()
    recipient = get_user_email()

    if recipient is None or not provider_cls.is_configured():
        logger.warning(
            f"Email tool unavailable: set {USER_EMAIL_ENV} and the "
            f"{EMAIL_PROVIDER!r} provider's credentials to enable it. "
            "Continuing without email tool."
        )
        return ToolBundle()

    provider = provider_cls(recipient=recipient)
    tools = provider.create_tools()

    logger.info(f"Email tool ready (provider={EMAIL_PROVIDER})")
    return ToolBundle(
        standard_tools=list(tools),
        instructions=[EMAIL_INSTRUCTION],
    )


__all__ = ["EMAIL_INSTRUCTION", "email_is_configured", "setup_email_tools"]
