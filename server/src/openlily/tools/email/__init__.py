"""Generic email tool, organized by provider.

Like ``tools/browser/``, this is brain-agnostic: the registry-driven runtime
layers it onto every brain when enabled. It mirrors the LiveKit
agent's ``send_email_to_user`` tool -- a single ``send_email_to_user`` function
that sends plain-text email to the user's own configured address (this is a
single-user assistant, so there's no free-form recipient).

The implementation is organized like ``tools/web/``: a registry of providers
(Resend today; SendGrid, SES, ... could follow), one selected via
``config.EMAIL_PROVIDER``. Each provider owns its own credentials and tool
schema. Adding a provider is a new module under ``tools/email/`` plus a registry
entry here and (optionally) selecting it in ``config.py``.

The runtime fails fast if the tool is enabled without both ``USER_EMAIL`` and
the selected provider's credentials. Direct ``setup_email_tools`` calls retain
their graceful empty-bundle behavior.
"""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger

from ..bundle import ToolBundle
from ..contracts import ToolActivation, ToolBackend, ToolId, ToolName, ToolSpec
from .base import EmailProvider
from .config import EMAIL_PROVIDER, USER_EMAIL_ENV, get_user_email

# Prompt snippet describing the email capability. Attached to the bundle so the
# system prompt mentions email only when the tool is actually wired in. Mirrors
# the LiveKit agent's prompt guidance. Provider-agnostic: the model just needs
# to know the capability exists.
EMAIL_INSTRUCTION = (
    "You can email the user; emails always go to the user's own address. Use it "
    "when the user asks to be emailed something, such as a summary or notes. "
    "Format the email nicely when possible (like in markdown format)."
)

def _load_resend_provider() -> type[EmailProvider]:
    """Import the Resend provider lazily, with an actionable error if its optional
    dependencies (``resend``, ``markdown``) aren't installed.

    They live in the ``email`` extra (see ``pyproject.toml``), so a base install
    won't have them. Importing here (not at module load) keeps ``import openlily``
    working without them, and turns a missing dep into a clear "install the extra"
    message rather than a raw ``ModuleNotFoundError``.
    """
    try:
        from .resend_provider import ResendProvider
    except ImportError as e:
        raise RuntimeError(
            "The email tool needs the optional 'email' dependencies, which "
            "aren't installed. Add them with `uv sync --extra email` (or "
            "`pip install 'openlily[email]'`) and relaunch."
        ) from e
    return ResendProvider


# Registry of available providers -> a loader that imports and returns the
# provider class on demand. Add new ones here; select via config.py.
_PROVIDER_LOADERS: dict[str, Callable[[], type[EmailProvider]]] = {
    "resend": _load_resend_provider,
}


def _provider_cls() -> type[EmailProvider]:
    try:
        loader = _PROVIDER_LOADERS[EMAIL_PROVIDER]
    except KeyError:
        raise ValueError(
            f"Unknown email provider {EMAIL_PROVIDER!r}; "
            f"available: {sorted(_PROVIDER_LOADERS)}"
        )
    return loader()


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


SPEC = ToolSpec(
    id=ToolId.EMAIL,
    activation=ToolActivation.CONFIGURED,
    backend=ToolBackend.LOCAL,
    setup=setup_email_tools,
    configurable_name=ToolName.EMAIL,
    is_configured=email_is_configured,
    requirement="USER_EMAIL and the email provider's credentials",
)


__all__ = ["EMAIL_INSTRUCTION", "SPEC", "email_is_configured", "setup_email_tools"]
