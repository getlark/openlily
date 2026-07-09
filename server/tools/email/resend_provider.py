"""Resend-backed email provider.

Implements the agent's ``send_email_to_user`` tool with Resend. The model writes
the body in Markdown, which we render to HTML (keeping the Markdown as the
plain-text fallback) so emails arrive nicely formatted. Reads its credentials
from the environment: the API key (``RESEND_API_KEY``) and the verified sender
address (``EMAIL_FROM``, tied to a domain verified with Resend).
``is_configured`` reports whether both are present so ``setup_email_tools`` can
skip the tool when they aren't.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

import markdown
import resend
from loguru import logger
from pipecat.services.llm_service import FunctionCallParams

from .base import EmailProvider

# Markdown extensions used to render the body to HTML. ``extra`` covers tables,
# fenced code, etc.; ``sane_lists`` avoids surprising list merging; ``nl2br``
# turns single newlines into line breaks so the email reads the way the model
# wrote it rather than collapsing soft wraps into one paragraph.
_MARKDOWN_EXTENSIONS = ["extra", "sane_lists", "nl2br"]

# Resend's provider-specific credentials, both read from the environment. The
# sender is a credential too: it must be an address on a domain verified with
# Resend, so it's provider-specific rather than a shared default.
RESEND_API_KEY_ENV = "RESEND_API_KEY"
EMAIL_FROM_ENV = "EMAIL_FROM"


class ResendProvider(EmailProvider):
    """Sends email via Resend, rendering a Markdown body to HTML.

    The model writes the body in Markdown; we render it to HTML for the ``html``
    part and keep the original Markdown as the ``text`` part, so HTML clients see
    formatting while plain-text clients still get a readable fallback.
    """

    def __init__(self, recipient: str) -> None:
        super().__init__(recipient)
        api_key = os.getenv(RESEND_API_KEY_ENV)
        sender = os.getenv(EMAIL_FROM_ENV)
        if not api_key:
            raise RuntimeError(f"{RESEND_API_KEY_ENV} is required for the Resend email provider")
        if not sender:
            raise RuntimeError(f"{EMAIL_FROM_ENV} is required for the Resend email provider")
        self._api_key = api_key
        self._sender = sender

    @classmethod
    def is_configured(cls) -> bool:
        return bool(os.getenv(RESEND_API_KEY_ENV)) and bool(os.getenv(EMAIL_FROM_ENV))

    def create_tools(self) -> list[Callable[..., Awaitable[None]]]:
        async def send_email_to_user(params: FunctionCallParams, subject: str, body: str) -> None:
            """Send an email to the user's own email address.

            Use when the user asks to be emailed something, such as a summary or
            notes. The recipient is always the user; you only choose the subject
            and body.

            Args:
                subject: A short, descriptive subject line for the email.
                body: The body of the email, written in Markdown. Use formatting
                    such as headings, bold, bullet/numbered lists, and links
                    where it makes the content clearer. Do not write raw HTML.
            """
            try:
                # Resend's SDK is blocking, and we let it block: the send is
                # quick and we want its success/failure in hand before reporting
                # the result back to the LLM (below), so the model can speak the
                # actual outcome.
                self._send(subject, body)
            except Exception:
                # The body is omitted from the log to avoid recording content.
                logger.exception(
                    "send_email_to_user failed (to={!r}, subject={!r})",
                    self._recipient,
                    subject,
                )
                await params.result_callback(
                    {"error": "The email failed to send. Tell the user and offer to try again."}
                )
                return
            await params.result_callback({"status": "sent"})

        return [send_email_to_user]

    def _send(self, subject: str, body: str) -> None:
        """Send an email via Resend (synchronous; the SDK is blocking).

        Renders the Markdown ``body`` to HTML for the ``html`` part and keeps the
        original Markdown as the ``text`` part (the fallback for plain-text
        clients). Sets the API key on the module at call time so a rotated key is
        picked up without a restart.
        """
        resend.api_key = self._api_key
        html = markdown.markdown(body, extensions=_MARKDOWN_EXTENSIONS)
        resend.Emails.send(
            {
                "from": self._sender,
                "to": [self._recipient],
                "subject": subject,
                "html": html,
                "text": body,
            }
        )
