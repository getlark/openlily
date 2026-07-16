"""Generic contract for email providers.

An *email provider* is a backend that can send mail on the user's behalf (Resend
today; SendGrid, SES, ... could follow). Each provider owns its own credentials
and client and defines the agent's email tool as a Pipecat direct function, so a
tool's schema matches exactly what that provider supports. Selecting a provider
and feeding its tools to the agent stays uniform across backends -- mirroring the
``ToolProvider`` setup in ``tools/web/``.
"""

from __future__ import annotations

from ..base import ToolProvider


class EmailProvider(ToolProvider):
    """A backend that sends email to the user's own address.

    The recipient is always the user (this is a single-user assistant), so it's
    fixed at construction rather than chosen per call. Everything else -- the
    API key *and* the verified sender address, which is tied to the provider's
    own verified domain -- is a provider-specific credential: subclasses read it
    in ``__init__`` and report whether those credentials are present via
    ``is_configured`` (declared on ``ToolProvider``) -- checked *before*
    construction so an unconfigured provider can be skipped with a warning
    rather than raising.
    """

    def __init__(self, recipient: str) -> None:
        self._recipient = recipient
