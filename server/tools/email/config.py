"""Email tool configuration: which provider, and the shared recipient/sender.

``EMAIL_PROVIDER`` selects one of the providers registered in
``tools/email/__init__.py``. The recipient (``USER_EMAIL``) is provider-agnostic
-- every provider sends to the user's own address -- so it lives here, while each
provider's own credentials (e.g. ``RESEND_API_KEY`` and the verified sender
address) live in that provider's module.

``USER_EMAIL`` is read fresh from the environment at call time: this module is
imported before ``load_dotenv()`` runs, so capturing it at import would miss
anything set in ``.env``.
"""

from __future__ import annotations

import os

# Which email provider to use. One of the keys in the registry in
# ``tools/email/__init__.py``. Add new providers there, then select one here.
EMAIL_PROVIDER = "resend"

# The user's own email address -- the sole recipient. Without it the email tool
# is skipped (we won't wire up a "send email" capability with nowhere to send).
USER_EMAIL_ENV = "USER_EMAIL"


def get_user_email() -> str | None:
    """The configured recipient (the user's own address), or ``None`` when unset."""
    email = os.getenv(USER_EMAIL_ENV)
    return email.strip() if email and email.strip() else None
