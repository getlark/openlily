"""Which brain the agent uses. One of the members of ``BrainName``.

``DEFAULT_BRAIN`` is the fallback used only when no ``brains.yaml`` exists.
Otherwise the brain is selected by the ``default_brain`` key in ``brains.yaml``
(required whenever that file is present; see ``brains/overrides.py``). Copy
``brains.yaml.example`` to ``brains.yaml`` to change it.
"""

from __future__ import annotations

from openlily.tools.contracts import ToolName

from .base import BrainName
from .overrides import get_brain_overrides

# Default brain when there's no ``brains.yaml``: Cartesia STT/TTS + OpenAI LLM.
DEFAULT_BRAIN = BrainName.CARTESIA_OPENAI


def get_brain_name() -> BrainName:
    """The configured brain: ``default_brain`` from brains.yaml, else ``DEFAULT_BRAIN``.

    When ``brains.yaml`` exists, ``default_brain`` is guaranteed to be set (the
    loader rejects a present file that omits it), so the fallback applies only
    when there's no file at all.
    """
    return get_brain_overrides().default_brain or DEFAULT_BRAIN


def get_enabled_tools() -> list[ToolName]:
    """The optional generic tools to enable, from ``tools`` in brains.yaml.

    Empty when there's no file, no ``tools`` key, or an empty list -- the
    always-on session tool is wired in regardless by ``tools.runtime``.
    """
    return get_brain_overrides().tools
