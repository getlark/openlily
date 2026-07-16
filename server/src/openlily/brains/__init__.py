"""Swappable brains (LLM harnesses) for the agent.

Select the active brain via ``default_brain`` in ``brains.yaml`` (copy
``brains.yaml.example``); without that file, ``brains/config.py``'s
``DEFAULT_BRAIN`` is used.
"""

from __future__ import annotations

from collections.abc import Callable

from openlily.tools.contracts import ToolName

# Cloud brains are lightweight, so import them eagerly. The local-model brain is
# NOT imported here: its module pulls in the optional on-device runtimes
# (mlx-whisper/whisper/kokoro -> torch/CUDA), which a cloud-only install won't
# have. It's loaded on demand in ``_load_local_whisper_ollama_kokoro`` instead.
from . import (
    cartesia_meta,
    cartesia_openai,
    openai_realtime,
    openai_standard,
)
from .base import (
    BrainName,
    BrainServices,
    BrainSpec,
)
from .config import get_brain_name, get_enabled_tools


def _load_local_whisper_ollama_kokoro() -> BrainSpec:
    """Import the local-model brain lazily, with an actionable error if its
    optional deps aren't installed.

    The brain's on-device runtimes live in the ``local-models`` extra (see
    ``pyproject.toml``), so a cloud-only install won't have them. Importing here
    (not at module load) keeps ``import brains`` working without those heavy
    deps, and turns a missing dep into a clear "install the extra" message rather
    than a raw ``ModuleNotFoundError``.
    """
    try:
        from . import local_whisper_ollama_kokoro
    except ImportError as e:
        raise RuntimeError(
            f"The {BrainName.LOCAL_WHISPER_OLLAMA_KOKORO.value!r} brain needs the "
            "optional on-device model dependencies, which aren't installed. Add "
            "them with `uv sync --extra local-models` (or "
            "`pip install '.[local-models]'`) and relaunch."
        ) from e
    return local_whisper_ollama_kokoro.SPEC


# Every built-in selectable brain -> a loader that returns its ``BrainSpec``.
# Cloud brains return their already-imported SPEC; the local brain imports on
# demand so its heavy optional deps are only required when it's actually selected.
_BRAIN_LOADERS: dict[BrainName, Callable[[], BrainSpec]] = {
    openai_standard.SPEC.name: lambda: openai_standard.SPEC,
    cartesia_openai.SPEC.name: lambda: cartesia_openai.SPEC,
    cartesia_meta.SPEC.name: lambda: cartesia_meta.SPEC,
    openai_realtime.SPEC.name: lambda: openai_realtime.SPEC,
    BrainName.LOCAL_WHISPER_OLLAMA_KOKORO: _load_local_whisper_ollama_kokoro,
}

# Brains registered at runtime by a library consumer via ``register_brain``.
# Keyed by the spec's name (a plain string, so custom brains aren't limited to
# the built-in ``BrainName`` members). Checked before the built-in loaders, so a
# consumer can also override a built-in brain by re-registering its name.
_CUSTOM_BRAINS: dict[str, BrainSpec] = {}


def register_brain(spec: BrainSpec) -> None:
    """Register a custom brain so ``get_brain(spec.name)`` (and the CLI) can select it.

    Lets a library consumer add a brain without editing this package: build a
    ``BrainSpec`` (its ``name`` may be any string, not just a ``BrainName``
    member) and register it at import time. Re-registering an existing name
    overrides it.
    """
    _CUSTOM_BRAINS[str(spec.name)] = spec


def get_brain(name: BrainName | str | None = None) -> BrainSpec:
    name = name or get_brain_name()
    key = str(name)
    if key in _CUSTOM_BRAINS:
        return _CUSTOM_BRAINS[key]
    for brain_name, loader in _BRAIN_LOADERS.items():
        if str(brain_name) == key:
            return loader()
    known = sorted([*(str(n) for n in _BRAIN_LOADERS), *_CUSTOM_BRAINS])
    raise ValueError(f"Unknown brain {name!r}; choose from {known}")


__all__ = [
    "BrainName",
    "BrainServices",
    "BrainSpec",
    "ToolName",
    "get_brain",
    "get_enabled_tools",
    "register_brain",
]
