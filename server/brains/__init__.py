"""Swappable brains (LLM harnesses) for the agent.

Select the active brain via ``default_brain`` in ``brains.yaml`` (copy
``brains.yaml.example``); without that file, ``brains/config.py``'s
``DEFAULT_BRAIN`` is used.
"""

from __future__ import annotations

from collections.abc import Callable

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
    ToolBundle,
    ToolName,
    close_tool_bundle,
    merge_tool_bundles,
    register_tool_bundle,
    tools_schema_from_bundle,
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


# Every selectable brain -> a loader that returns its ``BrainSpec``. Cloud
# brains return their already-imported SPEC; the local brain imports on demand so
# its heavy optional deps are only required when it's actually selected.
_BRAIN_LOADERS: dict[BrainName, Callable[[], BrainSpec]] = {
    openai_standard.SPEC.name: lambda: openai_standard.SPEC,
    cartesia_openai.SPEC.name: lambda: cartesia_openai.SPEC,
    cartesia_meta.SPEC.name: lambda: cartesia_meta.SPEC,
    openai_realtime.SPEC.name: lambda: openai_realtime.SPEC,
    BrainName.LOCAL_WHISPER_OLLAMA_KOKORO: _load_local_whisper_ollama_kokoro,
}


def get_brain(name: BrainName | None = None) -> BrainSpec:
    name = name or get_brain_name()
    if name not in _BRAIN_LOADERS:
        raise ValueError(f"Unknown brain {name!r}; choose from {sorted(_BRAIN_LOADERS)}")
    return _BRAIN_LOADERS[name]()


__all__ = [
    "BrainName",
    "BrainServices",
    "BrainSpec",
    "ToolBundle",
    "ToolName",
    "close_tool_bundle",
    "get_brain",
    "get_enabled_tools",
    "merge_tool_bundles",
    "register_tool_bundle",
    "tools_schema_from_bundle",
]
