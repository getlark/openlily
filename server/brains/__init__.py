"""Swappable brains (LLM harnesses) for the agent.

Select the active brain via ``default_brain`` in ``brains.yaml`` (copy
``brains.yaml.example``); without that file, ``brains/config.py``'s
``DEFAULT_BRAIN`` is used.
"""

from __future__ import annotations

from . import cartesia_openai, openai_realtime, openai_standard
from .base import (
    BrainName,
    BrainServices,
    BrainSpec,
    ToolBundle,
    close_tool_bundle,
    merge_tool_bundles,
    register_tool_bundle,
    tools_schema_from_bundle,
)
from .config import get_brain_name

BRAINS: dict[BrainName, BrainSpec] = {
    openai_standard.SPEC.name: openai_standard.SPEC,
    cartesia_openai.SPEC.name: cartesia_openai.SPEC,
    openai_realtime.SPEC.name: openai_realtime.SPEC,
}


def get_brain(name: BrainName | None = None) -> BrainSpec:
    name = name or get_brain_name()
    if name not in BRAINS:
        raise ValueError(f"Unknown brain {name!r}; choose from {sorted(BRAINS)}")
    return BRAINS[name]


__all__ = [
    "BRAINS",
    "BrainName",
    "BrainServices",
    "BrainSpec",
    "ToolBundle",
    "close_tool_bundle",
    "get_brain",
    "merge_tool_bundles",
    "register_tool_bundle",
    "tools_schema_from_bundle",
]
