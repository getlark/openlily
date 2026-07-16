"""openlily - a reusable Pipecat voice-agent toolkit.

Build an out-of-the-box voice agent from a config, or compose the individual
pieces into your own Pipecat pipeline. Everything is modular:

- High-level factory: :func:`create_agent` (and :func:`build_pipeline` /
  :func:`build_worker`) turn an :class:`AgentConfig` into a ready pipeline +
  worker for any transport (local audio, WebRTC dev runner, Pipecat Cloud).
- Swappable *brains* (STT/LLM/TTS harnesses) selected by name or supplied
  directly; add your own with :func:`register_brain`.
- Composable *tools* with a central registry; add your own with
  :func:`register_tool`.
- Optional flourishes you can toggle or drop in yourself: the readiness chime,
  the soft "working" cue, the idle keep-alive, and the conversation log observer.

Example (turnkey)::

    import openlily

    config = openlily.AgentConfig(brain="cartesia_openai", enabled_tools=["email"])
    await openlily.warmup(config)
    agent = await openlily.create_agent(my_transport, config)
    # ... add agent.worker to a WorkerRunner and run it ...

Example (compose your own)::

    from openlily import WorkingSoundProcessor, chime_pcm, get_brain

The local terminal bot lives in :mod:`openlily.cli`; local-audio building blocks
(the gated transport, wake word) live in :mod:`openlily.local`.
"""

from __future__ import annotations

from openlily.agent import (
    Agent,
    build_pipeline,
    build_worker,
    create_agent,
    warmup,
)
from openlily.brains import (
    BrainName,
    BrainServices,
    BrainSpec,
    get_brain,
    register_brain,
)
from openlily.config import (
    AgentConfig,
    WorkingSoundConfig,
)
from openlily.idle_keepalive import BotBusyFrame, IdleKeepaliveProcessor
from openlily.observers import ConversationLogObserver
from openlily.prompt import build_system_instruction
from openlily.sound import ReadinessChimeFrame, chime_pcm, working_sound_pcm
from openlily.tools.bundle import (
    ToolBundle,
    close_tool_bundle,
    merge_tool_bundles,
    register_tool_bundle,
    tools_schema_from_bundle,
)
from openlily.tools.contracts import (
    ToolActivation,
    ToolBackend,
    ToolId,
    ToolName,
    ToolSpec,
)
from openlily.tools.registry import register_tool
from openlily.tools.runtime import setup_tools, shutdown_tools, warmup_tools
from openlily.working_sound import WorkingSoundProcessor

__version__ = "0.1.0"

__all__ = [
    # Factory / config
    "Agent",
    "AgentConfig",
    "WorkingSoundConfig",
    "build_pipeline",
    "build_worker",
    "create_agent",
    "warmup",
    # Brains
    "BrainName",
    "BrainServices",
    "BrainSpec",
    "get_brain",
    "register_brain",
    # Tools
    "ToolActivation",
    "ToolBackend",
    "ToolBundle",
    "ToolId",
    "ToolName",
    "ToolSpec",
    "close_tool_bundle",
    "merge_tool_bundles",
    "register_tool",
    "register_tool_bundle",
    "setup_tools",
    "shutdown_tools",
    "tools_schema_from_bundle",
    "warmup_tools",
    # Components (compose-your-own)
    "BotBusyFrame",
    "ConversationLogObserver",
    "IdleKeepaliveProcessor",
    "ReadinessChimeFrame",
    "WorkingSoundProcessor",
    "build_system_instruction",
    "chime_pcm",
    "working_sound_pcm",
    "__version__",
]
