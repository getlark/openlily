"""Configuration objects for assembling an openlily agent.

``AgentConfig`` is the single injection point for the factory in
:mod:`openlily.agent`: it carries the brain, tool selection, and every knob the
pipeline/worker used to hardcode, each defaulting to today's behavior so an
empty-ish config reproduces the local bot. The library core reads *only* this
object -- never files or environment variables. The CLI (and only the CLI) turns
``brains.yaml`` + ``.env`` into an ``AgentConfig`` and hands it to the factory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.pipeline.worker import PipelineParams

    from openlily.brains import BrainName, BrainSpec
    from openlily.tools.contracts import ToolName

# Grace period after the user stops (or a tool call starts) before the first
# "working" motif plays; fast turns produce bot audio within this window and stay
# silent. Mirrors the value the local bot used.
WORKING_SOUND_INITIAL_DELAY_SECS = 0.8

# The idle keep-alive heartbeat (see idle_keepalive.py) is a correctness fix, not
# a flourish, so it is always on and its tuning is fixed rather than configurable.
# Upper bound on the heartbeat interval.
IDLE_KEEPALIVE_MAX_INTERVAL_SECS = 5.0

# Absolute worst-case cap on a single continuous "bot is busy" window, so a stuck
# session (e.g. STT returns nothing, LLM/TTS stalls) can eventually idle out.
IDLE_KEEPALIVE_MAX_BUSY_SECS = 300.0

# Seconds of silence (no user *or* bot speech) before the session ends itself.
DEFAULT_IDLE_TIMEOUT_SECS = 30.0


@dataclass(frozen=True)
class WorkingSoundConfig:
    """Tuning for the soft "working" cue played while the bot is busy."""

    initial_delay_secs: float = WORKING_SOUND_INITIAL_DELAY_SECS


@dataclass
class AgentConfig:
    """Everything needed to assemble one openlily pipeline + worker.

    Only ``brain`` is required; every other field defaults to the behavior of the
    stock local bot, so ``AgentConfig(brain=...)`` reproduces it. Toggle a
    feature off (e.g. ``working_sound=False``) or pass a config object to tune it.
    """

    # The brain to run: a ready ``BrainSpec`` (fully custom), or a ``BrainName`` /
    # plain string resolved against the brain registry (built-ins + any
    # ``register_brain``'d ones).
    brain: BrainSpec | BrainName | str

    # Optional configurable tools to enable by name (what ``brains.yaml``'s
    # ``tools:`` list holds). The brain's own declared tools and the always-on
    # session tool are added automatically -- do not list them here.
    enabled_tools: Sequence[ToolName | str] = field(default_factory=tuple)

    # Soft "working" cue during the gap before the bot speaks. ``True`` uses the
    # defaults; a ``WorkingSoundConfig`` tunes it; ``False`` omits the processor
    # entirely (byte-for-byte the pre-feature pipeline).
    working_sound: WorkingSoundConfig | bool = True

    # Play a short readiness chime the moment the pipeline can accept audio.
    readiness_chime: bool = True

    # Seconds of silence before the session ends itself. Defaults to ``None``,
    # which disables the idle timeout entirely: the session is never ended on
    # silence and runs until something else stops it. Set a float (e.g.
    # ``DEFAULT_IDLE_TIMEOUT_SECS``) to end the session after that much silence.
    # (The idle keep-alive heartbeat that protects long "thinking" turns from this
    # timeout is on whenever a timeout is set -- it's a correctness fix, not a
    # flourish -- and is skipped when the timeout is ``None`` since it exists only
    # to protect the timeout.)
    idle_timeout_secs: float | None = None

    # System prompt. ``None`` uses ``build_system_instruction`` (base rules +
    # active tools + date). Pass a string to fully override it, or a callable
    # ``(tool_instructions) -> str`` to compose your own from the active tools.
    system_instruction: str | Callable[[Sequence[str]], str] | None = None

    # Pipeline observers. ``None`` uses ``[ConversationLogObserver()]`` (logs
    # user/bot speech and tool calls). Pass ``[]`` for none, or your own list.
    observers: Sequence[BaseObserver] | None = None

    # VAD params for the user aggregator. ``None`` uses openlily's tuned defaults
    # (confidence=0.8, start_secs=0.3, min_volume=0.5), which reject short noise
    # bursts. Pass a ``VADParams`` to override.
    user_vad_params: VADParams | None = None

    # Worker pipeline params. ``None`` builds ``PipelineParams`` from
    # ``enable_metrics``; pass one to fully control it.
    pipeline_params: PipelineParams | None = None
    enable_metrics: bool = True


__all__ = [
    "DEFAULT_IDLE_TIMEOUT_SECS",
    "IDLE_KEEPALIVE_MAX_BUSY_SECS",
    "IDLE_KEEPALIVE_MAX_INTERVAL_SECS",
    "WORKING_SOUND_INITIAL_DELAY_SECS",
    "AgentConfig",
    "WorkingSoundConfig",
]
