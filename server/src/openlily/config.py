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
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

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

    # Optional tools to enable (what ``brains.yaml``'s ``tools:`` list holds):
    # built-in configurable tools by ``ToolName``, or ``register_tool``'d custom
    # tools by their plain-string id. The brain's own declared tools and the
    # always-on session tool are added automatically -- do not list them here.
    enabled_tools: Sequence[ToolName | str] = field(default_factory=tuple)

    # Soft "working" cue during the gap before the bot speaks. ``True`` uses the
    # defaults; a ``WorkingSoundConfig`` tunes it; ``False`` omits the processor
    # entirely (byte-for-byte the pre-feature pipeline).
    working_sound: WorkingSoundConfig | bool = True

    # Play a short readiness chime the moment the pipeline can accept audio.
    readiness_chime: bool = True

    # Spoken fallback when a user turn ends with no transcript and there is
    # nothing to recover (e.g. degraded audio trips the VAD but defeats STT on
    # the very first turn): the bot briefly says it couldn't make out what the
    # user said and asks them to repeat. ``True`` uses the default instruction;
    # a string overrides it (it's injected as a system message, so apps can
    # control tone); ``False`` disables the fallback. Cascade brains only: in
    # pipecat 1.4.0 the realtime services don't implement mid-session message
    # appends, so this is a documented no-op for realtime brains (see
    # openlily.turn_recovery).
    empty_turn_fallback: bool | str = True

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
    # ``(tool_guidance: str) -> str`` to compose your own: it receives the
    # pre-rendered ``<ToolGuidance>`` block for the session's active tools
    # (``""`` when no tools contribute guidance), so it can embed the block
    # unconditionally.
    system_instruction: str | Callable[[str], str] | None = None

    # Pipeline observers. ``None`` uses ``[ConversationLogObserver()]`` (logs
    # user/bot speech and tool calls). Pass ``[]`` for none, or your own list.
    observers: Sequence[BaseObserver] | None = None

    # VAD params for the user aggregator. ``None`` uses openlily's tuned defaults
    # (confidence=0.7, start_secs=0.3, min_volume=0.5): start_secs is slightly
    # above pipecat's default so short noise bursts don't open a speech segment,
    # and min_volume is *below* pipecat's because strict thresholds demonstrably
    # reject real speech from quiet mics (see the quiet-mic note in agent.py).
    # Pass a ``VADParams`` to override -- lower confidence/min_volume further if
    # the ConversationLogObserver warns that STT is transcribing during VAD
    # silence.
    user_vad_params: VADParams | None = None

    # User turn start/stop strategies for the user aggregator. ``None`` uses
    # openlily's defaults, which are pipecat's: turn starts on VAD *or* interim
    # transcription (fast barge-in), stops via the smart-turn analyzer. Pass a
    # ``UserTurnStrategies`` to take full control -- e.g.
    # ``UserTurnStrategies(start=[TranscriptionUserTurnStartStrategy()])`` to
    # gate turn starts on transcription so VAD false-triggers can't cancel an
    # in-flight response (openlily's previous default; costs seconds of barge-in
    # latency because STT only finalizes after an endpoint). Cascade brains
    # only: combining this with a realtime (speech-to-speech) brain raises,
    # because pipecat swaps in the service's external turn strategies only when
    # no custom strategies are passed, so an override would silently break turn
    # detection there.
    user_turn_strategies: UserTurnStrategies | None = None

    # Whether the user can barge in (interrupt the bot) while it's speaking.
    # ``True`` (default) is normal turn-taking: user speech during bot output
    # interrupts it. ``False`` disallows barge-in by muting the user's mic while
    # the bot speaks -- captured audio is dropped before it reaches STT (cascade
    # brains) or the realtime LLM (speech-to-speech brains), so neither can be
    # interrupted mid-utterance. The user is heard again as soon as the bot stops.
    allow_interruptions: bool = True

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
