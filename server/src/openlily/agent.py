"""Assemble an openlily pipeline + worker from an :class:`AgentConfig`.

This is the library's high-level factory. It is transport-agnostic: pass any
Pipecat ``BaseTransport`` (a local-audio transport, the dev runner's WebRTC
transport, or whatever a Pipecat Cloud deployment provides) and a config, and get
back a ready pipeline and worker. It reads nothing from the environment or disk --
all configuration comes through the ``AgentConfig``.

The pipeline shape mirrors the stock bot: cascade and realtime share everything
except whether STT/TTS are in the pipeline (a realtime speech-to-speech brain
does both internally). The assistant aggregator goes *after* ``transport.output()``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import BotSpeakingFrame, UserSpeakingFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

from openlily.brains import BrainSpec, get_brain
from openlily.config import (
    IDLE_KEEPALIVE_MAX_BUSY_SECS,
    IDLE_KEEPALIVE_MAX_INTERVAL_SECS,
    AgentConfig,
    WorkingSoundConfig,
)
from openlily.idle_keepalive import BotBusyFrame, IdleKeepaliveProcessor
from openlily.observers import ConversationLogObserver
from openlily.prompt import build_system_instruction
from openlily.sound import ReadinessChimeFrame, chime_pcm
from openlily.tools.bundle import (
    ToolBundle,
    register_tool_bundle,
    tools_schema_from_bundle,
)
from openlily.tools.runtime import setup_tools, warmup_tools
from openlily.turn_recovery import (
    DEFAULT_EMPTY_TURN_FALLBACK_INSTRUCTION,
    wire_empty_turn_recovery,
)
from openlily.working_sound import WorkingSoundProcessor


@dataclass
class Agent:
    """The assembled pieces for one session.

    ``tool_bundle`` is returned so the caller can run its cleanups (see
    ``close_tool_bundle``) when the session ends.
    """

    pipeline: Pipeline
    worker: PipelineWorker
    tool_bundle: ToolBundle


def resolve_brain(config: AgentConfig) -> BrainSpec:
    """Return the config's brain as a ``BrainSpec``, resolving a name/string.

    A ``BrainSpec`` is returned as-is; a ``BrainName``/string is looked up in the
    brain registry (built-ins plus any ``register_brain``'d brains).
    """
    if isinstance(config.brain, BrainSpec):
        return config.brain
    return get_brain(config.brain)


def _resolve_system_instruction(config: AgentConfig, tool_instructions: list[str]) -> str:
    """Compose the system prompt, honoring a string/callable override."""
    override = config.system_instruction
    if override is None:
        return build_system_instruction(tool_instructions)
    if callable(override):
        return override(tool_instructions)
    return override


def _idle_keepalive_interval_secs(idle_timeout_secs: float, max_interval_secs: float) -> float:
    """Heartbeat cadence kept safely below the idle timeout.

    A heartbeat must reset the idle timer before it expires, so we cap the
    interval at a third of the idle timeout (and at a small absolute ceiling),
    with a 1s floor for absurdly short timeouts.
    """
    return max(1.0, min(max_interval_secs, idle_timeout_secs / 3.0))


async def build_pipeline(
    transport: BaseTransport, config: AgentConfig
) -> tuple[Pipeline, ToolBundle]:
    """Assemble the pipeline for the configured brain.

    Returns the merged ``ToolBundle`` too, so the caller can run its cleanups
    (``close_tool_bundle``) when the session ends.
    """
    brain = resolve_brain(config)

    # Fail fast, before any tool/service setup: custom turn strategies are
    # unsupported with realtime brains. Pipecat's realtime mode swaps in the
    # service's external turn strategies (for services that emit their own turn
    # frames) only when no custom strategies are passed, so an override would
    # silently break turn detection.
    if brain.is_realtime and config.user_turn_strategies is not None:
        raise ValueError(
            "AgentConfig.user_turn_strategies is not supported with realtime "
            "(speech-to-speech) brains: pipecat swaps in the service's external "
            "turn strategies only when no custom strategies are passed, so an "
            "override would silently break turn detection. Leave it as None."
        )

    # Set up tools before building the LLM: the system prompt is composed from
    # the active tools' descriptions, and the LLM bakes in that prompt at
    # construction. The brain declares its tool ids; always-on and configured
    # tools are added by the tool runtime.
    tool_bundle = await setup_tools(brain.tools, config.enabled_tools)

    system_instruction = _resolve_system_instruction(config, tool_bundle.instructions)
    services = brain.build(system_instruction)

    # Now that the LLM exists, wire any LLM-dependent handlers (e.g. MCP tools).
    await register_tool_bundle(tool_bundle, services.llm)

    tools = tools_schema_from_bundle(tool_bundle)
    context = LLMContext(tools=tools) if tools else LLMContext()

    # start_secs is tuned slightly above pipecat's default (0.3 vs 0.2) so short
    # noise bursts don't open a speech segment and get shipped to STT, where
    # they hallucinate transcripts. confidence matches pipecat's default and
    # min_volume sits *below* it (0.5 vs 0.6): both conditions must pass for
    # speech to be detected, and stricter thresholds demonstrably classify real
    # speech from quiet mics as silence -- the turn then force-ends mid-sentence
    # on the stop watchdog while STT keeps transcribing (the
    # ConversationLogObserver warns when it sees this disagreement). The two
    # failure modes the old, stricter thresholds guarded against are now
    # mitigated structurally: hallucinated/ghost turns are healed by empty-turn
    # recovery, and noise bursts shorter than start_secs still never reach STT.
    # Quiet-mic setups that still trip the observer warning should lower
    # confidence/min_volume via ``user_vad_params``.
    vad_params = config.user_vad_params or VADParams(
        confidence=0.7,
        start_secs=0.3,
        min_volume=0.5,
    )
    # Disallow barge-in (config.allow_interruptions=False) by muting the user
    # while the bot speaks: the aggregator drops captured audio (InputAudioRawFrame
    # + turn/interruption frames) before it reaches STT (cascade) or the realtime
    # LLM (speech-to-speech), so neither can be interrupted mid-utterance. The
    # user is heard again the moment the bot stops. Empty list = normal barge-in.
    user_mute_strategies = [] if config.allow_interruptions else [AlwaysUserMuteStrategy()]

    # User turn strategies. ``None`` (the default) means pipecat's defaults:
    # turn starts on VAD *or* interim transcription, stops via the smart-turn
    # analyzer. The VAD start strategy is what makes barge-in fast -- a user
    # talking over the bot is heard the moment the VAD fires, instead of
    # waiting for STT to finalize a transcript (which some services only do
    # after an endpoint, i.e. seconds later). The cost is that a VAD
    # false-trigger (breath, cough, background noise) while the bot is
    # thinking can start a "ghost" turn and cancel the in-flight response;
    # empty-turn recovery (wired below) re-runs the LLM in exactly that case,
    # which is why openlily no longer gates cascade turn starts on
    # transcription. Callers who prefer the old trade-off can pass
    # ``UserTurnStrategies(start=[TranscriptionUserTurnStartStrategy()])``.
    # ``None`` also lets pipecat's realtime mode swap in external strategies
    # for realtime brains (custom strategies were rejected above).
    user_turn_strategies = config.user_turn_strategies

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
            user_mute_strategies=user_mute_strategies,
            user_turn_strategies=user_turn_strategies,
        ),
        # Realtime (speech-to-speech) services need different context-write
        # timing; the aggregator warns if this isn't set for them.
        realtime_service_mode=brain.is_realtime,
    )

    # Empty-turn recovery: if a user turn ends with no content while the
    # context still ends in an unanswered user message (an interruption --
    # e.g. a VAD false-trigger or a hallucinated empty transcript -- cancelled
    # the in-flight response), re-run the LLM so the question doesn't sit
    # unanswered forever. When there's nothing to recover either (VAD detected
    # speech but STT produced no transcript, e.g. degraded Bluetooth audio on
    # the first turn), the spoken fallback asks the user to repeat instead of
    # leaving the bot silent forever. Cascade brains only: in pipecat 1.4.0
    # the realtime services don't implement mid-session message appends. See
    # openlily.turn_recovery for the full story.
    empty_turn_fallback = config.empty_turn_fallback
    if brain.is_realtime or empty_turn_fallback is False:
        if brain.is_realtime and empty_turn_fallback is not False:
            logger.info(
                "Spoken empty-turn fallback disabled: realtime brains don't support "
                "mid-session context appends in pipecat 1.4.0"
            )
        fallback_instruction = None
    elif empty_turn_fallback is True:
        fallback_instruction = DEFAULT_EMPTY_TURN_FALLBACK_INSTRUCTION
    else:
        fallback_instruction = empty_turn_fallback
    wire_empty_turn_recovery(
        user_aggregator,
        assistant_aggregator,
        context,
        fallback_instruction=fallback_instruction,
    )

    # Idle keep-alive heartbeat so the bot's silent "thinking" time isn't counted
    # as idle and doesn't trip the session's idle timeout mid-turn. On whenever a
    # timeout is set (a correctness fix, not a flourish); BotBusyFrame is
    # registered in the worker's idle_timeout_frames (see build_worker). Skipped
    # when idle_timeout_secs is None, since there's no timeout to protect.
    idle_keepalive_processors = []
    if config.idle_timeout_secs is not None:
        idle_keepalive_processors = [
            IdleKeepaliveProcessor(
                interval_secs=_idle_keepalive_interval_secs(
                    config.idle_timeout_secs, IDLE_KEEPALIVE_MAX_INTERVAL_SECS
                ),
                max_busy_secs=IDLE_KEEPALIVE_MAX_BUSY_SECS,
            )
        ]

    # Soft "working" cue, sitting just before transport.output() so it sees the
    # turn/tool/TTS frames it gates on. When disabled it's simply omitted.
    working_sound_processors = []
    if config.working_sound is not False:
        working_sound = (
            config.working_sound
            if isinstance(config.working_sound, WorkingSoundConfig)
            else WorkingSoundConfig()
        )
        working_sound_processors = [
            WorkingSoundProcessor(initial_delay_secs=working_sound.initial_delay_secs)
        ]

    if brain.is_realtime:
        elements = [
            transport.input(),
            user_aggregator,
            services.llm,
            *idle_keepalive_processors,
            *working_sound_processors,
            transport.output(),
            assistant_aggregator,
        ]
    else:
        elements = [
            transport.input(),
            services.stt,
            user_aggregator,
            services.llm,
            services.tts,
            *idle_keepalive_processors,
            *working_sound_processors,
            transport.output(),
            assistant_aggregator,
        ]

    return Pipeline(elements), tool_bundle


def build_worker(pipeline: Pipeline, config: AgentConfig) -> PipelineWorker:
    """Wrap a pipeline in a ``PipelineWorker`` configured per ``AgentConfig``."""
    idle_timeout = config.idle_timeout_secs
    observers: list[BaseObserver] = (
        list(config.observers) if config.observers is not None else [ConversationLogObserver()]
    )
    pipeline_params = config.pipeline_params or PipelineParams(
        enable_metrics=config.enable_metrics,
        enable_usage_metrics=config.enable_metrics,
    )

    worker = PipelineWorker(
        pipeline,
        params=pipeline_params,
        # After this much silence, cancel the worker *and* the runner so the run
        # returns and the process can exit cleanly.
        idle_timeout_secs=idle_timeout,
        # Pipecat's idle detection resets only on these frames; we add
        # BotBusyFrame so the bot's silent thinking/tool time counts as activity.
        idle_timeout_frames=(BotSpeakingFrame, UserSpeakingFrame, BotBusyFrame),
        cancel_on_idle_timeout=True,
        cancel_runner_on_idle_timeout=True,
        observers=observers,
    )

    @worker.event_handler("on_idle_timeout")
    async def _on_idle_timeout(_worker):
        logger.info(
            f"Idle for {idle_timeout}s with no speech; ending session so the process can exit"
        )

    # Play a short "ding" the moment the pipeline is ready to accept audio, so the
    # user gets an audible acknowledgement that the bot is now listening. A
    # ReadinessChimeFrame doesn't count as bot speech, so it won't reset the idle
    # timer or trigger interruption logic.
    if config.readiness_chime:

        @worker.event_handler("on_pipeline_started")
        async def _on_pipeline_started(worker, _frame):
            pcm, sample_rate = chime_pcm()
            await worker.queue_frame(
                ReadinessChimeFrame(audio=pcm, sample_rate=sample_rate, num_channels=1)
            )
            logger.info("Pipeline ready; played readiness chime")

    return worker


async def create_agent(transport: BaseTransport, config: AgentConfig) -> Agent:
    """Build the pipeline and worker for a session in one call.

    Does not warm up models -- call :func:`warmup` first (once per process) if the
    brain or tools have slow first-run work. Run ``close_tool_bundle`` on the
    returned ``tool_bundle`` and ``shutdown_tools`` when the session/process ends.
    """
    pipeline, tool_bundle = await build_pipeline(transport, config)
    worker = build_worker(pipeline, config)
    return Agent(pipeline=pipeline, worker=worker, tool_bundle=tool_bundle)


async def _warmup_brain(brain: BrainSpec) -> None:
    """Eagerly download/load the brain's slow first-run models, once per process."""
    if brain.warmup is None:
        return
    logger.info("Warming up models (first run may download; this can take a while)...")
    await brain.warmup()
    logger.info("Warmup complete")


async def warmup(config: AgentConfig) -> None:
    """Eagerly load the brain's models and start selected MCP tools (in parallel).

    Optional but recommended once per process: it moves slow first-run work (model
    downloads, LLM cold starts, MCP server launches) to startup and fails fast on
    a known-broken setup rather than stalling mid-conversation.
    """
    brain = resolve_brain(config)
    await asyncio.gather(
        _warmup_brain(brain),
        warmup_tools(brain.tools, config.enabled_tools),
    )


__all__ = [
    "Agent",
    "build_pipeline",
    "build_worker",
    "create_agent",
    "resolve_brain",
    "warmup",
]
