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

    # Tuned tighter than the pipecat defaults so short noise bursts don't open a
    # speech segment and get shipped to STT, where they hallucinate transcripts.
    vad_params = config.user_vad_params or VADParams(
        confidence=0.8,
        start_secs=0.3,
        min_volume=0.5,
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=vad_params),
        ),
        # Realtime (speech-to-speech) services need different context-write
        # timing; the aggregator warns if this isn't set for them.
        realtime_service_mode=brain.is_realtime,
    )

    # Idle keep-alive heartbeat so the bot's silent "thinking" time isn't counted
    # as idle and doesn't trip the session's idle timeout mid-turn. Always on (a
    # correctness fix, not a flourish); BotBusyFrame is registered in the worker's
    # idle_timeout_frames (see build_worker).
    idle_keepalive = IdleKeepaliveProcessor(
        interval_secs=_idle_keepalive_interval_secs(
            config.idle_timeout_secs, IDLE_KEEPALIVE_MAX_INTERVAL_SECS
        ),
        max_busy_secs=IDLE_KEEPALIVE_MAX_BUSY_SECS,
    )

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
            idle_keepalive,
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
            idle_keepalive,
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
