"""Tests for the generic agent extension points.

Covers the app-facing extension surface added to ``AgentConfig``/``Agent``:
custom pipeline processors (``PipelineExtensions``), the exposed
context/aggregator pair, the ``app_resources`` passthrough to the worker, and
the ``user_idle_timeout_secs`` forwarding. Uses a fake ``BrainSpec`` whose
services are mocks -- no real STT/LLM/TTS, audio, or network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from openlily.agent import build_pipeline, build_worker, create_agent
from openlily.brains import BrainName, BrainServices, BrainSpec
from openlily.config import AgentConfig, PipelineExtensions


class _NoopProcessor(FrameProcessor):
    """A do-nothing processor apps might inject; passes every frame through."""

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _cascade_brain(services: BrainServices) -> BrainSpec:
    return BrainSpec(
        name=BrainName.CARTESIA_OPENAI,
        is_realtime=False,
        build=lambda _system_instruction: services,
    )


def _realtime_brain(services: BrainServices) -> BrainSpec:
    return BrainSpec(
        name=BrainName.OPENAI_REALTIME,
        is_realtime=True,
        build=lambda _system_instruction: services,
    )


def _cascade_services() -> BrainServices:
    return BrainServices(
        llm=MagicMock(spec=LLMService),
        stt=MagicMock(spec=STTService),
        tts=MagicMock(spec=TTSService),
    )


async def test_custom_processors_land_in_cascade_slots() -> None:
    after_llm = _NoopProcessor()
    before_output = _NoopProcessor()
    services = _cascade_services()
    transport = MagicMock()
    config = AgentConfig(
        brain=_cascade_brain(services),
        custom_processors=PipelineExtensions(
            after_llm=(after_llm,), before_output=(before_output,)
        ),
    )

    built = await build_pipeline(transport, config)

    processors = built.pipeline.processors
    llm_index = processors.index(services.llm)
    assert processors[llm_index + 1] is after_llm
    assert processors[llm_index + 2] is services.tts

    output_index = processors.index(transport.output())
    assert processors[output_index - 1] is before_output


async def test_custom_processors_land_in_realtime_slots() -> None:
    after_llm = _NoopProcessor()
    before_output = _NoopProcessor()
    services = BrainServices(llm=MagicMock(spec=LLMService))
    transport = MagicMock()
    config = AgentConfig(
        brain=_realtime_brain(services),
        custom_processors=PipelineExtensions(
            after_llm=(after_llm,), before_output=(before_output,)
        ),
        # Arm the idle keepalive so after_llm placement (before it) is exercised.
        idle_timeout_secs=30.0,
    )

    built = await build_pipeline(transport, config)

    processors = built.pipeline.processors
    llm_index = processors.index(services.llm)
    assert processors[llm_index + 1] is after_llm

    output_index = processors.index(transport.output())
    assert processors[output_index - 1] is before_output


async def test_defaults_insert_no_extra_processors() -> None:
    services = _cascade_services()
    transport = MagicMock()
    config = AgentConfig(brain=_cascade_brain(services))

    built = await build_pipeline(transport, config)

    processors = built.pipeline.processors
    llm_index = processors.index(services.llm)
    # Today's cascade shape: TTS directly follows the LLM.
    assert processors[llm_index + 1] is services.tts
    assert not any(isinstance(p, _NoopProcessor) for p in processors)


async def test_built_pipeline_exposes_wired_context_and_aggregators() -> None:
    services = _cascade_services()
    config = AgentConfig(brain=_cascade_brain(services))

    built = await build_pipeline(MagicMock(), config)

    processors = built.pipeline.processors
    assert built.user_aggregator in processors
    assert built.assistant_aggregator in processors
    assert built.user_aggregator.context is built.context
    assert built.assistant_aggregator.context is built.context


async def test_agent_carries_context_aggregators_and_app_resources() -> None:
    services = _cascade_services()
    resources = object()
    config = AgentConfig(brain=_cascade_brain(services), app_resources=resources)

    agent = await create_agent(MagicMock(), config)

    assert agent.context is agent.user_aggregator.context
    assert agent.user_aggregator in agent.pipeline.processors
    assert agent.assistant_aggregator in agent.pipeline.processors
    assert agent.worker.app_resources is resources


async def test_app_resources_default_is_none() -> None:
    services = _cascade_services()
    config = AgentConfig(brain=_cascade_brain(services))

    built = await build_pipeline(MagicMock(), config)
    worker = build_worker(built.pipeline, config)

    assert worker.app_resources is None


async def test_user_idle_timeout_forwarded_to_aggregator() -> None:
    services = _cascade_services()
    config = AgentConfig(brain=_cascade_brain(services), user_idle_timeout_secs=7.5)

    built = await build_pipeline(MagicMock(), config)

    assert built.user_aggregator._params.user_idle_timeout == 7.5


async def test_user_idle_timeout_defaults_to_disabled() -> None:
    services = _cascade_services()
    config = AgentConfig(brain=_cascade_brain(services))

    built = await build_pipeline(MagicMock(), config)

    assert built.user_aggregator._params.user_idle_timeout == 0
