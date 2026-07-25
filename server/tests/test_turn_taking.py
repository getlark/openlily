"""Tests for the turn-taking knobs and the VAD/STT disagreement warning.

Covers the ``AgentConfig.user_turn_strategies`` wiring (including the
realtime-brain rejection, which must fail fast before any service setup) and
``ConversationLogObserver``'s warning when STT transcribes speech during
sustained VAD silence -- the signature of VAD thresholds rejecting a quiet
mic's audio. No real services, audio, or network.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from openlily.agent import build_pipeline
from openlily.brains import BrainName, BrainSpec
from openlily.config import AgentConfig
from openlily.observers import _VAD_SILENCE_WARN_SECS, ConversationLogObserver

_NS = 1_000_000_000


def _pushed(frame: Frame, *, at_secs: float, source: object | None = None) -> FramePushed:
    return FramePushed(
        source=source if source is not None else MagicMock(),
        destination=MagicMock(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=int(at_secs * _NS),
    )


def _transcription(at_secs: float, *, source_spec: type = STTService) -> FramePushed:
    return _pushed(
        TranscriptionFrame("hello there", user_id="", timestamp=""),
        at_secs=at_secs,
        source=MagicMock(spec=source_spec),
    )


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    """Capture loguru WARNING-level messages emitted during the test."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    yield messages
    logger.remove(sink_id)


def _disagreements(messages: list[str]) -> list[str]:
    return [m for m in messages if "VAD silence" in m]


async def test_warns_when_stt_transcribes_during_sustained_vad_silence(
    warnings_log: list[str],
) -> None:
    observer = ConversationLogObserver()
    await observer.on_push_frame(_pushed(VADUserStartedSpeakingFrame(), at_secs=0.0))
    await observer.on_push_frame(_pushed(VADUserStoppedSpeakingFrame(), at_secs=1.0))
    # Transcript lands well past the disagreement threshold into VAD silence.
    await observer.on_push_frame(_transcription(1.0 + _VAD_SILENCE_WARN_SECS + 1.5))
    assert len(_disagreements(warnings_log)) == 1


async def test_no_warning_while_vad_reports_speech(warnings_log: list[str]) -> None:
    observer = ConversationLogObserver()
    await observer.on_push_frame(_pushed(VADUserStartedSpeakingFrame(), at_secs=0.0))
    await observer.on_push_frame(_transcription(10.0))
    assert _disagreements(warnings_log) == []


async def test_no_warning_for_normal_finalization_lag(warnings_log: list[str]) -> None:
    # STT routinely finalizes a little after the VAD stops; that's not a
    # disagreement.
    observer = ConversationLogObserver()
    await observer.on_push_frame(_pushed(VADUserStartedSpeakingFrame(), at_secs=0.0))
    await observer.on_push_frame(_pushed(VADUserStoppedSpeakingFrame(), at_secs=1.0))
    await observer.on_push_frame(_transcription(1.0 + _VAD_SILENCE_WARN_SECS - 0.5))
    assert _disagreements(warnings_log) == []


async def test_warns_when_vad_never_fires_at_all(warnings_log: list[str]) -> None:
    # Thresholds that reject the mic entirely produce no VAD frames; the
    # silence baseline is the first observed frame.
    observer = ConversationLogObserver()
    await observer.on_push_frame(_transcription(0.0))  # sets the baseline
    await observer.on_push_frame(_transcription(_VAD_SILENCE_WARN_SECS + 1.0))
    assert len(_disagreements(warnings_log)) == 1


async def test_warns_once_per_silence_stretch_and_rearms(warnings_log: list[str]) -> None:
    observer = ConversationLogObserver()
    await observer.on_push_frame(_pushed(VADUserStoppedSpeakingFrame(), at_secs=0.0))
    # Several transcript fragments in one silence stretch -> one warning.
    await observer.on_push_frame(_transcription(3.0))
    await observer.on_push_frame(_transcription(4.0))
    assert len(_disagreements(warnings_log)) == 1
    # A new speech/silence cycle re-arms the warning.
    await observer.on_push_frame(_pushed(VADUserStartedSpeakingFrame(), at_secs=5.0))
    await observer.on_push_frame(_pushed(VADUserStoppedSpeakingFrame(), at_secs=6.0))
    await observer.on_push_frame(_transcription(12.0))
    assert len(_disagreements(warnings_log)) == 2


async def test_no_warning_for_realtime_transcripts(warnings_log: list[str]) -> None:
    # Realtime services legitimately deliver transcripts long after VAD stops.
    observer = ConversationLogObserver()
    await observer.on_push_frame(_pushed(VADUserStoppedSpeakingFrame(), at_secs=0.0))
    await observer.on_push_frame(_transcription(10.0, source_spec=LLMService))
    assert _disagreements(warnings_log) == []


def test_user_turn_strategies_defaults_to_none() -> None:
    assert AgentConfig(brain="cartesia_openai").user_turn_strategies is None


async def test_custom_turn_strategies_rejected_for_realtime_brains() -> None:
    def _must_not_build(_system_instruction: str):
        raise AssertionError("brain.build must not run; the config check fails first")

    brain = BrainSpec(
        name=BrainName.OPENAI_REALTIME,
        is_realtime=True,
        build=_must_not_build,
    )
    config = AgentConfig(brain=brain, user_turn_strategies=UserTurnStrategies())
    with pytest.raises(ValueError, match="realtime"):
        await build_pipeline(MagicMock(), config)
