"""A soft "working" audio cue for the awkward silence while the bot is busy.

When the user finishes speaking, the bot can take a while to respond - a slow
LLM reasoning step, or a tool call (web search, email, browser) waiting on the
network. During that gap there's nothing to hear, so the user can't tell whether
anything is happening. This processor fills the gap with an occasional soft, low
"blip" (see ``working_sound_pcm``), spread out over a long quiet gap like a
low-key processing indicator, and stops the instant real bot audio starts (or
the user barges in).

It lives just before ``transport.output()`` in the pipeline (see ``_build_pipeline``
in ``bot.py``), so it sees the frames it needs and can push audio downstream to
the transport:

- ARM (start looping) on ``UserStoppedSpeakingFrame`` (now waiting on the bot, so
  this covers reasoning latency) or ``FunctionCallsStartedFrame`` (a tool call
  began, which covers tool latency and re-arms mid-turn).
- DISARM (stop looping) on ``TTSStartedFrame`` (real bot audio is starting; both
  brains emit it), with ``BotStartedSpeakingFrame`` as a backstop, plus
  ``InterruptionFrame`` (barge-in) and ``CancelFrame``/``EndFrame`` (shutdown).

The cue is pushed as a plain ``OutputAudioRawFrame`` (not ``TTSAudioRawFrame``),
so - like the readiness chime - it plays without counting as bot speech, without
resetting the idle timer, and without triggering interruption logic.
"""

from __future__ import annotations

import asyncio

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from sound import working_sound_pcm

# 16-bit mono PCM: two bytes per sample.
_BYTES_PER_SAMPLE = 2

# Quiet gap between blip cycles. Long and spread out for a low-key feel. This is
# realized by sleeping (not silence in the buffer) so the cue never sits in
# front of the bot's real speech in the output's FIFO. Tune the blip timbre in
# sound.py and this spacing here.
WORKING_SOUND_GAP_SECS = 4.0


class WorkingSoundProcessor(FrameProcessor):
    """Plays an occasional soft blip while the bot is busy, until audio or a barge-in."""

    def __init__(self, *, initial_delay_secs: float):
        """Initialize the processor.

        Args:
            initial_delay_secs: Grace period after arming before the first blip
                plays. Fast turns emit ``TTSStartedFrame`` within this window, so
                nothing is ever heard on them.
        """
        super().__init__()
        self._initial_delay_secs = initial_delay_secs
        self._pcm, self._sample_rate = working_sound_pcm()
        # Spacing between blip onsets: how long the blip audio lasts plus the
        # quiet gap. Sleeping this (rather than queuing silence) keeps the output
        # FIFO clear during the gap so bot audio plays promptly when it arrives.
        sample_count = len(self._pcm) // _BYTES_PER_SAMPLE
        blip_secs = sample_count / self._sample_rate
        self._interval_secs = blip_secs + WORKING_SOUND_GAP_SECS
        self._loop_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Arm/disarm the cue based on pipeline frames, then forward the frame."""
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStoppedSpeakingFrame, FunctionCallsStartedFrame)):
            self._arm()
        elif isinstance(
            frame,
            (
                TTSStartedFrame,
                BotStartedSpeakingFrame,
                InterruptionFrame,
                CancelFrame,
                EndFrame,
            ),
        ):
            await self._disarm()

        await self.push_frame(frame, direction)

    def _arm(self) -> None:
        """Start the loop (with its grace delay) if it isn't already running.

        If a loop is already active (e.g. a tool call fires right after the user
        stopped speaking), leave it alone rather than resetting the grace delay,
        which would briefly pause an already-playing cue.
        """
        if self._loop_task is None:
            self._loop_task = self.create_task(self._loop())

    async def _disarm(self) -> None:
        """Stop the loop if it's running."""
        task, self._loop_task = self._loop_task, None
        if task is not None:
            await self.cancel_task(task)

    async def _loop(self) -> None:
        """Wait out the grace delay, then push one blip every interval."""
        await asyncio.sleep(self._initial_delay_secs)
        while True:
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=self._pcm,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                ),
                FrameDirection.DOWNSTREAM,
            )
            await asyncio.sleep(self._interval_secs)

    async def cleanup(self) -> None:
        """Cancel the loop task on teardown."""
        await self._disarm()
        await super().cleanup()
