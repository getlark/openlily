"""Keep the bot's *thinking* time from being mistaken for an idle session.

Pipecat's idle detection (``PipelineWorker(idle_timeout_secs=...)``) resets its
timer only when one of ``idle_timeout_frames`` is pushed - by default
``BotSpeakingFrame``/``UserSpeakingFrame``, i.e. "someone is speaking." While the
bot is *busy but silent* - the LLM's time-to-first-token, or a tool call waiting
on the network - neither frame flows, so that gap counts as idle. A long enough
think then trips the timeout and cancels the session mid-turn, which is a bad UX.

This processor treats "the bot is busy" as activity. It arms on the frames that
mark the start of a busy gap (``UserStoppedSpeakingFrame``, or
``FunctionCallsStartedFrame`` so tool latency re-arms mid-turn) and disarms the
moment real bot audio begins (``TTSStartedFrame``, with
``BotStartedSpeakingFrame`` as a backstop) or the turn is torn down
(``InterruptionFrame``/``CancelFrame``/``EndFrame``). Once bot audio is flowing,
``BotSpeakingFrame`` keeps the idle timer alive on its own; after the bot stops,
the normal idle timer runs - which is exactly "measure idle from when the bot
finished."

While armed it pushes a :class:`BotBusyFrame` every ``interval_secs``. Add
``BotBusyFrame`` to the worker's ``idle_timeout_frames`` and keep
``interval_secs`` comfortably below ``idle_timeout_secs`` so a heartbeat always
lands before the timer expires.

It mirrors ``WorkingSoundProcessor``'s arm/disarm lifecycle but is deliberately
independent of it: the working-sound cue is an optional flourish that can be
turned off, whereas this idle keep-alive is a correctness fix that should always
run.
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
    SystemFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotBusyFrame(SystemFrame):
    """Heartbeat pushed while the bot is busy (thinking or running a tool).

    Listed in the worker's ``idle_timeout_frames`` so each one resets the idle
    timer, keeping a long busy gap from being mistaken for an idle session.
    """

    pass


class IdleKeepaliveProcessor(FrameProcessor):
    """Pushes a :class:`BotBusyFrame` on a fixed cadence while the bot is busy."""

    def __init__(self, *, interval_secs: float):
        """Initialize the processor.

        Args:
            interval_secs: Seconds between heartbeats while armed. Keep this
                comfortably below the worker's ``idle_timeout_secs`` so a
                heartbeat always lands before the idle timer expires.
        """
        super().__init__()
        self._interval_secs = interval_secs
        self._loop_task: asyncio.Task | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Arm/disarm the heartbeat based on pipeline frames, then forward the frame."""
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
        """Start the heartbeat loop if it isn't already running.

        Leave an already-running loop alone (e.g. a tool call fires right after
        the user stops) so the cadence isn't reset.
        """
        if self._loop_task is None:
            self._loop_task = self.create_task(self._loop())

    async def _disarm(self) -> None:
        """Stop the heartbeat loop if it's running."""
        task, self._loop_task = self._loop_task, None
        if task is not None:
            await self.cancel_task(task)

    async def _loop(self) -> None:
        """Push one heartbeat every interval until disarmed."""
        while True:
            await asyncio.sleep(self._interval_secs)
            await self.push_frame(BotBusyFrame(), FrameDirection.DOWNSTREAM)

    async def cleanup(self) -> None:
        """Cancel the heartbeat loop on teardown."""
        await self._disarm()
        await super().cleanup()
