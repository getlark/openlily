"""Always-on wake-word listener.

Ties a ``WakeWordEngine`` to an ``AudioSource`` and exposes a single blocking
call, ``wait_for_wake()``, that reads frames until the wake word is heard. It
deliberately owns the mic only for the duration of that call -- it opens the
source on entry and closes it before returning -- so a caller can hand the mic
off to something else (e.g. a voice session) and call ``wait_for_wake()`` again
afterward to resume listening.

This is synchronous/blocking by design; an async host can run it in a worker
thread (``asyncio.to_thread``).
"""

from __future__ import annotations

import time

from loguru import logger

from .audio import AudioSource, AudioSourceStalled
from .engine import WakeWordEngine

# Backoff between attempts to reopen a stalled/failed audio source. The device
# often needs a beat to recover (e.g. after the machine wakes from sleep), so we
# retry indefinitely - this is an always-on listener - with exponential backoff
# capped so it keeps probing reasonably often once the mic is usable again.
REOPEN_BACKOFF_INITIAL_SECS = 0.5
REOPEN_BACKOFF_MAX_SECS = 5.0


class WakeWordListener:
    """Listen on an ``AudioSource`` until the wake word fires."""

    def __init__(self, engine: WakeWordEngine, source: AudioSource) -> None:
        self._engine = engine
        self._source = source

    def wait_for_wake(self) -> str:
        """Block until a wake word is detected; return its label.

        Opens the audio source, reads frames until the engine reports a hit,
        then stops the source and resets the engine (so the audio leading up to
        the detection can't immediately re-fire on the next call).

        If the capture stream stalls (``AudioSourceStalled`` - e.g. the OS put
        the mic to sleep on a laptop-lid close and CoreAudio couldn't restart the
        audio unit on wake), the source is torn down and rebuilt rather than
        blocking forever on a dead stream, so listening self-heals.
        """
        self._open_with_retry()
        try:
            while True:
                try:
                    pcm = self._source.read()
                except AudioSourceStalled as e:
                    logger.warning(
                        f"Wake-word capture stalled ({e}); rebuilding the audio "
                        "source. This usually follows the machine sleeping/waking."
                    )
                    self._reopen_with_retry()
                    continue
                label = self._engine.process(pcm)
                if label is not None:
                    return label
        finally:
            try:
                self._source.stop()
            except Exception:
                logger.debug(
                    "Ignoring error while stopping wake-word audio source",
                    exc_info=True,
                )
            # Clear the rolling buffer so the just-heard wake word doesn't
            # linger and re-trigger when listening resumes.
            self._engine.reset()

    def _open_with_retry(self) -> None:
        """Open the source, retrying with backoff until it succeeds."""
        delay = REOPEN_BACKOFF_INITIAL_SECS
        while True:
            try:
                self._source.start()
                return
            except Exception:
                logger.exception(
                    f"Failed to open wake-word audio source; retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                delay = min(delay * 2, REOPEN_BACKOFF_MAX_SECS)

    def _reopen_with_retry(self) -> None:
        """Tear down a stalled source and reopen it, resetting the engine.

        The engine reset drops the buffered audio captured before the stall so it
        can't produce a spurious detection once frames flow again.
        """
        try:
            self._source.stop()
        except Exception:
            logger.debug(
                "Ignoring error while stopping stalled wake-word audio source",
                exc_info=True,
            )
        self._engine.reset()
        self._open_with_retry()
