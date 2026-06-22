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

from .audio import AudioSource
from .engine import WakeWordEngine


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
        """
        self._source.start()
        try:
            while True:
                pcm = self._source.read()
                label = self._engine.process(pcm)
                if label is not None:
                    return label
        finally:
            self._source.stop()
            # Clear the rolling buffer so the just-heard wake word doesn't
            # linger and re-trigger when listening resumes.
            self._engine.reset()
