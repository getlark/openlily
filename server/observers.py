"""Conversation + tool-result logging via a Pipecat observer.

An observer sees every frame as it's pushed between processors, in both
directions, so a single class can log the whole conversation regardless of the
brain in use:

- the realtime brain pushes the user's transcript *upstream* (from the
  speech-to-speech LLM service) and the bot's text *downstream*;
- the cascade brain pushes the user's transcript downstream from the STT
  service and the bot's text downstream from the LLM service.

This avoids the placement/direction gymnastics an in-pipeline ``FrameProcessor``
would need, and it's why the built-in ``TranscriptionLogObserver`` (which only
logs frames whose source is an ``STTService``) misses the realtime brain's user
transcripts.
"""

from __future__ import annotations

import json
import re

from loguru import logger
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService

# Tool-call args/results can be large (e.g. web_search payloads); cap what we
# dump so the logs stay a quick, readable summary rather than a wall of JSON.
_MAX_RESULT_CHARS = 500


def _truncate(text: str, limit: int = _MAX_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... ({len(text)} chars total)"


def _dump(value: object) -> str:
    """Render a tool's args/result as compact, truncated JSON for logging."""
    try:
        rendered = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(value)
    return _truncate(rendered)


# Matches a leading run of text ending in sentence punctuation that is *followed*
# by whitespace -- i.e. a sentence we know is complete because more text has
# already arrived after it. Requiring the trailing whitespace (rather than
# end-of-buffer) keeps us from flushing a half-finished sentence or splitting on
# things like "3.14" before the next delta lands.
_SENTENCE_RE = re.compile(r"\s*\S.*?[.!?]+(?=\s)", re.S)


class ConversationLogObserver(BaseObserver):
    """Log user speech, bot speech, and tool-call results to the console.

    Dedup strategy: ``on_push_frame`` fires once per processor hop, so a frame
    that traverses several processors would be logged repeatedly. We only act
    when ``data.source`` is the service that *emits* the frame (an ``STTService``
    or ``LLMService``), mirroring the built-in ``TranscriptionLogObserver``'s
    ``isinstance(src, STTService)`` trick. Tool frames are broadcast both ways,
    so for those we additionally require a downstream push.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Buffers the bot's text deltas for the current turn. We flush complete
        # sentences as they arrive (so the logs show progress while the bot is
        # still speaking), then flush whatever is left when the response ends.
        self._bot_buffer = ""

    def _flush_complete_sentences(self) -> None:
        """Log any complete sentences sitting at the front of the buffer."""
        while True:
            match = _SENTENCE_RE.match(self._bot_buffer)
            if not match:
                break
            sentence = match.group(0).strip()
            self._bot_buffer = self._bot_buffer[match.end() :]
            if sentence:
                logger.info(f"BOT: {sentence!r}")

    async def on_push_frame(self, data: FramePushed):
        src = data.source
        frame = data.frame
        direction = data.direction

        # User speech. Source is the STT service (cascade) or the realtime LLM
        # service (realtime), so accept either.
        if isinstance(frame, TranscriptionFrame) and isinstance(src, (STTService, LLMService)):
            text = (frame.text or "").strip()
            if text:
                logger.info(f"USER: {text!r}")
            return

        # Bot speech. Accumulate the LLM's text deltas and log each sentence as
        # soon as it completes, so a long answer shows up incrementally rather
        # than only after the whole turn finishes. Both frames originate at the
        # LLM service.
        if isinstance(src, LLMService):
            if isinstance(frame, LLMTextFrame):
                if frame.text:
                    self._bot_buffer += frame.text
                    self._flush_complete_sentences()
                return
            if isinstance(frame, LLMFullResponseEndFrame):
                # Flush the trailing sentence (which has no whitespace after it
                # to mark completion) plus anything else left in the buffer.
                remainder = self._bot_buffer.strip()
                self._bot_buffer = ""
                if remainder:
                    logger.info(f"BOT: {remainder!r}")
                return

        # Tool calls. These are broadcast upstream and downstream, so restrict
        # to the downstream push from the LLM service to log each one once.
        if isinstance(src, LLMService) and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, FunctionCallInProgressFrame):
                logger.info(f"TOOL CALL {frame.function_name}({_dump(frame.arguments)})")
                return
            if isinstance(frame, FunctionCallResultFrame):
                logger.info(f"TOOL RESULT {frame.function_name}: {_dump(frame.result)}")
                return
