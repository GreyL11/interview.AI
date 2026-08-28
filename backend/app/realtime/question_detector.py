import time
from collections import deque
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.intelligence.classifier import classify
from app.realtime.events import RejectionReason
from app.realtime.prompt_detector import (
    REASON_NO_PATTERN,
    extract_interview_prompt,
)
from app.schemas.classification import Classification

logger = get_logger(__name__)

#: How many rejected interviewer fragments to keep waiting for the question
#: they turn out to be setup for. Small on purpose -- this bridges a thought
#: split across a couple of utterances, not a transcript history.
_MAX_CONTEXT_SEGMENTS = 3
#: Hard cap on how much buffered context can be prepended, so one rambling
#: rejected utterance can't balloon the eventual Gemini prompt.
_MAX_CONTEXT_CHARS = 220


@dataclass
class Detection:
    accepted: bool
    text: str
    classification: Classification | None = None
    reason: RejectionReason | None = None
    supersedes: bool = False
    #: Which detection layer fired, for logs and diagnosis. Finer-grained than
    #: the wire-level RejectionReason, which the UI depends on.
    detail: str | None = None


def _preview(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}…"


class QuestionDetector:
    """Decides which finalised utterances are worth answering.

    The debounce is structural rather than timer-based: only final transcripts
    ever reach this class, so no amount of partial-transcript churn can trigger
    an LLM call.

    Whether something *is* a question is delegated to prompt_detector, which
    reads the utterance sentence by sentence. The phase 1 classifier is then
    asked only what kind of question it is — it was written for typed input and
    is not reliable at spotting a question buried after conversational filler.
    """

    def __init__(
        self,
        min_words: int | None = None,
        min_confidence: float | None = None,
        coalesce_ms: int | None = None,
        context_window_ms: int | None = None,
    ) -> None:
        self._min_words = min_words if min_words is not None else settings.question_min_words
        self._min_confidence = (
            min_confidence if min_confidence is not None else settings.question_min_confidence
        )
        self._coalesce_ms = (
            coalesce_ms if coalesce_ms is not None else settings.question_coalesce_ms
        )
        self._context_window_ms = (
            context_window_ms if context_window_ms is not None
            else settings.question_context_window_ms
        )
        self._last_accepted_at: float | None = None
        self._last_text: str = ""
        #: Interviewer utterances that were not themselves a question, kept
        #: around briefly in case the *next* utterance is the question they
        #: were setting up -- see `_context_prefix` / `_remember_as_context`.
        self._context: deque[tuple[float, str]] = deque(maxlen=_MAX_CONTEXT_SEGMENTS)

    def reset(self) -> None:
        self._last_accepted_at = None
        self._last_text = ""
        self._context.clear()

    def inspect(self, text: str, now: float | None = None, *, buffer_context: bool = True) -> Detection:
        """Decide whether `text` is worth answering.

        `buffer_context` gates a per-session buffer of recent *rejected*
        interviewer utterances that get prepended once a question actually
        lands, so "By using this study, just write a character count
        program." followed by "How many times each character is repeated?"
        reaches coaching as one question instead of the bare fragment. Callers
        must pass `buffer_context=False` for anything that isn't live
        interviewer speech (a typed question, the candidate's own mic) so it
        can neither draw on nor pollute that buffer.
        """
        now = now if now is not None else time.monotonic()
        text = text.strip()

        if len(text.split()) < self._min_words:
            logger.info('question_rejected reason=too_short text="%s"', _preview(text))
            return Detection(False, text, reason=RejectionReason.TOO_SHORT, detail="too_short")

        # A follow-on clause spoken right after a question is a correction, not a
        # new question: "How would you scale this? ... Actually, assume 10k QPS."
        combined = text
        supersedes = False
        if (
            self._last_accepted_at is not None
            and (now - self._last_accepted_at) * 1000 <= self._coalesce_ms
        ):
            combined = f"{self._last_text} {text}".strip()
            supersedes = True

        context_prefix = self._context_prefix(now) if buffer_context else ""

        match = extract_interview_prompt(combined)
        if match is None:
            if buffer_context:
                self._remember_as_context(text, now)
            logger.info(
                'question_rejected reason=%s text="%s"', REASON_NO_PATTERN, _preview(combined)
            )
            return Detection(
                False,
                combined,
                reason=RejectionReason.NOT_A_QUESTION,
                detail=REASON_NO_PATTERN,
            )

        # The extracted prompt always ends in "?", so the classifier sees a
        # well-formed question and answers the routing question rather than the
        # is-this-a-question question.
        classification = classify(match.prompt)

        if classification.confidence < self._min_confidence:
            if buffer_context:
                self._remember_as_context(text, now)
            logger.info(
                'question_rejected reason=low_confidence confidence=%.2f text="%s"',
                classification.confidence, _preview(match.prompt),
            )
            return Detection(
                False,
                match.prompt,
                classification,
                RejectionReason.LOW_CONFIDENCE,
                detail="low_confidence",
            )

        prompt = f"{context_prefix} {match.prompt}" if context_prefix else match.prompt
        if buffer_context:
            # Consumed: a later, unrelated question must not inherit this setup.
            self._context.clear()

        self._last_accepted_at = now
        # Coalescing works on raw speech; the cleaned prompt is what coaching sees.
        self._last_text = combined

        logger.info(
            'question_detected reason=%s category=%s text="%s"',
            match.reason, classification.category.value, _preview(prompt),
        )
        return Detection(
            True,
            prompt,
            classification,
            supersedes=supersedes,
            detail=match.reason,
        )

    def _remember_as_context(self, text: str, now: float) -> None:
        """A rejected utterance may still be the setup for the next question."""
        self._context.append((now, text))

    def _context_prefix(self, now: float) -> str:
        """Recent, still-relevant rejected utterances, oldest first, bounded in
        both age and length. Expired entries are dropped so they cannot
        resurface for a later, unrelated question."""
        while self._context and (now - self._context[0][0]) * 1000 > self._context_window_ms:
            self._context.popleft()

        if not self._context:
            return ""

        segments: list[str] = []
        total = 0
        for _, segment in reversed(self._context):
            if total + len(segment) > _MAX_CONTEXT_CHARS:
                break
            segments.append(segment)
            total += len(segment)
        segments.reverse()
        return " ".join(segments)
