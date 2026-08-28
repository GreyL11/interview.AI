import time
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
    ) -> None:
        self._min_words = min_words if min_words is not None else settings.question_min_words
        self._min_confidence = (
            min_confidence if min_confidence is not None else settings.question_min_confidence
        )
        self._coalesce_ms = (
            coalesce_ms if coalesce_ms is not None else settings.question_coalesce_ms
        )
        self._last_accepted_at: float | None = None
        self._last_text: str = ""

    def reset(self) -> None:
        self._last_accepted_at = None
        self._last_text = ""

    def inspect(self, text: str, now: float | None = None) -> Detection:
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

        match = extract_interview_prompt(combined)
        if match is None:
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

        self._last_accepted_at = now
        # Coalescing works on raw speech; the cleaned prompt is what coaching sees.
        self._last_text = combined

        logger.info(
            'question_detected reason=%s category=%s text="%s"',
            match.reason, classification.category.value, _preview(match.prompt),
        )
        return Detection(
            True,
            match.prompt,
            classification,
            supersedes=supersedes,
            detail=match.reason,
        )
