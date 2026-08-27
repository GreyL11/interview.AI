import time
from dataclasses import dataclass

from app.core.config import settings
from app.intelligence.classifier import classify
from app.realtime.events import RejectionReason
from app.schemas.classification import Category, Classification


@dataclass
class Detection:
    accepted: bool
    text: str
    classification: Classification | None = None
    reason: RejectionReason | None = None
    supersedes: bool = False


class QuestionDetector:
    """Decides which finalised utterances are worth answering.

    The debounce is structural rather than timer-based: only final transcripts
    ever reach this class, so no amount of partial-transcript churn can trigger
    an LLM call. What is left is filtering out the short acknowledgements
    ("right", "mm-hmm") that VAD will happily finalise as utterances.
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
            return Detection(False, text, reason=RejectionReason.TOO_SHORT)

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

        classification = classify(combined)

        if not classification.is_question or classification.category == Category.UNKNOWN:
            return Detection(False, combined, classification, RejectionReason.NOT_A_QUESTION)

        if classification.confidence < self._min_confidence:
            return Detection(False, combined, classification, RejectionReason.LOW_CONFIDENCE)

        self._last_accepted_at = now
        self._last_text = combined
        return Detection(True, combined, classification, supersedes=supersedes)
