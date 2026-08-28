import re
import time
from collections import deque
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import log_metric
from app.intelligence.classifier import classify
from app.realtime.events import RejectionReason
from app.realtime.prompt_detector import (
    REASON_IMPERATIVE_TASK,
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

_WORD = re.compile(r"[A-Za-z']+")

#: Words that are essentially never how a *complete* thought ends -- trailing
#: subordinators, conjunctions, prepositions, articles, copulas. Deliberately
#: excludes bare interrogatives (what/how/why/when alone) since those are
#: legitimate one-word follow-ups ("Why?"); the follow-up bypass path is
#: exempted from this check entirely for that reason (see `inspect`).
_DANGLING_WORDS = frozenset({
    "if", "that", "because", "so", "and", "but", "or", "nor",
    "the", "a", "an", "of", "to", "is", "are", "was", "were", "be", "been",
    "with", "for", "in", "on", "at", "as", "than", "while", "though",
    "although", "since", "unless", "until", "whether", "when", "where", "which", "who", "whom",
})


def _looks_incomplete(text: str) -> bool:
    """Deterministic "did this trail off mid-clause" check.

    Whisper's punctuation is unreliable (routinely a "." on a genuine
    question), so this looks at the last *word*, not trailing punctuation:
    "...what happens when" ends on a subordinator with nothing after it,
    which no complete interview question does.
    """
    words = _WORD.findall(text)
    return bool(words) and words[-1].lower() in _DANGLING_WORDS


@dataclass
class Detection:
    accepted: bool
    #: Clean, display-ready text: what the interviewer actually asked, or the
    #: raw rejected fragment. This is what the UI and session history show.
    text: str
    #: What should actually be sent to the LLM. Equal to `text` unless
    #: preceding setup context was attached -- kept separate so a verbose,
    #: context-prefixed prompt never has to leak into the coaching panel.
    effective_text: str = ""
    classification: Classification | None = None
    reason: RejectionReason | None = None
    supersedes: bool = False
    #: Which detection layer fired, for logs and diagnosis. Finer-grained than
    #: the wire-level RejectionReason, which the UI depends on.
    detail: str | None = None
    #: False means the question looks mid-clause and callers should give a
    #: brief stabilization window for a continuation before asking it.
    stable: bool = True

    def __post_init__(self) -> None:
        if not self.effective_text:
            self.effective_text = self.text


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
        followup_window_ms: int | None = None,
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
        self._followup_window_ms = (
            followup_window_ms if followup_window_ms is not None
            else settings.question_followup_window_ms
        )
        self._last_accepted_at: float | None = None
        self._last_text: str = ""
        #: Which layer accepted the last question -- see the merge-window
        #: comment in `inspect` for why this matters.
        self._last_accept_detail: str | None = None
        #: Interviewer utterances that were not themselves a question, kept
        #: around briefly in case the *next* utterance is the question they
        #: were setting up -- see `_context_prefix` / `_remember_as_context`.
        self._context: deque[tuple[float, str]] = deque(maxlen=_MAX_CONTEXT_SEGMENTS)

    def reset(self) -> None:
        self._last_accepted_at = None
        self._last_text = ""
        self._last_accept_detail = None
        self._context.clear()

    def inspect(self, text: str, now: float | None = None, *, buffer_context: bool = True) -> Detection:
        """Decide whether `text` is worth answering.

        `buffer_context` gates two session-scoped behaviours: a buffer of
        recent *rejected* interviewer utterances that get prepended once a
        question actually lands (so "By using this study, just write a
        character count program." followed by "How many times each character
        is repeated?" reaches coaching as one question instead of the bare
        fragment), and a narrow bypass that lets a short follow-up ("Why?",
        "How?") through when a question was accepted recently enough that the
        candidate is plausibly still reacting to its answer. Callers must pass
        `buffer_context=False` for anything that isn't live interviewer speech
        (a typed question, the candidate's own mic) so it can neither draw on
        nor pollute either mechanism.
        """
        now = now if now is not None else time.monotonic()
        text = text.strip()

        short = len(text.split()) < self._min_words
        followup_eligible = short and buffer_context and self._recent_question(now)
        if short and not followup_eligible:
            logger.info('question_rejected reason=too_short text="%s"', _preview(text))
            return Detection(False, text, reason=RejectionReason.TOO_SHORT, detail="too_short")

        # A follow-on clause spoken right after a question is a correction, not a
        # new question: "How would you scale this? ... Actually, assume 10k QPS."
        # A follow-up-bypassed fragment is excluded: merging "Okay." onto the
        # previous question's own "?" sentence would accept pure filler that
        # the length gate exists to keep out -- a follow-up must stand on its
        # own and lean on conversation history instead, never on a merge.
        #
        # The merge window itself adapts to *why* the last question was
        # accepted. A punctuation/interrogative accept ("How would you scale
        # this?") is a complete sentence Whisper terminated on its own, so a
        # short window is right -- it is a correction if anything follows.
        # An imperative-task accept with no terminal "?" ("Given an array of
        # integers, I want you to find two numbers") is exactly how a coding
        # problem's setup clause looks *before* its closing condition arrives
        # ("...whose sum equals a target value"), and that closing clause is
        # its own VAD-bounded utterance -- a full speech+silence-close cycle
        # away, comfortably longer than the correction window. Reusing the
        # (already longer) setup-context window here instead of inventing a
        # third number covers that realistic gap.
        merge_window_ms = (
            self._context_window_ms
            if self._last_accept_detail == REASON_IMPERATIVE_TASK
            else self._coalesce_ms
        )
        combined = text
        supersedes = False
        if (
            not followup_eligible
            and self._last_accepted_at is not None
            and (now - self._last_accepted_at) * 1000 <= merge_window_ms
        ):
            combined = f"{self._last_text} {text}".strip()
            supersedes = True

        context_prefix = self._context_prefix(now) if buffer_context else ""
        # A fragment only let through by the follow-up bypass is too thin to be
        # useful setup for something else -- remembering it would just be noise.
        may_remember = buffer_context and not followup_eligible

        match = extract_interview_prompt(combined)
        if match is None:
            if may_remember:
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
            if may_remember:
                self._remember_as_context(text, now)
            logger.info(
                'question_rejected reason=low_confidence confidence=%.2f text="%s"',
                classification.confidence, _preview(match.prompt),
            )
            return Detection(
                False,
                match.prompt,
                classification=classification,
                reason=RejectionReason.LOW_CONFIDENCE,
                detail="low_confidence",
            )

        effective = f"{context_prefix} {match.prompt}" if context_prefix else match.prompt
        if context_prefix:
            log_metric("question_context_attached", chars=len(context_prefix))
        if buffer_context:
            # Consumed: a later, unrelated question must not inherit this setup.
            self._context.clear()

        self._last_accepted_at = now
        # Coalescing works on raw speech; the cleaned prompt is what coaching sees.
        self._last_text = combined
        self._last_accept_detail = match.reason

        detail = match.reason
        if followup_eligible:
            detail = "follow_up"
            log_metric("question_follow_up_detected", text=match.prompt)

        # A follow-up is inherently short and already passed the sentence
        # classifier on its own merits ("Why?" ends in a bare interrogative
        # that would otherwise look "dangling") -- the completeness check
        # does not apply to it.
        stable = followup_eligible or not _looks_incomplete(match.prompt)
        if not stable:
            log_metric("question_looks_incomplete", text=match.prompt)

        logger.info(
            'question_detected reason=%s category=%s stable=%s text="%s"',
            detail, classification.category.value, stable, _preview(match.prompt),
        )
        return Detection(
            True,
            match.prompt,
            effective_text=effective,
            classification=classification,
            supersedes=supersedes,
            detail=detail,
            stable=stable,
        )

    def _recent_question(self, now: float) -> bool:
        """True if a question was accepted recently enough that a short,
        otherwise-too-short fragment can plausibly be a follow-up to it."""
        return (
            self._last_accepted_at is not None
            and (now - self._last_accepted_at) * 1000 <= self._followup_window_ms
        )

    def _remember_as_context(self, text: str, now: float) -> None:
        """A rejected utterance may still be the setup for the next question."""
        # Garbage STT (stray punctuation, broken fragments) isn't useful setup
        # for anything; skip it rather than let it pollute a later question.
        if sum(1 for c in text if c.isalpha()) < 3:
            return
        # Capped per segment, not just at prefix-build time, so one long
        # rambling utterance can't sit in memory uncapped for the rest of the
        # session if nothing ever consumes it.
        self._context.append((now, text[:_MAX_CONTEXT_CHARS]))

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
