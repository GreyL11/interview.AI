import re
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import log_metric
from app.intelligence.classifier import classify
from app.realtime.events import RejectionReason
from app.realtime.prompt_detector import (
    REASON_IMPERATIVE_TASK,
    REASON_PUNCTUATION,
    PromptMatch,
    is_acknowledgement,
    has_open_quantity,
    has_placeholder_object,
    REASON_CORRECTION,
    is_correction,
    is_scenario_without_request,
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
#: rejected utterance can't balloon the eventual LLM prompt.
_MAX_CONTEXT_CHARS = 220

_WORD = re.compile(r"[A-Za-z']+")

#: Words that are essentially never how a *complete* thought ends -- trailing
#: subordinators, conjunctions, prepositions, articles, copulas, and bare
#: interrogatives. A one-word follow-up ("Why?", "How?") never reaches this
#: check: it is below `question_min_words` and only ever accepted through the
#: follow-up bypass in `inspect`, which is exempted from this check entirely.
#: What lands here instead is always a *longer* sentence that happens to
#: trail off on one of these words ("Can you explain how", "Tell me about"),
#: which is incomplete regardless of length.
_DANGLING_WORDS = frozenset({
    "if", "that", "because", "so", "and", "but", "or", "nor",
    "the", "a", "an", "of", "to", "is", "are", "was", "were", "be", "been",
    "with", "for", "in", "on", "at", "as", "than", "while", "though",
    "although", "since", "unless", "until", "whether",
    "what", "how", "why", "when", "where", "which", "who", "whom", "about",
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


class Finality(StrEnum):
    """How finished the interviewer's request looks.

    A Whisper final only says a VAD segment closed. That is not the same
    thing as the interviewer having finished asking, which is not the same
    thing as having an exact question worth spending a provider call on:

        STT final  !=  interviewer turn final  !=  LLM question final

    This enum is the middle one. It decides whether the assembled text goes
    to the model now, or waits a bounded moment for the rest of the sentence.
    """

    #: Provably not a whole request yet -- a dangling conjunction, a bare
    #: trigger phrase, a premise with nothing asked for. Sending this can
    #: only ever be wrong, so it waits the longest.
    ACCUMULATING = "accumulating"
    #: A whole request by grammar, but worded the way interviewers word a
    #: request that still has a constraint coming ("find two numbers").
    #: Might be finished; holds briefly rather than betting either way.
    POTENTIALLY_COMPLETE = "potentially_complete"
    #: Send now. The overwhelming majority of questions land here.
    COMPLETE = "complete"


def _assess(match: PromptMatch, followup_eligible: bool) -> tuple[Finality, int]:
    """Decide finality from deterministic evidence, and how long to hold.

    Ordered most-certain first. Every branch names the evidence it acted on,
    and only the two non-COMPLETE branches ever wait -- there is deliberately
    no path here that delays an ordinary complete question.
    """
    # A follow-up ("Why?", "Can you elaborate?") is complete by construction:
    # it leans on the previous turn's context, and there is no continuation
    # coming that would make it more answerable.
    if followup_eligible:
        return Finality.COMPLETE, 0

    # Whisper put a "?" here itself, which is the strongest closure signal
    # available and outranks every heuristic below.
    #
    # It has to come first, not just before the semantic checks. The
    # last-word test cannot tell an object pronoun from a dangling
    # subordinator -- "Why did you choose that?" and "Can you explain that?"
    # both end on "that", and both are finished questions that a follow-up
    # depends on firing immediately. Whisper terminating the utterance is the
    # evidence that settles it. This does mean trusting a "?" on a fragment,
    # which is the right bet: the failure mode documented all over this
    # pipeline is Whisper giving "." to a real question, not "?" to a
    # half-spoken one.
    if match.reason == REASON_PUNCTUATION:
        return Finality.COMPLETE, 0

    # Syntactic incompleteness -- certain, and cheap to detect.
    if match.sparse or _looks_incomplete(match.prompt):
        return Finality.ACCUMULATING, settings.question_hold_incomplete_ms

    # A premise with no request in it yet ("Given an array of integers"), or a
    # request whose object is still a placeholder ("tell me about a time").
    # Neither is answerable, so both sit in the provably-incomplete tier.
    if is_scenario_without_request(match.prompt) or has_placeholder_object(match.prompt):
        return Finality.ACCUMULATING, settings.question_hold_incomplete_ms

    # A request naming a quantity but not the constraint on it ("find two
    # numbers"). Answerable in principle, just probably not what was meant,
    # so this gets the shorter hold.
    if has_open_quantity(match.prompt):
        return Finality.POTENTIALLY_COMPLETE, settings.question_stabilization_ms

    return Finality.COMPLETE, 0


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
    #: How finished this request looks -- see `Finality`.
    finality: Finality = Finality.COMPLETE
    #: How long the caller should hold this before asking, in ms. 0 for a
    #: complete question, which is the common case. Derived from `finality`
    #: rather than chosen by the caller, so the evidence and the wait stay
    #: in one place.
    hold_ms: int = 0
    #: Whisper itself terminated this utterance with "?". The strongest
    #: available evidence that the interviewer stopped asking, and the only
    #: thing that lets an accumulating turn fire without waiting out its
    #: continuation window -- which is what keeps "...in FastAPI?" from
    #: costing 2s of dead air at the end of an otherwise-assembled question.
    explicit_closure: bool = False

    @property
    def stable(self) -> bool:
        """True when this should go to the model immediately."""
        return self.finality is Finality.COMPLETE

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

    def close_turn(self) -> None:
        """Called once an answer has been delivered for the current turn.

        Keeps `_last_accepted_at` (follow-up recency still depends on it) but
        drops the accept *detail*, which is what selects the long
        imperative-task merge window. Without this, a coding question and the
        next, unrelated coding question asked a couple of seconds later would
        merge into one, because that window is 4s wide and nothing told the
        detector the first turn was over.
        """
        self._last_accept_detail = None

    def inspect(
        self,
        text: str,
        now: float | None = None,
        *,
        buffer_context: bool = True,
        accumulating: bool = False,
    ) -> Detection:
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
        #
        # `accumulating` overrides both: while a turn is still being held and
        # nothing has been sent, the caller is explicitly waiting for exactly
        # this continuation, so the window has to be at least as wide as the
        # hold it is waiting out. If it were narrower, the held fragment's
        # words would be dropped rather than assembled -- the fragment fires
        # or is cancelled, and the continuation arrives as a bare standalone
        # question that has lost everything said before it.
        if accumulating:
            merge_window_ms = settings.question_max_accumulation_ms
        elif self._last_accept_detail == REASON_IMPERATIVE_TASK:
            merge_window_ms = self._context_window_ms
        else:
            merge_window_ms = self._coalesce_ms

        combined = text
        supersedes = False
        if (
            not followup_eligible
            # An acknowledgement is not a continuation. Merging "That makes
            # sense." onto the imperative task before it rebuilds that task as
            # a fresh question and pays for a duplicate answer.
            and not is_acknowledgement(text)
            and self._last_accepted_at is not None
            # Speech-clock monotonicity. A final delivered out of order (a
            # short utterance overtaking a long one) would otherwise produce a
            # negative gap, which passes any <= window test and would splice
            # the two in the wrong order.
            and now >= self._last_accepted_at
            and (now - self._last_accepted_at) * 1000 <= merge_window_ms
            # A redelivered final must not be concatenated onto itself.
            and not self._is_repeat_of_tail(text)
        ):
            combined = f"{self._last_text} {text}".strip()
            supersedes = True

        context_prefix = self._context_prefix(now) if buffer_context else ""
        # A fragment only let through by the follow-up bypass is too thin to be
        # useful setup for something else -- remembering it would just be noise.
        may_remember = buffer_context and not followup_eligible

        match = extract_interview_prompt(combined)
        # Stray STT punctuation ("?? -- ...") ends in "?" and so reads as a
        # punctuation-terminated question, and merging it onto a real question
        # puts it last, where the sentence scanner prefers it. Requiring one
        # actual word keeps junk from hijacking a turn it was appended to.
        if match is not None and not _WORD.search(match.prompt):
            logger.info('question_rejected reason=no_words text="%s"', _preview(combined))
            return Detection(
                False, combined, reason=RejectionReason.NOT_A_QUESTION, detail="no_words"
            )
        if match is None:
            # A self-revision of a question that has *already been answered* is
            # itself answerable. "Design a rate limiter using Redis." then
            # "Actually, use Kafka instead." is not setup for some future
            # question -- the interviewer has changed the task and is waiting,
            # and the candidate is otherwise left reading an answer about the
            # wrong technology.
            #
            # Deliberately narrow, on the same footing as `followup_eligible`
            # above: it needs live interviewer speech and a question accepted
            # recently enough that this is plausibly a revision of it. A
            # correction with nothing to correct stays what it was -- setup
            # remembered for whatever comes next.
            if (
                buffer_context
                and is_correction(text)
                and self._recent_question(now)
                and _WORD.search(text)
            ):
                self._last_accepted_at = now
                self._last_text = text
                self._last_accept_detail = REASON_CORRECTION
                log_metric("question_correction_detected", chars=len(text))
                logger.info(
                    'question_detected reason=%s text="%s"',
                    REASON_CORRECTION, _preview(text),
                )
                # No context prefix and no merge: the task being revised lives
                # in conversation history, which `Relationship.CORRECTION`
                # already selects. Exact wording is preserved -- the revision
                # is the question, unrewritten.
                return Detection(
                    True,
                    text,
                    effective_text=text,
                    classification=classify(text),
                    detail=REASON_CORRECTION,
                    finality=Finality.COMPLETE,
                    hold_ms=0,
                )
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

        # Two kinds of premise, one field. `context_prefix` is setup from
        # *earlier utterances* the detector rejected and remembered; the
        # premise below is setup from earlier sentences of *this* utterance,
        # which the last-match sentence scan would otherwise discard. Both are
        # what the question is being asked against, so both belong in the text
        # the model sees and neither belongs in the coaching panel.
        #
        # Scoped to `text`, deliberately not to `combined`. A merge prepends
        # the previously accepted question, and the sentence scan is what
        # *drops* that -- a superseded question ("find the longest
        # substring... actually, merge intervals") must not come back as the
        # premise of the question that replaced it. Only sentences the
        # interviewer said in this same utterance qualify.
        own = match if combined == text else extract_interview_prompt(text)
        premise = own.premise if own is not None else ""
        setup = " ".join(part for part in (context_prefix, premise) if part)
        effective = f"{setup} {match.prompt}" if setup else match.prompt
        if setup:
            log_metric(
                "question_context_attached",
                chars=len(setup),
                from_buffer=len(context_prefix),
                from_utterance=len(match.premise),
            )
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

        finality, hold_ms = _assess(match, followup_eligible)
        if finality is not Finality.COMPLETE:
            # Interviewer speech is only logged when diagnostics are on; the
            # decision itself is always logged, so the hold is auditable
            # without putting the transcript on disk.
            log_metric(
                "question_not_final",
                finality=finality.value,
                hold_ms=hold_ms,
                sparse=match.sparse,
                chars=len(match.prompt),
                text=match.prompt if settings.question_detector_diagnostics else None,
            )

        logger.info(
            'question_detected reason=%s category=%s finality=%s hold_ms=%d text="%s"',
            detail, classification.category.value, finality.value, hold_ms,
            _preview(match.prompt),
        )
        return Detection(
            True,
            match.prompt,
            effective_text=effective,
            classification=classification,
            supersedes=supersedes,
            detail=detail,
            finality=finality,
            hold_ms=hold_ms,
            explicit_closure=match.reason == REASON_PUNCTUATION,
        )

    def _is_repeat_of_tail(self, text: str) -> bool:
        """Is this final a redelivery of what the turn already ends with?

        A duplicate final (the same utterance published twice) would
        otherwise merge onto itself -- "What is caching? What is caching?" --
        and be re-asked as a different question. Compared on normalised
        whitespace/case so punctuation drift between two passes of the same
        audio does not defeat it.
        """
        if not self._last_text:
            return False
        incoming = " ".join(text.lower().split())
        tail = " ".join(self._last_text.lower().split())
        return bool(incoming) and tail.endswith(incoming)

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
        # A self-revision replaces the premise it revises. Keeping both would
        # hand the model two contradictory constraints ("assume 1,000 QPS" and
        # "actually, assume 10,000 QPS") and let it pick, which is the one
        # outcome a correction is supposed to prevent.
        if is_correction(text) and self._context:
            log_metric("question_context_corrected", dropped=len(self._context))
            self._context.clear()
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
