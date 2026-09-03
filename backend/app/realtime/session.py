import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger
from app.core.metrics import LatencyTrace, elapsed_ms, log_metric
from app.documents.schemas import PERSONAL_KNOWLEDGE_TYPES, RetrievedChunk
from app.intelligence.answer_validator import AnswerValidationError, validate
from app.intelligence.router import Route, route_for
from app.llm.base import LLMClient, LLMError
from app.llm.prompts import build_prompt
from app.llm.streaming import extract_partial_summary, parse_answer_payload
from app.memory.base import SessionMemory
from app.realtime.attachments import (
    Attachment,
    AttachmentBuffer,
    AttachmentError,
    AttachmentKind,
    build_attachment,
)
from app.realtime.attachments import render as render_attachments
from app.realtime.attachments import summarise as summarise_attachments
from app.realtime.events import CancelReason, Event, EventType, event
from app.realtime.question_detector import Detection, QuestionDetector
from app.realtime.question_understanding import (
    QuestionUnderstander,
    Understanding,
    UnderstandingSource,
    select_context,
)
from app.retrieval.base import Retriever
from app.schemas.answer import Answer
from app.schemas.classification import Classification
from app.sessions.schemas import (
    RetrievalHit,
    TranscriptEntry,
    TranscriptSource,
    Turn,
    TurnStatus,
)
from app.storage.session_repository import SessionRepository

logger = get_logger(__name__)

Emitter = Callable[[Event], Awaitable[None]]

#: How often `_await_speech_end` re-checks whether the interviewer stopped.
#: Small against every accumulation budget, so it adds no meaningful latency.
_SPEECH_POLL_SECONDS = 0.2


@dataclass(frozen=True)
class _LiveQuestion:
    """The in-flight turn, enough of it to re-ask verbatim.

    `detail` is the deterministic detector's own reading (follow_up,
    imperative_task, punctuation, ...). Re-asking without it is what made a
    late paste degrade a follow-up into a new question whenever the
    classifier fell back.
    """

    question: str
    effective_question: str
    classification: Classification
    detail: str | None = None


class LiveSession:
    """Owns one practice session: transcript in, coaching answers out.

    Exactly one answer may be in flight. A new question cancels the previous
    task and every event carries its turn_id, so a late chunk from a superseded
    answer can be dropped on both sides rather than racing into the UI.
    """

    def __init__(
        self,
        session_id: str,
        sessions: SessionRepository,
        retriever: Retriever,
        llm: LLMClient,
        memory: SessionMemory,
        detector: QuestionDetector | None = None,
        summarizer=None,
        understander: QuestionUnderstander | None = None,
    ) -> None:
        self.session_id = session_id
        self._sessions = sessions
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._detector = detector or QuestionDetector()
        self._summarizer = summarizer
        #: Semantic understanding of a completed turn. Injected so tests can
        #: drive it with a fake completer; with no completer it degrades to
        #: the deterministic reading and costs nothing.
        self._understander = understander or QuestionUnderstander()

        self._seq = 0
        self._current_turn_id: int | None = None
        #: Newest summary text streamed for the in-flight turn, so that a
        #: supersede can preserve it instead of discarding it.
        self._current_partial: str = ""
        self._task: asyncio.Task | None = None
        #: A question detected as mid-clause waits here briefly for a
        #: continuation to supersede it -- see `consider` / `_delayed_ask`.
        self._pending_ask: asyncio.Task | None = None
        #: Speech-clock time of the first fragment of the turn currently being
        #: accumulated, and how many fragments it has taken. Bounds one turn's
        #: total hold and reports fragment counts; both reset once the turn is
        #: sent or abandoned.
        self._accumulating_since: float | None = None
        self._fragments = 0
        #: True between the VAD opening an interviewer utterance and its final
        #: transcript arriving. While this is set, a pause is provably not the
        #: end of the turn -- the interviewer is audibly still talking -- so a
        #: held fragment must not be sent no matter how long the hold was.
        self._speech_in_progress = False
        #: Material the interviewer pasted -- pending, or bound to the live
        #: turn. See app.realtime.attachments.
        self._attachments = AttachmentBuffer()
        #: The turn currently in flight, kept so a paste arriving just after
        #: it can re-ask that same question with the material attached rather
        #: than inventing a second turn for it. Carries `detail` too: without
        #: it the re-ask lost the deterministic layer's follow-up reading and
        #: degraded to new_question on the fallback path.
        self._live_question: _LiveQuestion | None = None
        #: The question that opened the current task thread. Task-continuation
        #: turns ("now do it in Java") need the original problem, which may be
        #: several turns back and out of a recent-window selection. Reset when
        #: a genuinely new question starts a new thread.
        self._thread_anchor: str | None = None
        self._background: set[asyncio.Task] = set()
        self._subscribers: set[Emitter] = set()
        self._replay: deque[Event] = deque(maxlen=settings.ws_replay_buffer)
        self._lock = asyncio.Lock()
        self._audio = None

    # ------------------------------------------------------------ subscribers

    def subscribe(self, emitter: Emitter) -> None:
        self._subscribers.add(emitter)

    def unsubscribe(self, emitter: Emitter) -> None:
        self._subscribers.discard(emitter)

    def replay_since(self, since_seq: int) -> list[Event]:
        """Events a reconnecting client missed. The session keeps running while
        the socket is down, so this is a catch-up, not a resume."""
        return [e for e in self._replay if e.seq > since_seq]

    async def emit(self, ev: Event) -> None:
        self._seq += 1
        ev.seq = self._seq
        self._replay.append(ev)
        for emitter in list(self._subscribers):
            try:
                await emitter(ev)
            except Exception:
                # A dead socket must not take down the session.
                logger.debug("subscriber_emit_failed seq=%d", ev.seq)
                self._subscribers.discard(emitter)

    # -------------------------------------------------------------- transcript

    async def on_transcript(
        self,
        text: str,
        source: TranscriptSource,
        is_final: bool,
        trace: LatencyTrace | None = None,
        now: float | None = None,
    ) -> None:
        if not is_final:
            await self.emit(
                event(EventType.TRANSCRIPT_PARTIAL, text=text, source=source.value)
            )
            return

        # The utterance VAD opened has now been transcribed, so the
        # interviewer is no longer mid-sentence as far as we can tell.
        if source == TranscriptSource.LOOPBACK:
            self._speech_in_progress = False

        self._sessions.add_transcript(
            TranscriptEntry(
                session_id=self.session_id, source=source, is_final=True, text=text
            )
        )
        await self.emit(event(EventType.TRANSCRIPT_FINAL, text=text, source=source.value))

        # Only the interviewer's channel drives question detection. The
        # candidate's own microphone is recorded for review, never answered.
        if source == TranscriptSource.MIC:
            return

        await self.consider(text, source, trace, now=now)

    async def on_context_attached(
        self,
        kind: str,
        content: str = "",
        name: str = "",
        image_bytes: bytes | None = None,
        now: float | None = None,
    ) -> None:
        """The interviewer pasted material (a table, a query, a screenshot).

        Three arrival orders, one turn in every case:

        * **paste, then ask** -- the material sits pending and is bound when
          the question is asked. No extra provider call.
        * **paste while a turn is accumulating** -- same thing; the fragments
          keep assembling and the material is bound when they finish. Still
          no extra call, which is why a paste mid-question is free.
        * **ask, then paste** -- the turn is already in flight, so it is
          re-asked *once* with the material attached. That supersedes the
          in-flight answer rather than opening a second turn, which is what
          keeps this from duplicating the question.

        A paste arriving after the answer is already delivered deliberately
        does *not* re-ask: the interviewer has the answer and can ask a
        follow-up, and re-asking there would answer the same question twice.
        """
        attach_at = now if now is not None else time.monotonic()
        try:
            if kind == AttachmentKind.IMAGE:
                # OCR is CPU-bound and this is the event loop -- a screenshot
                # pasted mid-answer must not stall the stream.
                attachment = await asyncio.to_thread(
                    build_attachment, kind, content, name, attach_at, image_bytes
                )
            else:
                attachment = build_attachment(kind, content, name, attach_at)
        except AttachmentError as exc:
            log_metric(
                "attachment_rejected",
                session_id=self.session_id,
                kind=kind,
                reason=exc.reason.value,
            )
            await self.emit(event(
                EventType.CONTEXT_REJECTED, kind=kind, reason=exc.reason.value,
                message=str(exc),
            ))
            return

        self._attachments.expire(attach_at)
        self._attachments.add(attachment)
        log_metric(
            "attachment_accepted",
            session_id=self.session_id,
            kind=attachment.kind.value,
            chars=len(attachment.content),
            from_image=attachment.from_image,
            pending=len(self._attachments.pending),
        )
        await self.emit(event(
            EventType.CONTEXT_ATTACHED,
            kind=attachment.kind.value,
            name=attachment.name,
            chars=len(attachment.content),
            from_image=attachment.from_image,
        ))

        # Still assembling, or nothing asked yet: the material will be bound
        # when the turn is sent. Nothing more to do, and no extra call.
        holding = self._pending_ask is not None and not self._pending_ask.done()
        if holding or self._live_question is None:
            return

        # A turn is in flight. Re-ask it once, now carrying the material --
        # same question, same relationship reading, so the understanding layer
        # and its fallback both see the turn for what it is rather than
        # treating it as a brand new question.
        live = self._live_question
        log_metric(
            "attachment_reask",
            session_id=self.session_id,
            question_id=self._current_turn_id,
            kind=attachment.kind.value,
            detail=live.detail,
        )
        await self.ask(
            live.question,
            live.classification,
            effective_question=live.effective_question,
            asked_at=attach_at,
            extend_attachments=True,
            detail=live.detail,
        )

    async def on_speech_start(self, utterance_id: int | None = None) -> None:
        """The interviewer began a new utterance (LOOPBACK only).

        This is the signal that makes turn assembly robust to fragments of
        arbitrary length. A hold can only ever be a guess at how long the next
        clause will take to say; knowing that speech is *currently happening*
        is a fact, and `_delayed_ask` waits on the fact rather than the guess.
        """
        self._speech_in_progress = True
        log_metric(
            "interviewer_speech_started",
            session_id=self.session_id,
            utterance_id=utterance_id,
            holding=self._pending_ask is not None and not self._pending_ask.done(),
        )

    async def consider(
        self,
        text: str,
        source: TranscriptSource = TranscriptSource.LOOPBACK,
        trace: LatencyTrace | None = None,
        *,
        now: float | None = None,
    ) -> None:
        # Only live interviewer speech may draw on or feed the detector's
        # preceding-context buffer -- a typed question is the user's own
        # prompt, not interviewer setup, and must behave exactly as before.
        # `now` is normally None (the detector falls back to time.monotonic())
        # -- it exists so a deterministic replay harness can drive the same
        # merge-window logic with synthetic timestamps instead of real sleeps.
        # A turn is "accumulating" while a fragment of it is still being held
        # and nothing has been sent. The detector widens its merge window in
        # that state, because a held fragment whose continuation failed to
        # merge would lose its words entirely.
        accumulating = self._pending_ask is not None and not self._pending_ask.done()
        detection = self._detector.inspect(
            text,
            now,
            buffer_context=source == TranscriptSource.LOOPBACK,
            accumulating=accumulating,
        )
        if settings.question_detector_diagnostics:
            log_metric(
                "question_detector_decision",
                session_id=self.session_id,
                source=source.value,
                text=text,
                detected=detection.accepted,
                category=detection.classification.category.value
                if detection.classification else None,
                confidence=detection.classification.confidence
                if detection.classification else None,
                reason=detection.reason.value if detection.reason else None,
                detail=detection.detail,
            )
        if not detection.accepted:
            await self.emit(
                event(
                    EventType.QUESTION_REJECTED,
                    text=detection.text,
                    reason=detection.reason.value if detection.reason else None,
                )
            )
            return

        if trace is not None:
            trace.question_detected_at = time.monotonic()

        # Any older mid-clause fragment is now superseded by real speech --
        # either this *is* its continuation (already merged by the detector's
        # own coalescing, and re-evaluated fresh below) or it is unrelated and
        # the stale fragment must not fire on its own.
        self._cancel_pending_ask()

        speech_now = now if now is not None else time.monotonic()
        hold_ms = detection.hold_ms

        # The fragment debounce. A turn that has already arrived in pieces is
        # very likely to arrive in one more, and an intermediate assembly can
        # easily read as complete on its own ("Can you explain how dependency
        # injection works" is a perfectly good question -- it just is not the
        # one being asked). Firing there is what used to cost a second
        # provider call, so a merged turn with no terminal "?" waits for the
        # next piece instead, and each new piece restarts that wait.
        #
        # Deliberately narrow: it needs an in-flight hold (so the turn has
        # demonstrably fragmented) and a merge. A one-shot complete question
        # never satisfies either and is never delayed by this.
        if hold_ms == 0 and accumulating and detection.supersedes:
            if detection.explicit_closure:
                log_metric(
                    "question_closure_detected",
                    session_id=self.session_id,
                    fragments=self._fragments + 1,
                )
            else:
                hold_ms = settings.question_continuation_ms

        if hold_ms > 0:
            if self._accumulating_since is None:
                self._accumulating_since = speech_now
                self._fragments = 1
            else:
                self._fragments += 1
            # Hard ceiling on one turn, so an interviewer who keeps qualifying
            # still gets an answer.
            spent_total = int((speech_now - self._accumulating_since) * 1000)
            hold_ms = max(
                0, min(hold_ms, settings.question_max_accumulation_ms - spent_total)
            )

        if hold_ms <= 0:
            self._end_accumulation()
            await self.ask(
                detection.text,
                detection.classification,
                effective_question=detection.effective_text,
                trace=trace,
                follow_up=detection.detail == "follow_up",
                asked_at=speech_now,
                # A merged detection is this turn continuing (a correction, a
                # closing clause), so it keeps the material already bound to
                # it instead of starting clean.
                extend_attachments=detection.supersedes,
                detail=detection.detail,
            )
            return

        # The hold is a window on the *speech* timeline, so whatever the STT
        # pass already spent counts against it: a 300ms transcription of a
        # fragment leaves 900ms of a 1200ms hold, not a fresh 1200ms. That
        # keeps a slow machine from stacking inference latency on top of the
        # wait, and means the deadline for a continuation is the same wall
        # clock instant regardless of how fast Whisper was.
        if trace is not None:
            spent = elapsed_ms(trace.speech_end_at, time.monotonic())
            hold_ms = max(0, hold_ms - spent)

        log_metric(
            "question_stabilization_started",
            session_id=self.session_id,
            finality=detection.finality.value,
            delay_ms=hold_ms,
            budget_ms=detection.hold_ms,
            fragments=self._fragments,
            merged=detection.supersedes,
        )
        self._pending_ask = asyncio.create_task(
            self._delayed_ask(detection, trace, hold_ms, speech_now)
        )

    def _update_thread_anchor(
        self, understanding: Understanding, question: str
    ) -> None:
        """Track which question opened the current task thread.

        A genuinely new question starts a new thread and becomes its own
        anchor. Everything else -- a follow-up, another method, a different
        language, a changed constraint -- is a refinement of the thread that
        is already open, so the anchor is left alone and stays reachable no
        matter how many turns the progression runs for.

        Only a real classification may move it: on a fallback the relationship
        is a guess, and moving the anchor on a guess would silently drop the
        original problem from a coding progression.
        """
        if understanding.source is not UnderstandingSource.LLM:
            return
        if understanding.is_new_task:
            self._thread_anchor = question

    async def _await_speech_end(self, started: float) -> int:
        """Block while the interviewer is mid-utterance. Returns ms waited.

        Polled rather than event-driven on purpose: the alternative is an
        asyncio primitive shared with a callback the STT thread drives, and a
        200ms poll against a multi-second accumulation ceiling costs nothing
        while being much harder to deadlock.
        """
        if not self._speech_in_progress:
            return 0
        ceiling = settings.question_max_accumulation_ms / 1000
        waited_from = time.monotonic()
        while self._speech_in_progress:
            if time.monotonic() - waited_from >= ceiling:
                log_metric(
                    "question_hold_ceiling_reached",
                    session_id=self.session_id,
                    ceiling_ms=settings.question_max_accumulation_ms,
                )
                break
            await asyncio.sleep(_SPEECH_POLL_SECONDS)
        return elapsed_ms(waited_from, time.monotonic())

    def _end_accumulation(self) -> None:
        """One turn's accumulation is over -- it is being sent, or it was
        cancelled. The next fragment starts a fresh budget."""
        self._accumulating_since = None
        self._fragments = 0

    def _cancel_pending_ask(self) -> None:
        if self._pending_ask is not None and not self._pending_ask.done():
            self._pending_ask.cancel()
        self._pending_ask = None

    def _abandon_accumulation(self) -> None:
        """Drop a half-accumulated turn without sending it (session close)."""
        self._cancel_pending_ask()
        self._end_accumulation()

    async def _delayed_ask(
        self,
        detection: Detection,
        trace: LatencyTrace | None,
        hold_ms: int | None = None,
        asked_at: float | None = None,
    ) -> None:
        """Give an unfinished question a bounded window to be superseded by
        its own continuation before spending a provider call on a fragment.

        The window length comes from the detection's own evidence, not from
        one global constant -- see `question_detector.Finality`.
        """
        started = time.monotonic()
        if hold_ms is None:
            hold_ms = detection.hold_ms
        try:
            await asyncio.sleep(hold_ms / 1000)
            # The hold was only ever a guess at how long the next clause takes
            # to say. If the interviewer is audibly mid-utterance when it
            # expires, that guess was simply too short, and firing now would
            # answer a fragment while its continuation is still being spoken.
            # Wait for the fact instead, bounded by the accumulation ceiling.
            waited = await self._await_speech_end(started)
            if waited:
                log_metric(
                    "question_hold_extended_by_speech",
                    session_id=self.session_id,
                    extra_ms=waited,
                    fragments=self._fragments,
                )
                # A final arrived while we waited, so `consider` has already
                # taken over this turn (it cancels this task). Reaching here
                # means it did not -- the utterance produced no usable final --
                # so fall through and send what we have.
        except asyncio.CancelledError:
            log_metric(
                "question_stabilization_superseded",
                session_id=self.session_id,
                waited_ms=elapsed_ms(started, time.monotonic()),
            )
            raise
        log_metric(
            "question_stabilization_completed",
            session_id=self.session_id,
            duration_ms=elapsed_ms(started, time.monotonic()),
            fragments=self._fragments,
        )
        # The window closed with no further speech: this turn is as complete
        # as it is going to get, so stop accumulating and send it.
        self._end_accumulation()
        await self.ask(
            detection.text,
            detection.classification,
            effective_question=detection.effective_text,
            trace=trace,
            follow_up=detection.detail == "follow_up",
            asked_at=asked_at,
            extend_attachments=detection.supersedes,
            detail=detection.detail,
        )

    # ------------------------------------------------------------------- asking

    async def ask(
        self,
        question: str,
        classification: Classification | None = None,
        *,
        effective_question: str | None = None,
        trace: LatencyTrace | None = None,
        follow_up: bool = False,
        asked_at: float | None = None,
        extend_attachments: bool = False,
        detail: str | None = None,
    ) -> int:
        """Start answering. Cancels any answer already in flight.

        `question` is what gets shown and persisted -- the interview panel and
        session history. `effective_question` is what the LLM actually sees;
        it defaults to `question` and only differs when the detector attached
        preceding setup context, so a verbose merged prompt never has to
        appear in the UI as if the interviewer said all of it.
        """
        effective_question = effective_question or question
        async with self._lock:
            cancel_started = time.monotonic()
            await self._cancel_current(CancelReason.SUPERSEDED)
            if trace is not None:
                trace.cancel_wait_ms = elapsed_ms(cancel_started, time.monotonic())

            if classification is None:
                from app.intelligence.classifier import classify

                classification = classify(question)

            # Bind whatever the interviewer pasted around this question. A
            # follow-up inherits the previous turn's material instead, so
            # "and what about row 3?" can still see the table -- without the
            # table being re-bound to an unrelated question later.
            # Bound against the *question's* clock, not the wall clock at the
            # moment we happen to reach here. A paste and a question are both
            # timestamped on the same monotonic clock in production, and a
            # held turn can be sent seconds after the words were spoken -- so
            # comparing the paste against "now" would measure the hold, not
            # the gap between the paste and the question.
            if follow_up:
                attachments = self._attachments.carry_forward()
            else:
                attachments = self._attachments.bind(
                    asked_at if asked_at is not None else time.monotonic(),
                    extend=extend_attachments,
                )
            self._live_question = _LiveQuestion(
                question, effective_question, classification, detail
            )

            turn = self._sessions.create_turn(
                Turn(
                    session_id=self.session_id,
                    seq=self._sessions.next_seq(self.session_id),
                    question=question,
                    category=classification.category.value,
                    domain=classification.domain.value,
                    confidence=classification.confidence,
                    status=TurnStatus.PENDING,
                )
            )
            self._current_turn_id = turn.turn_id
            self._current_partial = ""
            log_metric(
                "question_detected",
                session_id=self.session_id,
                question_id=turn.turn_id,
                # The join that closes the chain: every STT-side metric is
                # keyed by (session_id, utterance_id) and every answer-side
                # one by (session_id, question_id). This line is where those
                # two identifier spaces meet.
                utterance_id=trace.utterance_id if trace is not None else None,
                category=classification.category.value,
                confidence=classification.confidence,
            )
            # Emitted before generation starts so the UI can show what was heard
            # and how it was understood, even if the answer later fails.
            await self.emit(event(
                EventType.QUESTION_DETECTED,
                turn_id=turn.turn_id,
                question=question,
                classification=classification.model_dump(mode="json"),
            ))
            if trace is not None:
                trace.ask_started_at = time.monotonic()
            self._task = asyncio.create_task(
                self._answer(
                    turn.turn_id, question, effective_question, classification,
                    trace, attachments, detail,
                )
            )
            return turn.turn_id

    async def cancel(self, reason: CancelReason = CancelReason.USER_STOP) -> None:
        async with self._lock:
            await self._cancel_current(reason)

    async def _cancel_current(self, reason: CancelReason) -> None:
        task, turn_id = self._task, self._current_turn_id
        partial = self._current_partial
        self._task, self._current_turn_id = None, None
        self._current_partial = ""
        if task is None or task.done():
            return

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        if turn_id is None:
            return

        # An answer that already put text on screen is worth keeping: the
        # candidate may have been mid-read when the interviewer moved on.
        # One that produced nothing is just noise in the history.
        if partial.strip():
            self._sessions.interrupt_turn(turn_id, partial)
            log_metric(
                "answer_interrupted_with_partial",
                session_id=self.session_id,
                question_id=turn_id,
                reason=reason.value,
                answer_partial_chars=len(partial),
            )
        else:
            self._sessions.mark_turn(turn_id, TurnStatus.CANCELLED)
            log_metric(
                "answer_cancelled_before_content",
                session_id=self.session_id,
                question_id=turn_id,
                reason=reason.value,
            )

        await self.emit(
            event(
                EventType.ANSWER_CANCELLED,
                turn_id=turn_id,
                reason=reason.value,
                # Additive, optional fields: existing clients ignore them.
                interrupted=bool(partial.strip()),
                partial_summary=partial or None,
            )
        )

    def _is_current(self, turn_id: int) -> bool:
        return self._current_turn_id == turn_id

    async def _answer(
        self,
        turn_id: int,
        question: str,
        effective_question: str,
        classification: Classification,
        trace: LatencyTrace | None = None,
        attachments: list[Attachment] | None = None,
        detail: str | None = None,
    ) -> None:
        started = time.monotonic()
        try:
            log_metric(
                "question_processing_started",
                session_id=self.session_id,
                question_id=turn_id,
            )
            await self.emit(event(EventType.ANSWER_STARTED, turn_id=turn_id, question=question,
                                  classification=classification.model_dump(mode="json")))

            route = route_for(classification.category)

            # History first: a small SQLite read, and the classifier needs it
            # to judge the conversational relationship.
            history_started = time.monotonic()
            history = await asyncio.to_thread(
                self._memory.bounded_context, self.session_id
            )
            history_ms = elapsed_ms(history_started, time.monotonic())

            # Then the two genuinely slow, genuinely independent calls run
            # concurrently rather than back to back: retrieval (embedding +
            # FAISS + a SQLite join, all on worker threads) and the
            # understanding call (one bounded network round trip).
            #
            # Safe to overlap because retrieval does not depend on the
            # classification: its route comes from classification.category,
            # the deterministic classifier, which is already known. Nothing
            # here reads the understanding result before it has arrived.
            #
            # Cancellation is inherited -- this whole coroutine is
            # self._task, which a superseding question cancels, and
            # asyncio.gather propagates that into both legs. A stale
            # classification therefore cannot come back and influence an
            # answer, because understand() re-raises CancelledError rather
            # than converting it into a fallback.
            retrieval_started = time.monotonic()
            understanding_started = retrieval_started
            chunks, understanding = await asyncio.gather(
                self._retrieve(route, effective_question, turn_id),
                self._understander.understand(
                    effective_question,
                    session_id=self.session_id,
                    question_id=turn_id,
                    detail=detail,
                    history=history,
                    attachment_summaries=summarise_attachments(attachments or []),
                    utterance_id=trace.utterance_id if trace is not None else None,
                ),
            )
            settled_at = time.monotonic()
            context_found = bool(chunks)
            if trace is not None:
                trace.retrieval_ms = elapsed_ms(retrieval_started, settled_at)
            understanding_ms = elapsed_ms(understanding_started, settled_at)

            # Context selection, driven by the relationship the classifier
            # reported. Only ever narrows the already-bounded window, and
            # returns it whole whenever the classifier was not consulted -- so
            # the failure path keeps the behaviour this system had before
            # understanding existed. The thread anchor is what keeps the
            # underlying task reachable across a long progression.
            selection_started = time.monotonic()
            selected_history = select_context(
                history, understanding, self._thread_anchor
            )
            selection_ms = elapsed_ms(selection_started, time.monotonic())
            self._update_thread_anchor(understanding, question)

            prompt_started = time.monotonic()
            prompt = build_prompt(
                effective_question, classification.category,
                [c.as_context() for c in chunks], selected_history,
                attachments=render_attachments(attachments or []),
                understanding=understanding.as_prompt_section(),
            )
            if trace is not None:
                trace.prompt_build_ms = elapsed_ms(prompt_started, time.monotonic())
            log_metric(
                "llm_request_prepared",
                session_id=self.session_id,
                question_id=turn_id,
                prompt_chars=len(prompt),
                prompt_lines=prompt.count("\n") + 1 if prompt else 0,
                context_chunks=len(chunks),
                history_turns=len(selected_history) // 2,
                history_turns_available=len(history) // 2,
                # Each stage separately, so no cost hides inside another.
                history_latency_ms=history_ms,
                understanding_latency_ms=understanding_ms,
                context_selection_ms=selection_ms,
                relationship=understanding.relationship.value,
                understanding_source=understanding.source.value,
            )

            answer = await self._stream(turn_id, prompt, trace)
            answer = validate(answer, classification, context_found=context_found)

            if not self._is_current(turn_id):
                return  # superseded while the last chunk was in flight

            latency_ms = int((time.monotonic() - started) * 1000)
            self._sessions.complete_turn(
                turn_id,
                answer=answer,
                context_found=context_found,
                latency_ms=latency_ms,
                hits=[
                    RetrievalHit(chunk_id=c.chunk_id, score=c.score, rank=i)
                    for i, c in enumerate(chunks)
                ],
            )
            await self.emit(event(
                EventType.ANSWER_COMPLETED,
                turn_id=turn_id,
                answer=answer.model_dump(mode="json"),
                context_found=context_found,
                latency_ms=latency_ms,
                retrieval_hits=[
                    {"chunk_id": c.chunk_id, "document_id": c.document_id,
                     "score": round(c.score, 4), "title": c.title}
                    for c in chunks
                ],
            ))
            # This turn has been answered, so it is no longer something a
            # later utterance may extend. Keeps follow-up recency intact (a
            # bare "Why?" still works) but drops the long imperative-task
            # merge window, which would otherwise glue the *next* coding
            # question onto this one several seconds later.
            self._detector.close_turn()
            # The turn keeps its material until the *next* non-follow-up
            # question is asked, so a follow-up can still refer to the table.
            # `_live_question` is cleared, though: a paste arriving now has an
            # answered question to look at and must not re-ask it.
            self._live_question = None
            self._schedule_summary()

        except asyncio.CancelledError:
            raise
        except (LLMError, AnswerValidationError) as exc:
            self._sessions.mark_turn(turn_id, TurnStatus.FAILED)
            await self.emit(event(EventType.ANSWER_ERROR, turn_id=turn_id,
                                  code=type(exc).__name__, message=str(exc)))
        except Exception as exc:
            logger.exception("answer_pipeline_failed turn=%d", turn_id)
            self._sessions.mark_turn(turn_id, TurnStatus.FAILED)
            await self.emit(event(EventType.ANSWER_ERROR, turn_id=turn_id,
                                  code="InternalError", message=str(exc)))

    async def _retrieve(self, route: Route, question: str, turn_id: int) -> list[RetrievedChunk]:
        if route not in (Route.RAG, Route.FOLLOW_UP):
            return []
        await self.emit(event(
            EventType.ANSWER_RETRIEVING, turn_id=turn_id,
            knowledge_types=[k.value for k in PERSONAL_KNOWLEDGE_TYPES],
        ))
        return await self._retriever.retrieve(
            question, knowledge_types=list(PERSONAL_KNOWLEDGE_TYPES)
        )

    async def _stream(self, turn_id: int, prompt: str, trace: LatencyTrace | None = None) -> Answer:
        buffer = ""
        last_summary = ""
        requested_at = time.monotonic()
        saw_first_token = False
        if trace is not None:
            trace.llm_request_at = requested_at
        log_metric("llm_request_started", session_id=self.session_id, question_id=turn_id)
        async for chunk in self._llm.stream_answer(prompt):
            if not saw_first_token:
                saw_first_token = True
                now = time.monotonic()
                log_metric(
                    "llm_first_token_received",
                    session_id=self.session_id,
                    question_id=turn_id,
                    duration_ms=elapsed_ms(requested_at, now),
                )
                if trace is not None:
                    # This app never streams a textless chunk (JSON text only,
                    # function calling disabled), so "first response" and
                    # "first text token" are the same moment here.
                    trace.llm_first_response_at = now
                    trace.llm_first_text_token_at = now
            buffer += chunk
            if not self._is_current(turn_id):
                raise asyncio.CancelledError()
            partial = extract_partial_summary(buffer)
            if partial and partial != last_summary:
                is_first_visible = last_summary == ""
                last_summary = partial
                # Mirrored onto the session so a supersede can preserve
                # exactly what the user had already been shown.
                self._current_partial = partial
                await self.emit(event(EventType.ANSWER_DELTA, turn_id=turn_id, summary=partial))
                if is_first_visible and trace is not None:
                    trace.emit_first_token(turn_id)

        log_metric(
            "llm_response_completed",
            session_id=self.session_id,
            question_id=turn_id,
            duration_ms=elapsed_ms(requested_at, time.monotonic()),
            chars=len(buffer),
        )
        if not buffer.strip():
            raise LLMError("Model returned an empty response")
        return Answer.model_validate(parse_answer_payload(buffer))

    # ------------------------------------------------------------ summarisation

    def _schedule_summary(self) -> None:
        """Fire-and-forget: summarising must never delay the next question."""
        if self._summarizer is None or not hasattr(self._memory, "verbatim_from_seq"):
            return
        boundary = self._memory.verbatim_from_seq(self.session_id)
        if not self._summarizer.needs_summary(self.session_id, boundary):
            return

        task = asyncio.create_task(self._summarizer.summarize(self.session_id, boundary))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # -------------------------------------------------------------------- audio

    async def start_audio(self, pipeline) -> list[str]:
        """Begin live capture. Returns the channels actually opened."""
        if self._audio is not None:
            return [c.value for c in self._audio.channels]

        await asyncio.to_thread(pipeline.start)
        self._audio = pipeline
        channels = [c.value for c in pipeline.channels]
        await self.emit(event(EventType.SESSION_STATUS, audio="ok", channels=channels))
        logger.info("audio_attached session=%s channels=%s", self.session_id, channels)
        return channels

    async def stop_audio(self) -> None:
        if self._audio is None:
            return
        pipeline, self._audio = self._audio, None
        await asyncio.to_thread(pipeline.stop)
        await self.emit(event(EventType.SESSION_STATUS, audio="stopped"))

    @property
    def audio_active(self) -> bool:
        return self._audio is not None

    # ------------------------------------------------------------------ closing

    async def close(self) -> None:
        await self.stop_audio()
        self._abandon_accumulation()
        await self.cancel(CancelReason.SESSION_ENDED)
        for task in list(self._background):
            task.cancel()
        self._sessions.end(self.session_id)
        await self.emit(event(
            EventType.SESSION_ENDED,
            session_id=self.session_id,
            turn_count=len(self._sessions.get_turns(self.session_id)),
        ))
