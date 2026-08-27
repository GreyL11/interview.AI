import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from app.core.config import settings
from app.core.logging import get_logger
from app.documents.schemas import PERSONAL_KNOWLEDGE_TYPES, RetrievedChunk
from app.intelligence.answer_validator import AnswerValidationError, validate
from app.intelligence.router import Route, route_for
from app.llm.base import LLMClient, LLMError
from app.llm.prompts import build_prompt
from app.llm.streaming import extract_partial_summary, parse_answer_payload
from app.memory.base import SessionMemory
from app.realtime.events import CancelReason, Event, EventType, event
from app.realtime.question_detector import QuestionDetector
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
    ) -> None:
        self.session_id = session_id
        self._sessions = sessions
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._detector = detector or QuestionDetector()
        self._summarizer = summarizer

        self._seq = 0
        self._current_turn_id: int | None = None
        self._task: asyncio.Task | None = None
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
        self, text: str, source: TranscriptSource, is_final: bool
    ) -> None:
        if not is_final:
            await self.emit(
                event(EventType.TRANSCRIPT_PARTIAL, text=text, source=source.value)
            )
            return

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

        await self.consider(text)

    async def consider(self, text: str) -> None:
        detection = self._detector.inspect(text)
        if not detection.accepted:
            await self.emit(
                event(
                    EventType.QUESTION_REJECTED,
                    text=detection.text,
                    reason=detection.reason.value if detection.reason else None,
                )
            )
            return

        await self.ask(detection.text, detection.classification)

    # ------------------------------------------------------------------- asking

    async def ask(self, question: str, classification: Classification | None = None) -> int:
        """Start answering. Cancels any answer already in flight."""
        async with self._lock:
            await self._cancel_current(CancelReason.SUPERSEDED)

            if classification is None:
                from app.intelligence.classifier import classify

                classification = classify(question)

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
            # Emitted before generation starts so the UI can show what was heard
            # and how it was understood, even if the answer later fails.
            await self.emit(event(
                EventType.QUESTION_DETECTED,
                turn_id=turn.turn_id,
                question=question,
                classification=classification.model_dump(mode="json"),
            ))
            self._task = asyncio.create_task(self._answer(turn.turn_id, question, classification))
            return turn.turn_id

    async def cancel(self, reason: CancelReason = CancelReason.USER_STOP) -> None:
        async with self._lock:
            await self._cancel_current(reason)

    async def _cancel_current(self, reason: CancelReason) -> None:
        task, turn_id = self._task, self._current_turn_id
        self._task, self._current_turn_id = None, None
        if task is None or task.done():
            return

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        if turn_id is not None:
            self._sessions.mark_turn(turn_id, TurnStatus.CANCELLED)
            await self.emit(
                event(EventType.ANSWER_CANCELLED, turn_id=turn_id, reason=reason.value)
            )

    def _is_current(self, turn_id: int) -> bool:
        return self._current_turn_id == turn_id

    async def _answer(self, turn_id: int, question: str, classification: Classification) -> None:
        started = time.monotonic()
        try:
            await self.emit(event(EventType.ANSWER_STARTED, turn_id=turn_id, question=question,
                                  classification=classification.model_dump(mode="json")))

            route = route_for(classification.category)
            chunks = await self._retrieve(route, question, turn_id)
            context_found = bool(chunks)

            history = self._memory.bounded_context(self.session_id)
            prompt = build_prompt(
                question, classification.category, [c.as_context() for c in chunks], history
            )

            answer = await self._stream(turn_id, prompt)
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

    async def _stream(self, turn_id: int, prompt: str) -> Answer:
        buffer = ""
        last_summary = ""
        async for chunk in self._llm.stream_answer(prompt):
            buffer += chunk
            if not self._is_current(turn_id):
                raise asyncio.CancelledError()
            partial = extract_partial_summary(buffer)
            if partial and partial != last_summary:
                last_summary = partial
                await self.emit(event(EventType.ANSWER_DELTA, turn_id=turn_id, summary=partial))

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
        await self.cancel(CancelReason.SESSION_ENDED)
        for task in list(self._background):
            task.cancel()
        self._sessions.end(self.session_id)
        await self.emit(event(
            EventType.SESSION_ENDED,
            session_id=self.session_id,
            turn_count=len(self._sessions.get_turns(self.session_id)),
        ))
