"""One whole interview, replayed deterministically.

Fifteen turns through every mechanism in the pipeline at once: attachment
ownership, thread continuity, a constraint change, a reference to generated
code, a topic switch, a behavioural thread, a provider failure and the recovery
after it. The per-mechanism suites prove each piece; this proves they compose,
which is where an interview actually breaks.

Everything is scripted: explicit speech timestamps, a scripted classifier, a
fake provider. No sleeps, no hardware, no network.

The invariant that ties it together is the provider-call count. Every wasted
call is real money and real latency on the interviewer's clock, and every
missing one is a question that went unanswered -- so the count is asserted
turn by turn rather than at the end.
"""

import pytest

from app.llm.base import LLMError, LLMErrorKind
from app.llm.base import LLMClient
from app.memory.sqlite_memory import SqliteSessionMemory
from app.realtime.events import EventType
from app.schemas.answer import Answer
from app.sessions.schemas import TranscriptSource
from tests.fakes import SlowStreamingLLM
from tests.replay_harness import ReplayHarness
from tests.test_understanding_session import CountingCompleter, reply

asyncio_test = pytest.mark.asyncio

PASTED_CODE = (
    "def total_orders(rows):\n"
    "    return sum(r['amount'] for r in rows)"
)
ERROR_LOG = "TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'"
GENERATED_CODE = (
    "def total_orders(rows):\n"
    "    return sum(r['amount'] or 0 for r in rows)"
)


class ScriptedLLM(LLMClient):
    """Answers from a script, and can be made to fail for exactly one turn.

    A single `fail_next` flag rather than a whole fake provider: the failure
    under test is one turn breaking in the middle of a live interview, not a
    provider that is down.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.answer = Answer(summary="An answer.", key_points=["a", "b"])
        self.fail_next = False

    async def generate_answer(self, prompt: str) -> Answer:
        raise NotImplementedError

    async def stream_answer(self, prompt: str):
        self.prompts.append(prompt)
        if self.fail_next:
            self.fail_next = False
            raise LLMError(
                "The provider is unavailable.", kind=LLMErrorKind.NETWORK
            )
        payload = self.answer.model_dump_json()
        for i in range(0, len(payload), 40):
            yield payload[i:i + 40]


#: The classifier's reading of each interviewer turn, in order. Written out
#: rather than generated so the scenario reads as a conversation and each
#: turn's expected relationship is visible next to the words that produced it.
SCRIPT = [
    # 1. conceptual opener
    reply(intent="conceptual", relationship="new_question", topic="aggregation"),
    # 4. about the pasted code
    reply(intent="troubleshooting", relationship="new_question",
          needs_attachments=True, topic="aggregation bug"),
    # 5. "Why?"
    reply(intent="conceptual", relationship="follow_up",
          needs_previous_context=True, needs_attachments=True),
    # 6. implementation
    reply(intent="coding", relationship="new_implementation",
          needs_previous_context=True, needs_attachments=True),
    # 7. complexity
    reply(intent="conceptual", relationship="follow_up", needs_previous_code=True),
    # 8. constraint change
    reply(intent="conceptual", relationship="constraint_change",
          constraints=["100 million rows"], needs_previous_context=True),
    # 9. optimization
    reply(intent="optimization", relationship="follow_up",
          needs_previous_code=True, needs_previous_context=True),
    # 10. about the generated code
    reply(intent="conceptual", relationship="follow_up", needs_previous_code=True),
    # 11. unrelated topic
    reply(intent="conceptual", relationship="new_question", topic="decorators"),
    # 12. behavioural opener
    reply(intent="behavioral", relationship="new_question"),
    # 13. behavioural follow-up
    reply(intent="behavioral", relationship="follow_up",
          needs_previous_context=True),
    # 14. the turn whose provider call fails
    reply(intent="conceptual", relationship="new_question"),
    # 15. the turn after the failure
    reply(intent="conceptual", relationship="new_question"),
]


@asyncio_test
async def test_a_whole_interview(monkeypatch):
    completer = CountingCompleter(*SCRIPT)
    llm = ScriptedLLM()
    h = ReplayHarness(
        llm=llm, memory_factory=SqliteSessionMemory, monkeypatch=monkeypatch
    )
    from app.realtime.question_understanding import QuestionUnderstander

    h.live._understander = QuestionUnderstander(completer)

    async def say(text, now):
        await h.live.on_speech_start(1)
        await h.live.on_transcript(
            text, TranscriptSource.LOOPBACK, is_final=True, now=now
        )
        await h.settle()

    async def candidate(text, now):
        """MIC: recorded, never answered."""
        await h.live.on_transcript(
            text, TranscriptSource.MIC, is_final=True, now=now
        )
        await h.settle()

    async def paste(content, now, kind="code"):
        await h.live.on_context_attached(kind=kind, content=content, now=now)
        await h.settle()

    def answers() -> list:
        return h.result.of(EventType.ANSWER_COMPLETED)

    try:
        # -- 1. conceptual opener ------------------------------------------
        await say("How would you total the amounts across orders?", now=100.0)
        assert len(llm.prompts) == 1
        assert "total the amounts across orders" in llm.prompts[-1]
        assert "Previous Q&A" not in llm.prompts[-1], "an opener carried history"

        # -- 2. the candidate thinks out loud ------------------------------
        await candidate("I would probably use a generator expression.", now=110.0)
        assert len(llm.prompts) == 1, "candidate speech was answered"

        # -- 3. the candidate pastes their code ----------------------------
        await paste(PASTED_CODE, now=120.0)
        assert len(llm.prompts) == 1, "a paste was answered as a question"
        assert h.result.of(EventType.CONTEXT_ATTACHED)

        # -- 4. the interviewer asks about the paste -----------------------
        await paste(ERROR_LOG, now=125.0, kind="text")
        await say("Why is this failing?", now=126.0)
        assert len(llm.prompts) == 2
        turn4 = llm.prompts[-1]
        assert PASTED_CODE in turn4 and ERROR_LOG in turn4, "material not bound"
        assert turn4.index(PASTED_CODE) < turn4.index(ERROR_LOG), "order lost"
        # Provenance: the paste is material, not something the interviewer said.
        assert "MATERIAL THE INTERVIEWER PROVIDED" in turn4
        for entry in h.result.of(EventType.TRANSCRIPT_FINAL):
            assert PASTED_CODE not in entry.data.get("text", "")

        # -- 5. a bare follow-up claims the same material ------------------
        await say("Why?", now=135.0)
        assert len(llm.prompts) == 3
        assert PASTED_CODE in llm.prompts[-1], "a follow-up lost the material"
        assert "Previous Q&A" in llm.prompts[-1], "a follow-up lost the thread"

        # -- 6. implementation --------------------------------------------
        llm.answer = Answer(
            summary="Coalesce the nulls before summing.",
            approach=["guard each row"],
            code=GENERATED_CODE,
            complexity={"time": "O(n)", "space": "O(1)"},
        )
        await say("Now give me the implementation.", now=145.0)
        assert len(llm.prompts) == 4
        assert "edge_cases" in llm.prompts[-1], "not asked for the coding shape"
        assert "Now give me the implementation" in llm.prompts[-1]

        # -- 7. complexity, referring to the code just generated -----------
        await say("What's the complexity?", now=155.0)
        assert len(llm.prompts) == 5
        assert GENERATED_CODE in llm.prompts[-1], "generated code unreachable"
        assert "FROM YOUR OWN EARLIER ANSWER" in llm.prompts[-1]

        # -- 8a. a bare premise is setup, not a question -------------------
        # "Assume the table has one hundred million rows." asks for nothing,
        # so it costs no provider call. `assume` is deliberately absent from
        # the imperative vocabulary for exactly this reason. It is remembered
        # and attached to whatever the interviewer asks next.
        await say("Assume the table has one hundred million rows.", now=160.0)
        assert len(llm.prompts) == 5, "a bare premise was answered on its own"
        assert completer.calls == 5, "a bare premise reached the classifier"
        rejected = h.result.of(EventType.QUESTION_REJECTED)
        assert rejected, "the premise was not reported as rejected"

        # -- 8b. the interviewer revises the figure ------------------------
        await say("No, make that one hundred million rows.", now=165.0)
        assert len(llm.prompts) == 6, "the correction produced no answer"
        turn8 = llm.prompts[-1]
        assert "one hundred million rows" in turn8
        # The *active* thread is still reachable -- which is the debugging
        # thread opened at step 4, not the session's opening question. Turn 4
        # was a new subject (the failing code, not how to total amounts), so
        # the anchor moved there and the opener is legitimately out of the
        # window. Asserting the opener would be asserting a leak.
        assert "Previous Q&A" in turn8, "a constraint change lost the thread"
        assert "complexity" in turn8, "the immediately preceding turn was dropped"
        assert "Why is this failing" in turn8, "the thread anchor was dropped"

        # -- 9. optimization under the new constraint ----------------------
        await say("How would you optimize it?", now=175.0)
        assert len(llm.prompts) == 7
        assert GENERATED_CODE in llm.prompts[-1]
        assert "one hundred million" in llm.prompts[-1], "latest constraint dropped"

        # -- 10. asking about the generated code directly ------------------
        await say("Why did you write it that way?", now=185.0)
        assert len(llm.prompts) == 8
        assert GENERATED_CODE in llm.prompts[-1]

        # -- 11. a completely unrelated question ---------------------------
        await say("Now explain Python decorators.", now=300.0)
        assert len(llm.prompts) == 9
        turn11 = llm.prompts[-1]
        assert "Python decorators" in turn11
        assert PASTED_CODE not in turn11, "pasted material crossed a topic boundary"
        assert GENERATED_CODE not in turn11, "generated code crossed a boundary"
        assert "one hundred million" not in turn11, "a stale constraint carried over"
        assert "Previous Q&A" not in turn11, "an unrelated question inherited history"

        # -- 12/13. a behavioural thread -----------------------------------
        llm.answer = Answer(summary="Situation, task, action, result.")
        await say("Tell me about a time you disagreed with your manager.",
                  now=320.0)
        assert len(llm.prompts) == 10
        assert "Situation" in llm.prompts[-1], "not asked for the behavioural shape"

        await say("What did you do?", now=330.0)
        assert len(llm.prompts) == 11
        assert "Previous Q&A" in llm.prompts[-1], "the behavioural thread broke"
        assert "disagreed with your manager" in llm.prompts[-1]

        # -- 14. the provider fails on one turn ----------------------------
        llm.fail_next = True
        answers_before = len(answers())
        await say("What is a covering index?", now=400.0)
        assert len(llm.prompts) == 12, "the failed turn did not reach the provider"
        assert len(answers()) == answers_before, "a failed turn produced an answer"
        errors = h.result.of(EventType.ANSWER_ERROR)
        assert len(errors) == 1
        assert errors[-1].data["code"] == "LLMError"

        # -- 15. and the next question still works -------------------------
        await say("What is a hash map?", now=410.0)
        assert len(llm.prompts) == 13, "the failure poisoned the next turn"
        assert len(answers()) == answers_before + 1
        assert "hash map" in llm.prompts[-1]

        # ================================================ whole-run invariants

        # One classifier call per interviewer turn. Never for the candidate's
        # speech, never for a paste, never for a fragment.
        assert completer.calls == 13, (
            f"expected one understanding call per interviewer turn, got "
            f"{completer.calls}"
        )
        assert len(llm.prompts) == 13

        # Every turn ended somewhere terminal, and nothing is still in flight.
        assert len(answers()) == 12
        assert len(h.result.of(EventType.ANSWER_ERROR)) == 1
        assert h.live._task is None or h.live._task.done()

        # No stale answer survived: each completion belongs to a distinct turn,
        # in order.
        turn_ids = [e.turn_id for e in answers()]
        assert turn_ids == sorted(turn_ids)
        assert len(set(turn_ids)) == len(turn_ids), "a turn completed twice"

        # Bounded: the last prompt of a 13-turn interview is not larger than an
        # early one by more than the material it deliberately carries.
        assert len(llm.prompts[-1]) < 2 * len(llm.prompts[0])
        assert llm.prompts[-1].count("\nQ: ") <= 3

        # Provenance held throughout: the candidate's own speech never became a
        # question, and no answer text was ever replayed as a transcript line.
        interviewer_turns = h.result.of(EventType.QUESTION_DETECTED)
        assert len(interviewer_turns) == 13
        assert all(
            "generator expression" not in e.data["question"]
            for e in interviewer_turns
        ), "candidate speech became an interviewer question"

        # Attachment ownership: bound to the turns that used it, and nothing is
        # left pending to contaminate a later question.
        assert h.live._attachments.pending == []

        # The exact wording of every interviewer turn survived.
        for spoken in (
            "Why is this failing",
            "Now give me the implementation",
            "No, make that one hundred million rows",
            "Tell me about a time you disagreed with your manager",
            "What is a hash map",
        ):
            assert any(spoken in e.data["question"] for e in interviewer_turns), (
                f"lost the exact wording of {spoken!r}"
            )
        # The rejected premise keeps its wording too -- it is reported to the
        # UI as heard-but-not-answered rather than silently dropped.
        assert any(
            "Assume the table has one hundred million rows" in e.data.get("text", "")
            for e in rejected
        ), "the premise was not reported verbatim"
    finally:
        h.dispose()
