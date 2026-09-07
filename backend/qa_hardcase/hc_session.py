"""HARD-CASE ADVERSARIAL PASS -- LiveSession layer (report sections 2, 3, 5,
8-13, 18).

Drives the REAL LiveSession end to end (detector -> accumulation ->
understanding -> attachments -> answer) through tests.replay_harness, with
fake LLM/completer/storage exactly as the production suite does. No
production code is touched.

Run: backend/.venv/Scripts/python.exe qa_hardcase/hc_session.py
"""

import asyncio
import json
import time

from hc_common import Recorder  # noqa: E402  (path bootstrap lives there)

from app.core.config import settings  # noqa: E402
from app.realtime.events import EventType  # noqa: E402
from app.realtime.question_understanding import QuestionUnderstander  # noqa: E402
from app.sessions.schemas import TranscriptSource  # noqa: E402
from tests.fakes import SlowStreamingLLM  # noqa: E402
from tests.replay_harness import ReplayHarness  # noqa: E402

rec = Recorder("hc_session")

# The real defaults, captured before any harness overwrites them globally.
HOLD_INCOMPLETE_MS = settings.question_hold_incomplete_ms
STABILIZATION_MS = settings.question_stabilization_ms
CONTINUATION_MS = settings.question_continuation_ms
MAX_ACCUMULATION_MS = settings.question_max_accumulation_ms


def reply(**overrides) -> str:
    body = {
        "intent": "conceptual", "relationship": "new_question", "topic": "t",
        "domain": "d", "task": "task", "constraints": [], "requested_output": [],
        "entities": [], "needs_previous_context": False, "needs_previous_answer": False,
        "needs_previous_code": False, "needs_attachments": False, "confidence": 0.9,
    }
    body.update(overrides)
    return json.dumps(body)


class CountingCompleter:
    """Counts understanding calls and replays scripted classifications.

    Same shape as tests/test_understanding_session.py's, reproduced here so
    this script cannot break, or be broken by, the production suite.
    """

    def __init__(self, *responses: str, delay: float = 0.0, error: Exception | None = None):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.delay = delay
        self.error = error

    async def complete_json(self, prompt: str, *, model: str = "", timeout_seconds: float = 5.0) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if not self._responses:
            return reply()
        return self._responses.pop(0)

    @property
    def calls(self) -> int:
        return len(self.prompts)


def build(*, real_holds: bool = False, llm=None, completer=None, stabilization_ms: int = 1):
    h = ReplayHarness(llm=llm if llm is not None else SlowStreamingLLM(chunk_delay=0),
                      stabilization_ms=stabilization_ms)
    if real_holds:
        # The harness squashes both hold budgets to 1ms for speed; the pause
        # matrix needs the production values or the hold expires before the
        # continuation is even sent.
        settings.question_hold_incomplete_ms = HOLD_INCOMPLETE_MS
        settings.question_stabilization_ms = STABILIZATION_MS
    if completer is not None:
        h.live._understander = QuestionUnderstander(completer)
    return h


async def say(h, text: str, at: float | None = None, source: str = "LOOPBACK") -> None:
    await h.live.on_transcript(
        text, TranscriptSource[source], True,
        now=at if at is not None else time.monotonic(),
    )


def snapshot(h) -> dict:
    r = h.result
    return {
        "detected": [e.data["question"] for e in r.of(EventType.QUESTION_DETECTED)],
        "rejected": [e.data["text"] for e in r.of(EventType.QUESTION_REJECTED)],
        "completed": r.completed_turn_ids(),
        "cancelled": r.cancelled_turn_ids(),
        "failed": [e.turn_id for e in r.of(EventType.ANSWER_ERROR)],
    }


# ======================================================================= 2
async def section_2_pauses() -> None:
    """The mandated pause matrix, with REAL hold timing.

    A fragment held for question_hold_incomplete_ms (3000ms) either has its
    continuation merged onto it (pause < hold) or fires alone and the
    continuation lands as its own question (pause > hold).
    """
    print("\n--- 2. pause matrix (real timing, ~40s) ---")
    for pause in (0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        h = build(real_holds=True)
        try:
            t0 = time.monotonic()
            await say(h, "How would you design a distributed", t0)
            await asyncio.sleep(pause)
            await say(h, "system that processes millions of events per second?",
                      time.monotonic())
            await h.settle()
            await asyncio.sleep(0.05)
            await h.settle()
            snap = snapshot(h)
            merged_into_one = len(snap["detected"]) == 1 and len(snap["completed"]) == 1
            at_the_boundary = abs(pause - HOLD_INCOMPLETE_MS / 1000) < 1e-9
            if at_the_boundary:
                # HC-008. The hold expiring and the continuation arriving are
                # two events with no defined order, so both outcomes are
                # legal and neither is asserted. What is asserted is what a
                # race may never do: answer a turn twice, or lose a word.
                answered = " ".join(snap["detected"]).lower()
                expected = "no duplicate answer, no words lost"
                actual = expected if (
                    len(snap["detected"]) in (1, 2)
                    and len(snap["completed"]) == len(set(snap["completed"]))
                    and len(snap["completed"]) <= len(snap["detected"])
                    and "millions of events per second" in answered
                    and "distributed" in answered
                ) else (
                    f"{len(snap['detected'])} question(s), "
                    f"{len(snap['completed'])} answer(s), answered={answered!r}"
                )
            else:
                expected = "one-question-one-answer" if pause < HOLD_INCOMPLETE_MS / 1000 \
                    else "fragment-fired-then-second-question"
                actual = "one-question-one-answer" if merged_into_one else (
                    "fragment-fired-then-second-question"
                    if len(snap["detected"]) == 2 else f"other:{len(snap['detected'])}q"
                )
            rec.add("2-pause", f"pause={pause}s", expected, actual,
                    boundary=at_the_boundary, **snap)
        finally:
            await h.close()
    # Restore, so later sections get the fast harness defaults again.
    settings.question_hold_incomplete_ms = 1
    settings.question_stabilization_ms = 1


# ======================================================================= 2b
async def section_2_fragment_ladder() -> None:
    """Every prefix of one question, alone, with REAL hold timing.

    Two separate questions per fragment:
      * does it fire BEFORE its hold has elapsed (a premature answer)?
      * does it eventually fire at all (the documented abandoned-fragment
        fallback -- nothing is silently lost)?
    """
    print(chr(10) + "--- 2b. lone fragments through LiveSession (real timing) ---")
    for text, premature_expected in [
        ("How would you", False),
        ("How would you design", False),
        ("How would you design a", False),
        ("How would you design a distributed", False),
        ("How would you design a distributed system", True),   # complete: fires now
        ("How would you design a distributed system?", True),  # complete: fires now
    ]:
        h = build(real_holds=True)
        try:
            await say(h, text)
            await asyncio.sleep(0.25)          # well inside a 3000ms hold
            early = snapshot(h)
            fired_early = bool(early["completed"]) or bool(early["detected"])
            rec.add(
                "2b-premature", text,
                f"fires_within_250ms={premature_expected}",
                f"fires_within_250ms={fired_early}",
                **early,
            )
            await h.settle()                   # waits the hold out
            await asyncio.sleep(0.05)
            await h.settle()
            late = snapshot(h)
            rec.add(
                "2b-eventual", text, "eventually answered",
                "eventually answered" if late["completed"] else "LOST",
                **late,
            )
        finally:
            await h.close()
    settings.question_hold_incomplete_ms = 1
    settings.question_stabilization_ms = 1


# ======================================================================= 3
async def section_3_nested_clause() -> None:
    """The nested-clause gap, end to end: does the detector's premature
    COMPLETE actually cost a duplicate question/answer in the session?"""
    print("\n--- 3. nested clause end to end ---")
    CASES = [
        ["Explain how Kafka handles", "partitions?"],
        ["Explain how Kafka handles", "partitions when brokers fail?"],
        ["Tell me how the database", "handles replication?"],
        ["Describe how the system", "should recover from failure?"],
        ["How would you ensure", "that the service", "does not lose data?"],
        ["Can you explain why", "the consumer", "falls behind?"],
    ]
    for parts in CASES:
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter()
        h = build(llm=llm, completer=completer)
        try:
            t = time.monotonic()
            for index, part in enumerate(parts):
                await say(h, part, t + index * 0.4)
                await asyncio.sleep(0.01)
            await h.settle()
            snap = snapshot(h)
            # A 1ms hold cannot survive a 400ms gap, so whether the fragment
            # is answered before its continuation arrives is a coin flip --
            # measured over four runs, this section is the only one in the
            # file that flips. Both outcomes are correct by design: either the
            # continuation merges into one question, or the fragment was
            # already answered and the short continuation is read as a genuine
            # follow-up to it, which is what the follow-up bypass is for once
            # an answer exists. Recorded as an observation, because a check
            # that cannot tell a defect from a legal outcome is not a check.
            # The contract for these same scripts is asserted on production
            # timing in 3c below, and in tests/test_clause_saturation.py.
            rec.note(
                "3-e2e-fast-hold-observation", " + ".join(parts),
                questions=len(snap["detected"]), answers=len(snap["completed"]),
                understanding_calls=completer.calls, answer_calls=llm.started,
                note="squashed 1ms hold: outcome races with the continuation. "
                     "See 3c for production timing.",
                **snap,
            )
        finally:
            await h.close()

    print(chr(10) + "--- 3b. nested clause, continuation after the answer landed ---")
    for parts in CASES:
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter()
        h = build(llm=llm, completer=completer)
        try:
            t = time.monotonic()
            for index, part in enumerate(parts):
                await say(h, part, t + index * 0.4)
                await h.settle()   # the fragment's answer fully streams first
            snap = snapshot(h)
            # Settling between utterances *plus* a 1ms hold is a configuration
            # production never produces: the hold is 3000ms and the gap here is
            # 400ms, so the continuation always arrives mid-hold. Recorded as
            # an observation rather than a verdict -- 3c is the section that
            # asserts the contract, on the production timing.
            rec.note(
                "3b-e2e-settled-observation", " + ".join(parts),
                questions=len(snap["detected"]), answers=len(snap["completed"]),
                understanding_calls=completer.calls, answer_calls=llm.started,
                note="squashed hold: each fragment is answered before its "
                     "continuation is spoken. See 3c for production timing.",
                **snap,
            )
        finally:
            await h.close()


    print(chr(10) + "--- 3c. nested clause with REAL hold timing, 1.5s pauses ---")
    for parts in CASES:
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter()
        h = build(real_holds=True, llm=llm, completer=completer)
        try:
            await say(h, parts[0])
            for part in parts[1:]:
                await asyncio.sleep(1.5)
                await say(h, part)
            await h.settle()
            await asyncio.sleep(0.05)
            await h.settle()
            snap = snapshot(h)
            rec.add(
                "3c-e2e-real-holds", " + ".join(parts),
                "1 question(s), 1 answer(s)",
                f"{len(snap['detected'])} question(s), {len(snap['completed'])} answer(s)",
                understanding_calls=completer.calls, answer_calls=llm.started,
                **snap,
            )
        finally:
            await h.close()
    settings.question_hold_incomplete_ms = 1
    settings.question_stabilization_ms = 1

    # The limitation the fix knowingly keeps. A subordinate clause whose
    # predicate has no argument is undecidable from grammar alone -- "how
    # Kafka works" is finished, "how Kafka handles" is not -- so the tie is
    # broken by whether the speaker terminated the utterance. When Whisper
    # supplies a "." on a clause that was still open, the fragment fires.
    # Recorded, not asserted: closing it needs verb argument structure, which
    # a deterministic detector does not have.
    print(chr(10) + "--- 3e. known limitation: a period on an open clause ---")
    for parts in (
        ["Explain how Kafka handles.", "partitions?"],
        ["Tell me how the database.", "handles replication?"],
    ):
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter()
        h = build(real_holds=True, llm=llm, completer=completer)
        try:
            await say(h, parts[0])
            await asyncio.sleep(1.5)
            await say(h, parts[1])
            await h.settle()
            await asyncio.sleep(0.05)
            await h.settle()
            snap = snapshot(h)
            rec.note(
                "3e-known-limitation", " + ".join(parts),
                questions=len(snap["detected"]), answers=len(snap["completed"]),
                answer_calls=llm.started, detected=snap["detected"],
                note="Whisper terminating an open clause with '.' is trusted "
                     "as closure, so this costs a second question. Same input "
                     "without the period reassembles correctly (see 3c).",
            )
        finally:
            await h.close()
    settings.question_hold_incomplete_ms = 1
    settings.question_stabilization_ms = 1

    # Latency cost of the over-hold: how long does a COMPLETE-by-grammar short
    # request actually sit before the answer starts?
    print(chr(10) + "--- 3d. over-hold latency (real timing) ---")
    for text in (
        "When does it rebalance",
        "Are there tradeoffs",
        "Implement this",
        "Optimize this",
        "Explain this.",
        "Optimize it.",
        "What is Kafka",
    ):
        h = build(real_holds=True)
        try:
            started = time.monotonic()
            await say(h, text)
            await asyncio.sleep(0.2)
            quick = bool(snapshot(h)["detected"])
            await h.settle()
            await asyncio.sleep(0.05)
            await h.settle()
            waited_ms = int((time.monotonic() - started) * 1000)
            snap = snapshot(h)
            rec.add(
                "3d-over-hold", text, "answered within 200ms",
                "answered within 200ms" if quick else f"held ~{waited_ms}ms",
                answered=bool(snap["completed"]), waited_ms=waited_ms, **snap,
            )
        finally:
            await h.close()
    settings.question_hold_incomplete_ms = 1
    settings.question_stabilization_ms = 1


# ======================================================================= 5
async def section_5_imperatives() -> None:
    print("\n--- 5. imperative tasks end to end ---")
    for text, should_answer in [
        ("Explain Kafka", True),
        ("Explain Kafka replication", True),
        ("Show code", True),
        ("Show Python code", True),
        ("Show me the code", True),
        ("Implement this in Python", True),
        ("Optimize this query", True),
        ("Compare Kafka and RabbitMQ", True),
        ("Design a pipeline for millions of events", True),
    ]:
        h = build()
        try:
            await say(h, text)
            await h.settle()
            snap = snapshot(h)
            rec.add("5-e2e", text, f"answered={should_answer}",
                    f"answered={bool(snap['completed'])}", **snap)
        finally:
            await h.close()

    # A one-word imperative is rejected outright (below question_min_words)
    # and is NOT remembered as setup context -- so does its continuation
    # still reach an answer?
    print("\n--- 5b. one-word imperative then its object ---")
    for parts in (["Explain", "Kafka replication."], ["Optimize", "this query."]):
        h = build()
        try:
            t = time.monotonic()
            await say(h, parts[0], t)
            await say(h, parts[1], t + 0.5)
            await h.settle()
            snap = snapshot(h)
            rec.add(
                "5b-split-imperative", " + ".join(parts),
                "reaches an answer", "reaches an answer" if snap["completed"] else "lost",
                **snap,
            )
        finally:
            await h.close()


# ======================================================================= 8
async def section_8_contamination() -> None:
    """A 12-turn sequence with two hard topic switches. The prompt sent for a
    question after a switch must not carry the previous topic's history."""
    print("\n--- 8. context contamination ---")
    SEQUENCE = [
        ("Write a Python function for the longest substring.", reply(relationship="new_question", topic="longest substring")),
        ("What's the time complexity?", reply(relationship="follow_up", topic="longest substring", needs_previous_context=True, needs_previous_answer=True)),
        ("Can you optimize it?", reply(relationship="new_implementation", topic="longest substring", needs_previous_code=True)),
        ("Show me the implementation.", reply(relationship="new_implementation", topic="longest substring", needs_previous_code=True)),
        ("What's the memory complexity?", reply(relationship="follow_up", topic="longest substring", needs_previous_answer=True)),
        ("Explain the tradeoff.", reply(relationship="clarification", topic="longest substring", needs_previous_answer=True)),
        ("Now switch to Kafka.", reply(relationship="new_question", topic="kafka")),
        ("What is consumer rebalancing?", reply(relationship="new_question", topic="kafka")),
        ("How would you handle it?", reply(relationship="follow_up", topic="kafka", needs_previous_context=True)),
        ("What happens if a broker dies?", reply(relationship="follow_up", topic="kafka", needs_previous_context=True)),
        ("Now switch to SQL.", reply(relationship="new_question", topic="sql")),
        ("How would you optimize this query?", reply(relationship="new_question", topic="sql")),
    ]
    llm = SlowStreamingLLM(chunk_delay=0)
    completer = CountingCompleter(*[r for _, r in SEQUENCE])
    h = build(llm=llm, completer=completer)
    try:
        t = time.monotonic()
        for index, (text, _) in enumerate(SEQUENCE):
            await say(h, text, t + index * 5.0)
            await h.settle()
        snap = snapshot(h)
        rec.note("8-sequence", "12 turns", understanding_calls=completer.calls,
                 answer_calls=llm.started, **snap)

        # Contamination test: the prompt for the LAST question (SQL) must not
        # mention the Python problem or Kafka. Prompts are indexed by answer
        # call, one per turn that actually reached the LLM.
        prompts = llm.prompts
        rec.add(
            "8-prompt-count", "one prompt per answered turn",
            f"{len(snap['completed'])}", f"{len(prompts)}",
        )
        if prompts:
            last = prompts[-1].lower()
            rec.add("8-contamination", "SQL prompt free of 'longest substring'",
                    "absent", "absent" if "longest substring" not in last else "PRESENT",
                    prompt_tail=prompts[-1][-600:])
            rec.add("8-contamination", "SQL prompt free of 'rebalancing'",
                    "absent", "absent" if "rebalanc" not in last else "PRESENT")
            kafka_prompt = prompts[7].lower() if len(prompts) > 7 else ""
            rec.add("8-contamination", "Kafka prompt free of 'longest substring'",
                    "absent", "absent" if "longest substring" not in kafka_prompt else "PRESENT",
                    prompt_tail=prompts[7][-600:] if len(prompts) > 7 else "")
    finally:
        await h.close()


# ======================================================================= 9
async def section_9_pronouns() -> None:
    print("\n--- 9. pronoun / referent chains ---")
    SEQUENCE = [
        ("Explain Kafka consumer groups.", reply(relationship="new_question", topic="kafka consumer groups")),
        ("How does it work?", reply(relationship="follow_up", topic="kafka consumer groups", needs_previous_context=True)),
        ("What happens when they rebalance?", reply(relationship="follow_up", topic="kafka consumer groups", needs_previous_context=True)),
        ("How would you handle that?", reply(relationship="follow_up", topic="kafka consumer groups", needs_previous_context=True)),
        ("Why is this necessary?", reply(relationship="clarification", topic="kafka consumer groups", needs_previous_answer=True)),
        ("Can you optimize it?", reply(relationship="new_implementation", topic="kafka consumer groups")),
        ("Show me.", reply(relationship="continuation", topic="kafka consumer groups", needs_previous_code=True)),
        ("Now explain PostgreSQL replication.", reply(relationship="new_question", topic="postgresql replication")),
        ("How does it work?", reply(relationship="follow_up", topic="postgresql replication", needs_previous_context=True)),
    ]
    llm = SlowStreamingLLM(chunk_delay=0)
    completer = CountingCompleter(*[r for _, r in SEQUENCE])
    h = build(llm=llm, completer=completer)
    try:
        t = time.monotonic()
        for index, (text, _) in enumerate(SEQUENCE):
            await say(h, text, t + index * 5.0)
            await h.settle()
        snap = snapshot(h)
        rec.add(
            "9-chain", "9 utterances all reach an answer",
            "9 answers", f"{len(snap['completed'])} answers",
            detected=snap["detected"], rejected=snap["rejected"],
            understanding_calls=completer.calls, answer_calls=llm.started,
        )
        # The thread anchor must have moved to PostgreSQL once a genuinely new
        # question started a new thread.
        rec.add("9-thread-anchor", "anchor follows the topic switch",
                "postgresql",
                "postgresql" if "PostgreSQL" in (h.live._thread_anchor or "")
                else f"stale:{h.live._thread_anchor!r}")
        if len(llm.prompts) >= 9:
            last = llm.prompts[-1].lower()
            rec.add("9-contamination", "'How does it work?' after the switch",
                    "no kafka", "no kafka" if "consumer group" not in last else "KAFKA PRESENT",
                    prompt_tail=llm.prompts[-1][-600:])
    finally:
        await h.close()


# ====================================================================== 10
async def section_10_corrections() -> None:
    print("\n--- 10. correction attack ---")
    CASES = [
        ("Explain Kafka replication.", "Actually I mean PostgreSQL replication."),
        ("How would you design a batch pipeline?", "Actually make it streaming."),
        ("What database would you use?", "No, assume we must use PostgreSQL."),
    ]
    for first, correction in CASES:
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter(
            reply(relationship="new_question"),
            reply(relationship="correction", needs_previous_context=True),
        )
        h = build(llm=llm, completer=completer)
        try:
            t = time.monotonic()
            await say(h, first, t)
            await h.settle()
            await say(h, correction, t + 2.0)
            await h.settle()
            snap = snapshot(h)
            rec.add(
                "10-correction", f"{first!r} -> {correction!r}",
                "2 questions, 2 answers",
                f"{len(snap['detected'])} questions, {len(snap['completed'])} answers",
                understanding_calls=completer.calls, answer_calls=llm.started, **snap,
            )
            if len(llm.prompts) >= 2:
                rec.note("10-correction-prompt", correction,
                         prompt_tail=llm.prompts[-1][-800:])
        finally:
            await h.close()


# ====================================================================== 11
async def section_11_duplicates() -> None:
    print("\n--- 11. duplicate / race attack ---")
    Q = "How would you design a rate limiter?"

    async def run(name: str, script, expected_answers: int, **kw):
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter()
        h = build(llm=llm, completer=completer)
        try:
            await script(h)
            await h.settle()
            snap = snapshot(h)
            rec.add(
                "11-race", name,
                f"{expected_answers} answer(s)", f"{len(snap['completed'])} answer(s)",
                understanding_calls=completer.calls, answer_calls=llm.started,
                detected=snap["detected"], cancelled=snap["cancelled"], **kw,
            )
        finally:
            await h.close()

    async def same_final_twice(h):
        t = time.monotonic()
        await say(h, Q, t)
        await h.settle()
        await say(h, Q, t + 0.2)

    async def same_partial_twice(h):
        await h.live.on_transcript(Q, TranscriptSource.LOOPBACK, False)
        await h.live.on_transcript(Q, TranscriptSource.LOOPBACK, False)

    async def partial_final_late_partial(h):
        t = time.monotonic()
        await h.live.on_transcript(Q, TranscriptSource.LOOPBACK, False)
        await say(h, Q, t)
        await h.settle()
        await h.live.on_transcript(Q, TranscriptSource.LOOPBACK, False)

    async def out_of_order_finals(h):
        t = time.monotonic()
        await say(h, "How would you design a rate limiter?", t + 5.0)
        await h.settle()
        await say(h, "What is a semaphore?", t)  # earlier speech clock

    async def empty_final(h):
        await say(h, "")

    async def whitespace_final(h):
        await say(h, "   \t  ")

    async def stt_garbage(h):
        await say(h, "?? -- ...")

    async def garbage_then_question(h):
        t = time.monotonic()
        await say(h, "?? -- ...", t)
        await h.settle()
        await say(h, Q, t + 0.3)

    await run("same final twice", same_final_twice, 1)
    await run("same partial twice", same_partial_twice, 0)
    await run("partial -> final -> late partial", partial_final_late_partial, 1)
    await run("out-of-order finals", out_of_order_finals, 2)
    await run("empty final", empty_final, 0)
    await run("whitespace final", whitespace_final, 0)
    await run("STT garbage final", stt_garbage, 0)
    await run("garbage then real question", garbage_then_question, 1)


# ====================================================================== 12
async def section_12_rapid_fire() -> None:
    print("\n--- 12. rapid fire ---")
    POOL = [
        "Write a Python function that reverses a linked list.",
        "What is the time complexity of that?",
        "How would you optimize this query?",
        "Explain Kafka consumer rebalancing.",
        "What is a partition key?",
        "Design a data pipeline for a million events per second.",
        "How would you handle backpressure?",
        "What is the difference between a list and a tuple?",
        "Implement a rate limiter in Python.",
        "Actually make it a sliding window.",
        "Explain SQL window functions.",
        "How would you index this table?",
        "What is a CTE?",
        "Compare Kafka and RabbitMQ.",
        "How does exactly-once delivery work?",
        "Write a SQL query for the top five customers.",
        "What does EXPLAIN ANALYZE tell you?",
        "How would you shard PostgreSQL?",
        "Explain the GIL.",
        "What is a generator in Python?",
        "How would you test this?",
        "What is idempotency?",
        "Design a URL shortener.",
        "How would you scale it to a billion links?",
        "What database would you use?",
    ]
    for count in (20, 30, 50):
        llm = SlowStreamingLLM(chunk_delay=0)
        completer = CountingCompleter()
        h = build(llm=llm, completer=completer)
        try:
            t = time.monotonic()
            for index in range(count):
                await say(h, POOL[index % len(POOL)], t + index * 5.0)
                await h.settle()
            snap = snapshot(h)
            detected = len(snap["detected"])
            answered = len(snap["completed"])
            duplicates = detected - len(set(snap["detected"]))
            rec.add(
                "12-rapid-fire", f"{count} questions -- all detected",
                f"{count} detected", f"{detected} detected",
                answered=answered, cancelled=len(snap["cancelled"]),
                failed=len(snap["failed"]), rejected=snap["rejected"],
                repeated_texts=duplicates,
            )
            rec.add(
                "12-invariant", f"{count}: understanding calls <= detected",
                "within budget",
                "within budget" if completer.calls <= detected else "EXCEEDED",
                understanding_calls=completer.calls, detected=detected,
            )
            rec.add(
                "12-invariant", f"{count}: answer calls <= detected",
                "within budget",
                "within budget" if llm.started <= detected else "EXCEEDED",
                answer_calls=llm.started, detected=detected,
            )
        finally:
            await h.close()


# ====================================================================== 13
async def section_13_attachments() -> None:
    print("\n--- 13. attachment lifecycle ---")
    TABLE = "id,name\n1,a\n2,b"

    async def paste_then_ask(h, t):
        await h.live.on_context_attached("text", TABLE, "t.csv", now=t)
        await say(h, "What does this table show?", t + 1.0)

    async def ask_then_paste(h, t):
        # No settle: the paste has to arrive while the answer is in flight.
        # After ANSWER_COMPLETED, LiveSession clears `_live_question` and
        # deliberately does not re-ask -- that is the separate scenario below.
        await say(h, "What does this table show?", t)
        await asyncio.sleep(0.01)
        await h.live.on_context_attached("text", TABLE, "t.csv", now=t + 0.5)

    async def paste_mid_accumulation(h, t):
        await say(h, "How would you design a distributed", t)
        await h.live.on_context_attached("text", TABLE, "t.csv", now=t + 0.2)
        await say(h, "system for this table?", t + 0.5)

    async def stale_paste(h, t):
        stale = t - (settings.context_attachment_window_ms / 1000) - 30
        await h.live.on_context_attached("text", TABLE, "t.csv", now=stale)
        await say(h, "What does this table show?", t)

    async def two_pastes(h, t):
        await h.live.on_context_attached("text", TABLE, "a.csv", now=t)
        await h.live.on_context_attached("text", "SELECT 1;", "b.sql", now=t + 0.1)
        await say(h, "What do these show?", t + 1.0)

    async def attachment_only_referential(h, t):
        await h.live.on_context_attached("text", TABLE, "t.csv", now=t)
        await say(h, "Explain this.", t + 0.5)

    async def followup_after_paste(h, t):
        await h.live.on_context_attached("text", TABLE, "t.csv", now=t)
        await say(h, "What does this table show?", t + 0.5)
        await h.settle()
        await say(h, "Why?", t + 2.0)

    SCENARIOS = [
        ("fresh paste, then ask", paste_then_ask, True, reply(needs_attachments=True)),
        ("ask, then paste mid-answer (re-ask once)", ask_then_paste, True, reply(needs_attachments=True)),
        ("paste mid-accumulation", paste_mid_accumulation, True, reply(needs_attachments=True)),
        ("stale paste + needs_attachments=true", stale_paste, True, reply(needs_attachments=True)),
        ("stale paste + needs_attachments=false", stale_paste, False, reply(needs_attachments=False)),
        ("two pastes, one question", two_pastes, True, reply(needs_attachments=True)),
        ("attachment-only context ('Explain this.')", attachment_only_referential, True, reply(needs_attachments=True)),
        ("follow-up inherits the paste", followup_after_paste, True, reply(needs_attachments=True)),
    ]
    for name, script, expect_in_prompt, understanding in SCENARIOS:
        # Non-zero chunk delay so "mid-answer" is a real state, not a race.
        llm = SlowStreamingLLM(chunk_delay=0.02)
        completer = CountingCompleter(*[understanding] * 4)
        h = build(llm=llm, completer=completer)
        try:
            await script(h, time.monotonic())
            await h.settle()
            await asyncio.sleep(0.02)
            await h.settle()
            snap = snapshot(h)
            reached = any("id,name" in p or "SELECT 1" in p for p in llm.prompts)
            rec.add(
                "13-attachment", name,
                f"material_in_prompt={expect_in_prompt}",
                f"material_in_prompt={reached}",
                answers=len(snap["completed"]), detected=snap["detected"],
                answer_calls=llm.started, understanding_calls=completer.calls,
                cancelled=snap["cancelled"],
            )
        finally:
            await h.close()

    # A paste arriving AFTER the answer is delivered must not re-ask: that is
    # documented behaviour in LiveSession.on_context_attached.
    llm = SlowStreamingLLM(chunk_delay=0)
    h = build(llm=llm, completer=CountingCompleter())
    try:
        t = time.monotonic()
        await say(h, "What does this table show?", t)
        await h.settle()
        await h.live.on_context_attached("text", TABLE, "t.csv", now=t + 1.0)
        await h.settle()
        snap = snapshot(h)
        rec.add(
            "13-attachment", "paste after the answer completed -- no re-ask",
            "1 answer(s)", f"{len(snap['completed'])} answer(s)",
            answer_calls=llm.started, detected=snap["detected"],
        )
    finally:
        await h.close()


# ====================================================================== 18
async def section_18_concurrency() -> None:
    print("\n--- 18. concurrency / races ---")

    async def concurrent_finals():
        """Two finals dispatched with no await between them."""
        llm = SlowStreamingLLM(chunk_delay=0.005)
        h = build(llm=llm, completer=CountingCompleter())
        try:
            t = time.monotonic()
            await asyncio.gather(
                say(h, "How would you design a rate limiter?", t),
                say(h, "What is a semaphore?", t + 0.1),
            )
            await h.settle()
            snap = snapshot(h)
            rec.add(
                "18-concurrency", "two finals dispatched concurrently",
                "no exception, <=2 answers",
                "no exception, <=2 answers"
                if len(snap["completed"]) <= 2 else f"{len(snap['completed'])} answers",
                answer_calls=llm.started, **snap,
            )
        finally:
            await h.close()

    async def partial_and_final():
        llm = SlowStreamingLLM(chunk_delay=0.005)
        h = build(llm=llm, completer=CountingCompleter())
        try:
            t = time.monotonic()
            await asyncio.gather(
                h.live.on_transcript("How would you design", TranscriptSource.LOOPBACK, False),
                say(h, "How would you design a rate limiter?", t),
                h.live.on_transcript("partial again", TranscriptSource.LOOPBACK, False),
            )
            await h.settle()
            snap = snapshot(h)
            rec.add("18-concurrency", "partial + final interleaved",
                    "1 answer(s)", f"{len(snap['completed'])} answer(s)", **snap)
        finally:
            await h.close()

    async def attach_during_accumulation():
        llm = SlowStreamingLLM(chunk_delay=0.005)
        h = build(llm=llm, completer=CountingCompleter(reply(needs_attachments=True)))
        try:
            t = time.monotonic()
            await asyncio.gather(
                say(h, "How would you design a distributed", t),
                h.live.on_context_attached("text", "id,name\n1,a", "t.csv", now=t),
                say(h, "system for this data?", t + 0.3),
            )
            await h.settle()
            snap = snapshot(h)
            rec.add("18-concurrency", "accumulation + attachment concurrently",
                    ">=1 answer, no exception",
                    ">=1 answer, no exception" if snap["completed"] else "no answer",
                    answer_calls=llm.started, **snap)
        finally:
            await h.close()

    async def cancel_then_new_question():
        llm = SlowStreamingLLM(chunk_delay=0.05)
        h = build(llm=llm, completer=CountingCompleter())
        try:
            t = time.monotonic()
            await say(h, "How would you design a rate limiter?", t)
            await asyncio.sleep(0.01)
            await h.live.cancel()
            await say(h, "What is a semaphore?", t + 0.2)
            await h.settle()
            snap = snapshot(h)
            rec.add("18-concurrency", "cancellation then a new question",
                    "1 completed, 1 cancelled",
                    f"{len(snap['completed'])} completed, {len(snap['cancelled'])} cancelled",
                    answer_calls=llm.started, **snap)
        finally:
            await h.close()

    async def provider_failure_then_next():
        from tests.fakes import BrokenLLM
        h = build(llm=BrokenLLM(), completer=CountingCompleter())
        try:
            t = time.monotonic()
            await say(h, "How would you design a rate limiter?", t)
            await h.settle()
            await say(h, "What is a semaphore?", t + 5.0)
            await h.settle()
            snap = snapshot(h)
            rec.add("18-concurrency", "provider failure then the next question",
                    "2 failures, 2 detected",
                    f"{len(snap['failed'])} failures, "
                    f"{len(snap['detected'])} detected", **snap)
        finally:
            await h.close()

    async def task_leak_check():
        llm = SlowStreamingLLM(chunk_delay=0)
        h = build(llm=llm, completer=CountingCompleter())
        before = len(asyncio.all_tasks())
        try:
            t = time.monotonic()
            for index in range(10):
                await say(h, f"Explain topic number {index} in detail.", t + index * 5.0)
                await h.settle()
        finally:
            await h.close()
        await asyncio.sleep(0.05)
        after = len(asyncio.all_tasks())
        rec.add("18-task-leak", "10 turns leave no lingering tasks",
                "no growth", "no growth" if after <= before else f"+{after - before}",
                before=before, after=after)

    async def loop_blocking():
        """A heartbeat coroutine measures the worst gap the event loop ever
        goes unserviced while a burst of turns runs."""
        gaps: list[float] = []
        stop = False

        async def heartbeat():
            last = time.monotonic()
            while not stop:
                await asyncio.sleep(0.005)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        beat = asyncio.create_task(heartbeat())
        llm = SlowStreamingLLM(chunk_delay=0)
        h = build(llm=llm, completer=CountingCompleter())
        try:
            t = time.monotonic()
            for index in range(20):
                await say(h, f"Explain topic number {index} in detail.", t + index * 5.0)
                await h.settle()
        finally:
            await h.close()
        stop = True
        await asyncio.sleep(0.02)
        beat.cancel()
        worst = max(gaps) * 1000 if gaps else 0.0
        rec.add("18-loop-block", "worst event-loop stall during 20 turns",
                "under 250ms", "under 250ms" if worst < 250 else f"{worst:.0f}ms",
                worst_ms=round(worst, 1), samples=len(gaps))

    for scenario in (
        concurrent_finals, partial_and_final, attach_during_accumulation,
        cancel_then_new_question, provider_failure_then_next,
        task_leak_check, loop_blocking,
    ):
        await scenario()


async def main() -> None:
    await section_2_pauses()
    await section_2_fragment_ladder()
    await section_3_nested_clause()
    await section_5_imperatives()
    await section_8_contamination()
    await section_9_pronouns()
    await section_10_corrections()
    await section_11_duplicates()
    await section_12_rapid_fire()
    await section_13_attachments()
    await section_18_concurrency()
    rec.dump()


if __name__ == "__main__":
    asyncio.run(main())
