"""QA RED TEAM -- SECOND ADVERSARIAL PASS (mission section 46).

Five scenarios not exercised by probe_detector.py / probe_session.py /
probe_attachments.py:

  1. Correction-dedup end-to-end: does a later premise correction actually
     replace the earlier one in the answer prompt, or does the model see both?
  2. Paste arriving mid-accumulation (while a fragment is held, before its
     continuation lands) -- session.py's own docstring claims this binds for
     free; verify.
  3. Ordering guarantee under cancellation: no event for a superseded turn
     is ever emitted *after* the first event of the turn that superseded it.
  4. Duplicate-with-drift: a second "final" that is the first one plus a
     trailing addendum ("...? Actually never mind.") -- must not be treated
     as a plain merge-onto-self, and must not silently vanish either.
  5. Follow-up window boundary: a short follow-up delivered at exactly
     question_followup_window_ms, and one ms past it.

Run: backend/.venv/Scripts/python.exe qa_redteam/probe_second_pass.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.realtime.events import EventType  # noqa: E402
from app.realtime.question_detector import QuestionDetector  # noqa: E402
from app.realtime.question_understanding import QuestionUnderstander  # noqa: E402
from app.sessions.schemas import TranscriptSource  # noqa: E402
from tests.fakes import SlowStreamingLLM  # noqa: E402
from tests.replay_harness import ReplayEvent, ReplayHarness  # noqa: E402

results: list[dict] = []


def record(section: str, name: str, ok: bool, **detail) -> None:
    results.append({"section": section, "name": name, "ok": ok, **detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {section} :: {name}"
          + ("" if ok else f"  DETAIL={json.dumps(detail, default=str)}"))


class CountingCompleter:
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def complete_json(self, prompt: str, *, model: str, timeout_seconds: float) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else _reply()

    @property
    def calls(self) -> int:
        return len(self.prompts)


def _reply(**overrides) -> str:
    body = {
        "intent": "conceptual", "relationship": "new_question", "topic": "t",
        "domain": "d", "task": "task", "constraints": [], "requested_output": [],
        "entities": [], "needs_previous_context": False, "needs_previous_answer": False,
        "needs_previous_code": False, "needs_attachments": False, "confidence": 0.9,
    }
    body.update(overrides)
    return json.dumps(body)


async def scenario_1_correction_dedup() -> None:
    llm = SlowStreamingLLM(chunk_delay=0)
    h = ReplayHarness(llm=llm, stabilization_ms=1)
    await h.live.on_transcript(
        "Assume 1,000 QPS.", TranscriptSource.LOOPBACK, True, now=0.0
    )
    await h.settle()
    await h.live.on_transcript(
        "Actually, assume 10,000 QPS.", TranscriptSource.LOOPBACK, True, now=1.0
    )
    await h.settle()
    await h.live.on_transcript(
        "What's the bottleneck at that scale?", TranscriptSource.LOOPBACK, True, now=2.0
    )
    await h.settle()
    prompt = llm.prompts[-1] if llm.prompts else ""
    has_old = "1,000 QPS" in prompt
    has_new = "10,000 QPS" in prompt
    record(
        "SP1-correction_dedup", "later_premise_correction_replaces_earlier_in_prompt",
        has_new and not has_old,
        has_old_premise=has_old, has_new_premise=has_new,
        prompt_excerpt=prompt[:500],
    )
    await h.close()


async def scenario_2_paste_mid_accumulation() -> None:
    llm = SlowStreamingLLM(chunk_delay=0)
    # A generous real hold (well above the ~50ms this needs) so this isn't
    # flaky under system load -- the hold itself is real asyncio.sleep, and a
    # margin as tight as a bare 200ms was observed to occasionally lose the
    # race against ordinary scheduling/logging overhead on a busy machine.
    h = ReplayHarness(llm=llm, stabilization_ms=3000)
    await h.live.on_transcript(
        "How would you design a distributed", TranscriptSource.LOOPBACK, True, now=0.0
    )
    # A fragment this incomplete should be ACCUMULATING -- paste arrives
    # while the hold is open, before the continuation.
    await asyncio.sleep(0.05)
    await h.live.on_context_attached("text", content="CONSTRAINT: must be region-aware", now=0.05)
    await h.live.on_transcript(
        "system that processes millions of events per second?",
        TranscriptSource.LOOPBACK, True, now=0.3,
    )
    await h.settle(rounds=50)
    prompt = llm.prompts[-1] if llm.prompts else ""
    bound_for_free = "region-aware" in prompt
    completed = h.result.completed_turn_ids()
    record(
        "SP2-paste_mid_accumulation", "paste_during_hold_binds_to_eventual_merged_question",
        bound_for_free and len(completed) == 1,
        prompt_has_pasted_constraint=bound_for_free, completed_turn_ids=completed,
        note="session.py's own on_context_attached docstring claims: 'paste "
             "while a turn is accumulating -- same thing; the fragments "
             "keep assembling and the material is bound when they finish. "
             "Still no extra call.' Verifying that claim in isolation.",
    )
    await h.close()


async def scenario_3_cancellation_ordering() -> None:
    slow = SlowStreamingLLM(chunk_delay=0.05)
    h = ReplayHarness(llm=slow, stabilization_ms=1)
    await h.play(
        [
            ReplayEvent(0, "What is a binary search tree?"),
            ReplayEvent(10, "What is a Kafka consumer group?"),
        ],
        settle_between=False,
    )
    await h.settle()
    events = h.result.events
    # Find the first event belonging to the SECOND (superseding) turn, and
    # confirm no event for the FIRST turn appears after it.
    turn_ids_in_order = [e.turn_id for e in events if e.turn_id is not None]
    if len(set(turn_ids_in_order)) < 2:
        record(
            "SP3-cancellation_ordering", "no_stale_event_after_supersession", True,
            note="Only one turn_id observed (first turn cancelled before any "
                 "turn-scoped event was emitted) -- trivially satisfied.",
            turn_ids_in_order=turn_ids_in_order,
        )
    else:
        first_turn = turn_ids_in_order[0]
        second_turn = next(t for t in turn_ids_in_order if t != first_turn)
        first_index_of_second = turn_ids_in_order.index(second_turn)
        stale_after = any(
            t == first_turn for t in turn_ids_in_order[first_index_of_second:]
        )
        record(
            "SP3-cancellation_ordering", "no_stale_event_after_supersession",
            not stale_after,
            turn_ids_in_order=turn_ids_in_order,
            note="INVARIANT 8/14: once the superseding turn's first event is "
                 "on the wire, no later event may belong to the turn it "
                 "replaced.",
        )
    await h.close()


async def scenario_4_duplicate_with_drift() -> None:
    d = QuestionDetector()
    now = 0.0
    det1 = d.inspect("What is a hash map?", now=now, buffer_context=True)
    now += 1.0
    det2 = d.inspect(
        "What is a hash map? Actually never mind.", now=now, buffer_context=True
    )
    # This must not come back as "What is a hash map? What is a hash map?
    # Actually never mind." (self-referential double merge), and it must not
    # be silently swallowed either -- the interviewer explicitly said
    # "never mind", which is new information a real system should surface
    # somehow (even if only as a rejection/cancel signal).
    doubled = det2.accepted and det2.text.count("hash map?") >= 2
    record(
        "SP4-duplicate_with_drift", "final_A_then_finalA_plus_addendum",
        not doubled,
        det1_text=det1.text if det1.accepted else None,
        det2_accepted=det2.accepted, det2_text=det2.text,
        det2_supersedes=det2.supersedes,
        note="'What is a hash map? Actually never mind.' arriving as a "
             "second final shortly after the identical question was "
             "already accepted. Recording actual reconstruction; a "
             "silent 'never mind' is itself worth flagging even if not "
             "technically a duplicate-answer bug.",
    )


async def scenario_5_followup_window_boundary() -> None:
    window_ms = settings.question_followup_window_ms
    for label, offset_ms in (("at_boundary", 0), ("one_ms_past", 1)):
        d = QuestionDetector()
        d.inspect("What is eventual consistency?", now=0.0, buffer_context=True)
        now = (window_ms + offset_ms) / 1000
        det = d.inspect("Why?", now=now, buffer_context=True)
        record(
            "SP5-followup_window_boundary", f"why_{label}_{window_ms + offset_ms}ms",
            True,  # descriptive
            accepted=det.accepted, detail=det.detail,
            note="Recording actual accept/reject exactly at and 1ms past "
                 "question_followup_window_ms -- an off-by-one here would "
                 "silently flip a real follow-up into TOO_SHORT noise.",
        )


async def main() -> None:
    await scenario_1_correction_dedup()
    await scenario_2_paste_mid_accumulation()
    await scenario_3_cancellation_ordering()
    await scenario_4_duplicate_with_drift()
    await scenario_5_followup_window_boundary()

    print()
    failed = [r for r in results if not r["ok"]]
    print(f"TOTAL={len(results)} FAIL={len(failed)}")
    Path(__file__).with_name("probe_second_pass_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )


if __name__ == "__main__":
    asyncio.run(main())
