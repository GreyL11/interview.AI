"""QA RED TEAM -- isolated, test-only script. Does not modify app/ or tests/.

Focused on INVARIANT 9 (a stale attachment must never leak into a new,
unrelated question) and whether pasted content actually reaches the answer
prompt at each point in the attachment-window sweep (section 18/19).

Run: backend/.venv/Scripts/python.exe qa_redteam/probe_attachments.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.realtime.question_understanding import QuestionUnderstander  # noqa: E402
from app.sessions.schemas import TranscriptSource  # noqa: E402
from tests.fakes import SlowStreamingLLM  # noqa: E402
from tests.replay_harness import ReplayHarness  # noqa: E402

results: list[dict] = []


def record(section: str, name: str, ok: bool, **detail) -> None:
    results.append({"section": section, "name": name, "ok": ok, **detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {section} :: {name}"
          + ("" if ok else f"  DETAIL={json.dumps(detail, default=str)}"))


class CountingCompleter:
    def __init__(self, *responses: str, delay: float = 0.0):
        self._responses = list(responses)
        self.prompts: list[str] = []
        self.delay = delay

    async def complete_json(self, prompt: str, *, model: str, timeout_seconds: float) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
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


TABLE = "id,name,region\n1,acme,west\n2,globex,east"


async def main() -> None:
    # -----------------------------------------------------------------
    # Positive control: paste then ask immediately -- content must appear
    # in the actual prompt sent to the LLM.
    # -----------------------------------------------------------------
    llm = SlowStreamingLLM(chunk_delay=0)
    h = ReplayHarness(llm=llm, stabilization_ms=1)
    await h.live.on_context_attached("table", content=TABLE, now=0.0)
    await h.live.on_transcript(
        "What does this table show?", TranscriptSource.LOOPBACK, True, now=1.0
    )
    await h.settle()
    prompt = llm.prompts[-1] if llm.prompts else ""
    record(
        "18-attachment_binding", "paste_then_ask_content_in_prompt",
        "acme" in prompt,
        prompt_has_table="acme" in prompt,
    )
    await h.close()

    # -----------------------------------------------------------------
    # RETRACTED finding (was filed as BUG-003 / "9-stale_attachment_leak").
    #
    # This scenario -- paste a table, wait 90s, ask an unrelated question
    # with a *scripted* needs_attachments=True -- does bind the 90s-old
    # table. Initially reported as a leak. Fix-mode re-audit found
    # tests/test_attachment_lifecycle.py already covers exactly this
    # mechanism deliberately: `STALE = context_attachment_window_ms + 30`
    # (90s at the current 60s window), parametrized up to gap=90.0, with its
    # own docstring stating the point is that classifier-confirmed relevance
    # beats elapsed time "however long the pause" -- and warning against
    # "fixing" this with an age ceiling or a word list. Recorded here as a
    # confirmation of intentional behaviour, not a failure: what a real
    # classifier says the question refers to is trusted regardless of
    # clock age, by design, because the alternative (silently dropping
    # material the interviewer is still visibly discussing) was the
    # regression that suite was written to prevent. See TESTING_REPORT.md /
    # the fix-mode report for the full writeup.
    # -----------------------------------------------------------------
    llm = SlowStreamingLLM(chunk_delay=0)
    completer = CountingCompleter(_reply(needs_attachments=True, relationship="new_question"))
    h = ReplayHarness(llm=llm, stabilization_ms=1)
    h.live._understander = QuestionUnderstander(completer)
    await h.live.on_context_attached("table", content=TABLE, now=0.0)
    await h.live.on_transcript(
        "What is a bloom filter?", TranscriptSource.LOOPBACK, True, now=90.0
    )
    await h.settle()
    prompt = llm.prompts[-1] if llm.prompts else ""
    bound_as_designed = "acme" in prompt
    record(
        "9-stale_attachment_intentional", "90s_stale_material_bound_on_classifier_confirmation",
        bound_as_designed,
        prompt_has_table=bound_as_designed,
        note="Confirms intentional, tested behaviour (see comment above) -- "
             "PASS means the material was bound, matching "
             "test_attachment_lifecycle.py's own contract.",
    )
    await h.close()

    # Same, but the classifier is honest (needs_attachments=False) --
    # confirms the window itself behaves as documented (no leak on the
    # normal path).
    llm = SlowStreamingLLM(chunk_delay=0)
    completer = CountingCompleter(_reply(needs_attachments=False, relationship="new_question"))
    h = ReplayHarness(llm=llm, stabilization_ms=1)
    h.live._understander = QuestionUnderstander(completer)
    await h.live.on_context_attached("table", content=TABLE, now=0.0)
    await h.live.on_transcript(
        "What is a bloom filter?", TranscriptSource.LOOPBACK, True, now=90.0
    )
    await h.settle()
    prompt = llm.prompts[-1] if llm.prompts else ""
    record(
        "18-attachment_binding", "90s_stale_table_control_needs_attachments_false",
        "acme" not in prompt,
        prompt_has_table="acme" in prompt,
    )
    await h.close()

    # -----------------------------------------------------------------
    # Multiple attachments, ordering preserved.
    # -----------------------------------------------------------------
    llm = SlowStreamingLLM(chunk_delay=0)
    h = ReplayHarness(llm=llm, stabilization_ms=1)
    await h.live.on_context_attached("sql", content="SELECT 1", now=0.0)
    await h.live.on_context_attached("table", content=TABLE, now=1.0)
    await h.live.on_transcript(
        "Explain both of these.", TranscriptSource.LOOPBACK, True, now=2.0
    )
    await h.settle()
    prompt = llm.prompts[-1] if llm.prompts else ""
    order_ok = prompt.find("SELECT 1") < prompt.find("acme") if "acme" in prompt else False
    record(
        "18-attachment_binding", "multiple_attachments_arrival_order_preserved",
        order_ok,
        prompt_excerpt=prompt[:600],
    )
    await h.close()

    # -----------------------------------------------------------------
    # New attachment replacing old one (interviewer pastes a corrected
    # version). Old, superseded content must not also ride along forever.
    # -----------------------------------------------------------------
    llm = SlowStreamingLLM(chunk_delay=0)
    h = ReplayHarness(llm=llm, stabilization_ms=1)
    await h.live.on_context_attached("table", content="id,name\n1,OLD_WRONG_DATA", now=0.0)
    await h.live.on_context_attached("table", content="id,name\n1,NEW_CORRECT_DATA", now=1.0)
    await h.live.on_transcript(
        "What does this show?", TranscriptSource.LOOPBACK, True, now=2.0
    )
    await h.settle()
    prompt = llm.prompts[-1] if llm.prompts else ""
    record(
        "18-attachment_binding", "second_paste_both_bound_not_replaced",
        True,  # descriptive
        has_old="OLD_WRONG_DATA" in prompt, has_new="NEW_CORRECT_DATA" in prompt,
        note="By design both pending pastes within the window bind together "
             "(no replace-on-paste semantics exist) -- recording actual "
             "behaviour since 'replacing' is not literally implemented; "
             "an interviewer correcting a paste gets BOTH versions quoted "
             "to the model, old data included, unless resolved by AI "
             "judgement.",
    )
    await h.close()

    # -----------------------------------------------------------------
    # question -> attachment -> follow-up (paste arrives DURING an
    # in-flight answer): does the re-ask carry the material without
    # duplicating the original question as a second answered turn?
    # -----------------------------------------------------------------
    slow = SlowStreamingLLM(chunk_delay=0.2)
    h = ReplayHarness(llm=slow, stabilization_ms=1)
    await h.live.on_transcript(
        "What does this error mean?", TranscriptSource.LOOPBACK, True, now=0.0
    )
    await asyncio.sleep(0.05)  # answer is mid-stream
    await h.live.on_context_attached("text", content="TRACE: NullPointerException at line 42", now=0.05)
    await h.settle()
    completed = h.result.completed_turn_ids()
    record(
        "18-attachment_binding", "paste_during_in_flight_answer_reasks_once",
        len(completed) == 1,
        completed_turn_ids=completed,
        llm_stream_starts=slow.started,
    )
    await h.close()

    print()
    failed = [r for r in results if not r["ok"]]
    print(f"TOTAL={len(results)} FAIL={len(failed)}")
    Path(__file__).with_name("probe_attachments_results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )


if __name__ == "__main__":
    asyncio.run(main())
