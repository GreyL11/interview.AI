"""QA RED TEAM -- isolated, test-only script. Does not modify app/ or tests/.

Drives the REAL app.realtime.question_detector.QuestionDetector and
app.realtime.prompt_detector directly (no LLM, no asyncio) through the
fragment-reconstruction / finality / pause-attack / duplicate / chatter
matrices (report sections 4, 5, 6, 9, 15, 16, 24).

Run: backend/.venv/Scripts/python.exe qa_redteam/probe_detector.py
Prints one line per scenario: PASS/FAIL, plus a JSON dump for FAIL/interesting
cases. Never raises on an individual scenario -- collects all outcomes so one
early bug does not hide the rest.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.realtime.question_detector import Finality, QuestionDetector  # noqa: E402

results: list[dict] = []


def record(section: str, name: str, ok: bool, **detail) -> None:
    results.append({"section": section, "name": name, "ok": ok, **detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {section} :: {name}"
          + ("" if ok else f"  DETAIL={json.dumps(detail, default=str)}"))


def new_detector(**kw) -> QuestionDetector:
    return QuestionDetector(**kw)


# ===================================================================
# Section 4: question reconstruction -- fragment then continuation.
# INVARIANT 1: an incomplete question must never reach answer generation.
# ===================================================================

FRAGMENT_CASES = [
    ("A", "How would you", "design a distributed system?"),
    ("B", "How would you design", "a distributed system?"),
    ("C", "How would you design a", "distributed system?"),
    ("D", "How would you design a distributed", "system?"),
    ("E", "What's the", "time complexity?"),
    ("F", "Can you", "optimize it?"),
    ("G", "How would you", "handle database replication?"),
    ("H", "What happens if", "Kafka goes down?"),
]

for label, frag1, frag2 in FRAGMENT_CASES:
    d = new_detector()
    now = 1000.0
    det1 = d.inspect(frag1, now=now, buffer_context=True)
    # INVARIANT 1 check: the bare fragment alone must never be `accepted and
    # stable` (i.e. must never be ready to answer as-is).
    fragment_would_answer = det1.accepted and det1.stable
    now += 0.4  # 400ms pause, realistic mid-utterance gap
    det2 = d.inspect(f"{frag1} {frag2}", now=now, buffer_context=True, accumulating=True)
    full_accepted_and_stable = det2.accepted and det2.stable
    ok = (not fragment_would_answer) and full_accepted_and_stable
    record(
        "4-reconstruction", f"case_{label}", ok,
        frag1=frag1, det1_accepted=det1.accepted, det1_stable=det1.stable,
        det1_finality=det1.finality.value if det1.accepted else None,
        combined_accepted=det2.accepted, combined_stable=det2.stable,
        combined_text=det2.text,
    )


# ===================================================================
# Section 5: pause attack. Continuation delivered after a real speech gap
# of 100ms/300ms/500ms/1s/2s/3s/4s/5s. `accumulating=True` models the
# session having kept the hold open (LiveSession._await_speech_end would do
# this in production as long as speech_in_progress); this probes the
# detector's own merge-window arithmetic in isolation.
# ===================================================================

PAUSE_MS = [100, 300, 500, 1000, 2000, 3000, 4000, 5000]
for pause_ms in PAUSE_MS:
    d = new_detector()
    now = 2000.0
    det1 = d.inspect("How would you design a distributed", now=now, buffer_context=True)
    now += pause_ms / 1000
    det2 = d.inspect(
        "system that processes millions of events per second?",
        now=now, buffer_context=True, accumulating=True,
    )
    ok = det2.accepted and det2.stable and "distributed" in det2.text
    record(
        "5-pause_attack", f"pause_{pause_ms}ms", ok,
        det1_finality=det1.finality.value if det1.accepted else "rejected",
        det2_accepted=det2.accepted, det2_text=det2.text,
    )

# NOTE: an earlier version of this probe also swept the same gaps with
# `accumulating=False` at the bare QuestionDetector level, modelling a caller
# that has forgotten a hold is live. That contract violation never happens in
# production -- LiveSession.consider always passes the real `accumulating`
# state, tracked via `_pending_ask` -- so failures under it were an artifact
# of the probe, not evidence of a product bug. The real, meaningful version of
# this check is end-to-end through LiveSession itself, with unmodified hold
# timings: see qa_redteam/probe_session.py's
# `split_becomes_two_answered_turns_at_2500ms_gap`.


# ===================================================================
# Section 6: finality attack -- punctuation is not the sole finality signal.
# ===================================================================

FINALITY_CASES = [
    "What is Kafka?",
    "What is Kafka",
    "Why?",
    "How?",
    "How does that work?",
    "Can you explain?",
    "Can you?",
    "Show me.",
    "Explain.",
    "Answer this.",
    "Describe the architecture.",
    "Walk me through the implementation.",
    "Tell me what happens if Kafka fails.",
    "Now handle retries.",
    "Now optimize this.",
]

for text in FINALITY_CASES:
    d = new_detector()
    # Seed a recent question so follow-up-eligible short fragments ("Why?",
    # "How?") get a fair chance, matching how they'd actually arrive live.
    d.inspect("What is Kafka?", now=100.0, buffer_context=True)
    det = d.inspect(text, now=100.5, buffer_context=True)
    record(
        "6-finality", text, True,  # descriptive, not pass/fail -- see report
        accepted=det.accepted,
        finality=det.finality.value if det.accepted else None,
        reason=det.reason.value if det.reason else None,
        detail=det.detail,
    )


# ===================================================================
# Section 9: short follow-up attack. HIGH RISK per the brief.
# ===================================================================

SHORT_FOLLOWUPS = [
    "Why?", "How?", "Then?", "Then what?", "Why not?", "How so?", "Really?",
    "Which one?", "What about it?", "And?", "Also?", "Can you?",
    "Can you explain?", "Show me.", "Implementation?", "Complexity?",
    "Tradeoffs?", "Example?", "More detail.", "Go deeper.",
]

for text in SHORT_FOLLOWUPS:
    d = new_detector()
    d.inspect("What is the time complexity of merge sort?", now=200.0, buffer_context=True)
    det = d.inspect(text, now=205.0, buffer_context=True)  # 5s later, plausible follow-up
    record(
        "9-short_followup", text, True,
        accepted=det.accepted, detail=det.detail,
        finality=det.finality.value if det.accepted else None,
        reason=det.reason.value if det.reason else None,
    )

# Same set, but with NO prior question at all -- these must be rejected
# (nothing to follow up on), never answered as their own question.
for text in SHORT_FOLLOWUPS:
    d = new_detector()
    det = d.inspect(text, now=1.0, buffer_context=True)
    bad = det.accepted  # a bare "Why?" with zero prior context must not fire
    record(
        "9-short_followup_no_context", text, not bad,
        accepted=det.accepted, detail=det.detail,
    )


# ===================================================================
# Section 15: this/that/it/they referent chain -- detector-level check that
# each of these still gets *detected* (referent resolution itself is the
# understanding layer's job, exercised in probe_understanding.py).
# ===================================================================

REFERENT_CHAIN = [
    "What is Kafka?",
    "How does it scale?",
    "Why is that useful?",
    "What happens when they fail?",
    "Can you optimize it?",
    "How would you implement that?",
    "What's its bottleneck?",
]

d = new_detector()
now = 5000.0
chain_ok = True
chain_detail = []
for text in REFERENT_CHAIN:
    det = d.inspect(text, now=now, buffer_context=True)
    chain_detail.append({"text": text, "accepted": det.accepted, "detail": det.detail})
    if not det.accepted:
        chain_ok = False
    now += 4.0
record("15-referent_chain", "this_that_it_they", chain_ok, chain=chain_detail)


# ===================================================================
# Section 16: duplicate / replay attack.
# INVARIANT 6: duplicate transcript events must not produce duplicate answers.
# ===================================================================

# partial A, partial A, partial A, final A -- partials never reach the
# detector at all in production (LiveSession.on_transcript short-circuits
# non-final text), so at the detector level the only thing to probe is
# repeated *finals* of the same text.
d = new_detector()
now = 6000.0
det_a1 = d.inspect("How would you design a URL shortener?", now=now, buffer_context=True)
now += 1.0
det_a2 = d.inspect("How would you design a URL shortener?", now=now, buffer_context=True)
# The repeat must not reach the model at all, and must certainly not be
# concatenated onto itself ("...shortener? How would you design a URL
# shortener?"). Updated in fix pass 2: this row previously required
# `det_a2.accepted`, which is the behaviour HC-003 identified as the bug --
# accepting the copy opened a second turn and spent a second understanding
# call and a second answer call. The note below already anticipated the
# alternative; dedupe is now the contract, and it subsumes the old assertion
# because text that never reaches extraction cannot double up.
ok = (
    not det_a2.accepted
    and det_a2.detail == "duplicate_redelivery"
    and not det_a2.supersedes
)
record(
    "16-duplicate_final", "final_A_final_A", ok,
    det_a2_accepted=det_a2.accepted, det_a2_detail=det_a2.detail,
    det_a2_supersedes=det_a2.supersedes,
    note="a second identical final inside question_redelivery_ms is a "
         "redelivery of the live turn and is dropped; the same text asked "
         "again later is answered (see same_question_twice_30s_apart)",
)

# same question twice, well separated (a candidate re-asks, or interviewer
# repeats verbatim after no answer was heard) -- must still detect cleanly.
d = new_detector()
now = 7000.0
det_b1 = d.inspect("What is a hash map?", now=now, buffer_context=True)
now += 30.0
det_b2 = d.inspect("What is a hash map?", now=now, buffer_context=True)
record(
    "16-duplicate_final", "same_question_twice_30s_apart", det_b2.accepted,
    det_b2_text=det_b2.text, det_b2_supersedes=det_b2.supersedes,
)

# out-of-order finals: an older, shorter utterance's final delivered AFTER a
# newer, longer one already landed (STT queue reordering under load).
d = new_detector()
t_new = 8010.0
t_old = 8005.0  # earlier speech, later delivery
det_new = d.inspect("How would you shard this database?", now=t_new, buffer_context=True)
det_old_late = d.inspect("Okay", now=t_old, buffer_context=True)  # now < last_accepted_at
ok = not det_old_late.accepted or not det_old_late.supersedes
record(
    "17-race_out_of_order", "late_older_final_after_newer_final", ok,
    det_new_text=det_new.text,
    det_old_late_accepted=det_old_late.accepted,
    det_old_late_supersedes=det_old_late.supersedes,
    note="INVARIANT 7: an out-of-order (clock-earlier) final must not merge "
         "onto / corrupt the already-accepted newer question",
)


# ===================================================================
# Section 24: interviewer chatter must never trigger answer generation.
# ===================================================================

CHATTER = [
    "Okay.", "Alright.", "Sounds good.", "Sure.", "Interesting.", "Right.",
    "Good.", "Let's move on.", "Next question.", "Let me think.", "Hmm.",
    "Okay, great.",
]

for text in CHATTER:
    d = new_detector()
    d.inspect("What is a load balancer?", now=9000.0, buffer_context=True)
    det = d.inspect(text, now=9002.0, buffer_context=True)
    ok = not det.accepted
    record("24-chatter", text, ok, accepted=det.accepted, detail=det.detail)


# ===================================================================
# Summary
# ===================================================================

failed = [r for r in results if not r["ok"]]
print()
print(f"TOTAL={len(results)} FAIL={len(failed)}")
Path(__file__).with_name("probe_detector_results.json").write_text(
    json.dumps(results, indent=2, default=str)
)
