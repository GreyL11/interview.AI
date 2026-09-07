"""HARD-CASE ADVERSARIAL PASS -- detector layer (report sections 2-7, 17).

Drives the REAL QuestionDetector / prompt_detector. No production code
touched. Verdicts are one of:

    COMPLETE -- accepted, Finality.COMPLETE: goes to the LLM immediately
    HOLD     -- accepted, ACCUMULATING/POTENTIALLY_COMPLETE: waits for more
    REJECT   -- not accepted at all

Fires no LLM calls: `classify` is the deterministic phase-1 classifier.
"""

import itertools

from hc_common import Recorder  # noqa: E402  (path bootstrap lives there)

from app.realtime.prompt_detector import (  # noqa: E402
    is_navigation_chatter,
    is_referentially_empty,
)
from app.realtime.question_detector import Finality, QuestionDetector  # noqa: E402

rec = Recorder("hc_detector")

COMPLETE, HOLD, REJECT = "COMPLETE", "HOLD", "REJECT"


def verdict(
    text: str,
    *,
    prior: str | None = None,
    pending_context: bool = False,
    now: float = 100.0,
) -> tuple[str, dict]:
    """Fresh detector, one utterance. `prior` seeds a recently-accepted
    question 1s earlier, so follow-up/merge behaviour can be exercised."""
    det = QuestionDetector()
    if prior is not None:
        det.inspect(prior, now - 1.0)
    d = det.inspect(text, now, has_pending_context=pending_context)
    label = (
        REJECT if not d.accepted
        else COMPLETE if d.finality is Finality.COMPLETE
        else HOLD
    )
    return label, {
        "detail": d.detail,
        "finality": d.finality.value,
        "hold_ms": d.hold_ms,
        "prompt": d.text,
        "effective": d.effective_text,
    }


def check(section: str, text: str, expected: str, **kw) -> None:
    label, detail = verdict(text, **kw)
    rec.add(section, text, expected, label, **detail)


# ===================================================================== 2
# Question finality: growing fragments of one question.
print("\n--- 2. question finality: growing fragments ---")
GROWTH = [
    ("How would you", HOLD),
    ("How would you design", HOLD),
    ("How would you design a", HOLD),
    ("How would you design a distributed", HOLD),
    # A whole request by grammar. Genuinely complete on its own -- the
    # continuation test below is what proves the split case reassembles.
    ("How would you design a distributed system", COMPLETE),
]
for text, expected in GROWTH:
    check("2-growth", text, expected)

print("\n--- 2. continuations, merged onto the held fragment ---")
# Merge path: the fragment is in flight (accumulating=True widens the window
# to question_max_accumulation_ms), then the continuation arrives.
for cont, expected in [
    ("system?", COMPLETE),
    ("system that processes millions of events", COMPLETE),
    ("system that processes millions of events per second?", COMPLETE),
]:
    det = QuestionDetector()
    first = det.inspect("How would you design a distributed", 100.0)
    second = det.inspect("How would you design a distributed", 100.0, accumulating=True) \
        if False else det.inspect(cont, 100.5, accumulating=True)
    label = (
        REJECT if not second.accepted
        else COMPLETE if second.finality is Finality.COMPLETE else HOLD
    )
    rec.add(
        "2-merge", f"[held] + {cont!r}", expected, label,
        merged=second.supersedes, prompt=second.text,
        first_finality=first.finality.value, detail=second.detail,
    )

# ===================================================================== 3
# Nested-clause attack: a subordinate wh-clause with no verb of its own.
print("\n--- 3. nested clause ---")
NESTED = [
    ("Explain how Kafka handles", HOLD),
    ("Tell me how the database", HOLD),
    ("Describe how the system", HOLD),
    ("How would you ensure", HOLD),
    ("How would you ensure that the service", HOLD),
    ("Can you explain why", HOLD),
    ("Can you explain why the consumer", HOLD),
]
for text, expected in NESTED:
    check("3-nested", text, expected)

print("\n--- 3. nested clause, reassembled ---")
NESTED_PAIRS = [
    (["Explain how Kafka handles", "partitions?"], COMPLETE),
    (["Explain how Kafka handles", "partitions when brokers fail?"], COMPLETE),
    (["Tell me how the database", "handles replication?"], COMPLETE),
    (["Describe how the system", "should recover from failure?"], COMPLETE),
    (["How would you ensure", "that the service", "does not lose data?"], COMPLETE),
    (["Can you explain why", "the consumer", "falls behind?"], COMPLETE),
]
for parts, expected in NESTED_PAIRS:
    det = QuestionDetector()
    steps = []
    last = None
    t = 100.0
    for index, part in enumerate(parts):
        last = det.inspect(part, t, accumulating=index > 0)
        steps.append({
            "text": part, "accepted": last.accepted,
            "finality": last.finality.value, "merged": last.supersedes,
            "prompt": last.text,
        })
        t += 0.5
    label = (
        REJECT if not last.accepted
        else COMPLETE if last.finality is Finality.COMPLETE else HOLD
    )
    rec.add(
        "3-reassembly", " + ".join(repr(p) for p in parts), expected, label,
        final_prompt=last.text, steps=steps,
    )

# ===================================================================== 4
# Interrogative attack -- every interrogative/auxiliary opener, three lengths.
print("\n--- 4. interrogative ladder ---")
LADDER = [
    (["How would you", "How would you handle", "How would you handle retries?"],
     [HOLD, HOLD, COMPLETE]),
    (["What would you", "What would you use", "What would you use for storage?"],
     [HOLD, HOLD, COMPLETE]),
    (["Why does", "Why does Kafka", "Why does Kafka rebalance?"],
     [HOLD, HOLD, COMPLETE]),
    (["Can you", "Can you explain", "Can you explain replication?"],
     [HOLD, HOLD, COMPLETE]),
    (["Could you", "Could you show", "Could you show the implementation?"],
     [HOLD, HOLD, COMPLETE]),
    # "Would you choose Kafka" is a whole yes/no question -- see the note in
    # HARD_CASE_E2E_REPORT.md section 4, where this label was already
    # identified as an expectation error rather than a product failure.
    (["Would you", "Would you choose Kafka", "Would you choose Kafka for this?"],
     [HOLD, COMPLETE, COMPLETE]),
    (["When does", "When does it rebalance", "When does it rebalance?"],
     [HOLD, COMPLETE, COMPLETE]),
    (["Where would you", "Where would you store", "Where would you store the offsets?"],
     [HOLD, HOLD, COMPLETE]),
    (["Which one", "Which one would you", "Which one would you pick for this?"],
     [HOLD, HOLD, COMPLETE]),
    (["Who owns", "Who owns the schema", "Who owns the schema registry?"],
     [HOLD, COMPLETE, COMPLETE]),
    (["Should we", "Should we shard", "Should we shard the table?"],
     [HOLD, HOLD, COMPLETE]),
    # "Do you use" keeps HOLD: a second-person subject with a bare verb is the
    # interviewer asking the candidate to do something, and the thing is
    # missing. The three below flip to COMPLETE because a *third*-person
    # subject with a bare verb is a whole question with no object at all --
    # which is the shape HC-002 names in its SHOULD-COMPLETE-IMMEDIATELY
    # list ("When does it rebalance", "Are there tradeoffs", "Is it durable").
    (["Do you", "Do you use", "Do you use Kafka here?"],
     [HOLD, HOLD, COMPLETE]),
    (["Does it", "Does it scale", "Does it scale to a million events?"],
     [HOLD, COMPLETE, COMPLETE]),
    (["Is it", "Is it durable", "Is it durable by default?"],
     [HOLD, COMPLETE, COMPLETE]),
    (["Are there", "Are there tradeoffs", "Are there tradeoffs with this?"],
     [HOLD, COMPLETE, COMPLETE]),
    (["Will it", "Will it survive", "Will it survive a broker loss?"],
     [HOLD, COMPLETE, COMPLETE]),
]
for parts, expectations in LADDER:
    for part, expected in zip(parts, expectations):
        check("4-interrogative", part, expected)

# ===================================================================== 5
# Imperative / task attack -- the interrogative fix must not have broken these.
print("\n--- 5. imperative / task ---")
IMPERATIVE = [
    # Rejected, not held: below `question_min_words`. HC-005 requires that it
    # never produces an answer on its own, and that its continuation still
    # reaches one -- the second half is asserted end to end in hc_session.py.
    ("Explain", REJECT),
    ("Explain Kafka", COMPLETE),
    ("Explain Kafka replication", COMPLETE),
    ("Show", REJECT),          # one word, below question_min_words
    ("Show code", COMPLETE),
    ("Show Python code", COMPLETE),
    ("Implement", REJECT),
    ("Implement this", COMPLETE),
    ("Implement this in Python", COMPLETE),
    ("Optimize", REJECT),
    ("Optimize this", COMPLETE),
    ("Optimize this query", COMPLETE),
    ("Compare", REJECT),
    ("Compare Kafka and RabbitMQ", COMPLETE),
    ("Design", REJECT),
    ("Design a pipeline", COMPLETE),
    ("Design a pipeline for millions of events", COMPLETE),
]
for text, expected in IMPERATIVE:
    check("5-imperative", text, expected)

# ===================================================================== 6
# Chatter attack -- pure navigation rejected, attached questions answered.
print("\n--- 6. chatter ---")
NAVIGATION = [
    "Next question.",
    "Okay, next question.",
    "Alright, next question.",
    "Let's move to the next question.",
    "Moving on.",
    "Okay, moving on.",
    "One more question.",
    "Last question.",
    "Final question.",
    "We'll go to the next question.",
    "Let's do another question.",
]
for text in NAVIGATION:
    check("6-navigation", text, REJECT)
    rec.note("6-navigation-helper", text, is_navigation_chatter=is_navigation_chatter(text))

ATTACHED = [
    "Next question: what is Kafka?",
    "Next question, what is Kafka?",
    "Okay, next question - explain Kafka.",
    "Okay, next question — explain Kafka.",
    "Here's the next question: how does Kafka work?",
    "One more question: would you use Kafka?",
]
for text in ATTACHED:
    check("6-attached", text, COMPLETE)

# ===================================================================== 7
# Referential fragments, in all three context states.
print("\n--- 7. referential fragments ---")
REFERENTIAL = [
    # text, zero-context, with-previous-question, with-pending-attachment
    ("Which one?", REJECT, COMPLETE, COMPLETE),
    ("Which?", REJECT, COMPLETE, COMPLETE),
    ("Can you?", REJECT, COMPLETE, COMPLETE),
    ("Can you explain?", REJECT, COMPLETE, COMPLETE),
    ("Show me.", REJECT, COMPLETE, COMPLETE),
    ("Show me?", REJECT, COMPLETE, COMPLETE),
    ("Then what?", REJECT, COMPLETE, COMPLETE),
    ("Why not?", REJECT, COMPLETE, COMPLETE),
    ("How so?", REJECT, COMPLETE, COMPLETE),
    ("What about it?", REJECT, COMPLETE, COMPLETE),
    # "this" is a governed object-standin, so these name a topic that can
    # exist on screen with no prior turn -- see is_referentially_empty.
    ("What about this?", REJECT, COMPLETE, COMPLETE),
    ("Explain this.", COMPLETE, COMPLETE, COMPLETE),
    ("Optimize it.", COMPLETE, COMPLETE, COMPLETE),
    # Genuinely self-contained short requests: must be unaffected by context.
    ("What is Kafka?", COMPLETE, COMPLETE, COMPLETE),
    ("Explain Kafka.", COMPLETE, COMPLETE, COMPLETE),
]
PRIOR = "How would you design a rate limiter?"
for text, zero, with_prior, with_attachment in REFERENTIAL:
    check("7-zero-context", text, zero)
    check("7-prior-question", text, with_prior, prior=PRIOR)
    check("7-pending-attachment", text, with_attachment, pending_context=True)
    rec.note("7-helper", text, is_referentially_empty=is_referentially_empty(text))

# ===================================================================== 17
# Property-style fuzzing. Two invariants, no hand-labelling:
#   I1  a fragment ending on a dangling function word is never COMPLETE
#   I2  a question Whisper terminated with "?" is never left ACCUMULATING
print("\n--- 17. property fuzzing ---")

OPENERS = [
    "How would you", "What would you", "Why does", "Can you", "Could you",
    "Would you", "Should we", "Do you", "Does it", "Is it", "Are there",
    "Will it", "When does", "Where would you", "Which one", "Who owns",
    "Explain", "Describe", "Show me", "Tell me", "Design", "Implement",
    "Optimize", "Compare", "Walk me through",
]
VERBS = ["design", "handle", "scale", "optimize", "explain", "store", "choose", ""]
DETERMINERS = ["a", "the", "this", ""]
MODIFIERS = ["distributed", "streaming", "partitioned", "large", ""]
NOUNS = ["system", "pipeline", "queue", "table", "service", ""]
TAILS = ["", "that", "if", "with", "for", "because", "when"]

DANGLING = {
    "if", "that", "because", "so", "and", "but", "or", "nor", "the", "a", "an",
    "of", "to", "is", "are", "was", "were", "be", "been", "with", "for", "in",
    "on", "at", "as", "than", "while", "though", "although", "since", "unless",
    "until", "whether", "what", "how", "why", "when", "where", "which", "who",
    "whom", "about",
}

fuzz_cases = 0
i1_violations = 0
i2_violations = 0
exceptions = 0
for opener, verb, det_word, modifier, noun, tail in itertools.product(
    OPENERS, VERBS, DETERMINERS, MODIFIERS, NOUNS, TAILS
):
    parts = [p for p in (opener, verb, det_word, modifier, noun, tail) if p]
    text = " ".join(parts)
    words = text.split()
    if len(words) < 2:
        continue
    fuzz_cases += 1
    try:
        unpunctuated, detail = verdict(text)
        punctuated, pdetail = verdict(text + "?")
    except Exception as exc:  # noqa: BLE001 -- an exception here IS the finding
        exceptions += 1
        rec.add("17-exception", text, "no-exception", type(exc).__name__, error=str(exc))
        continue
    # I1: trailing function word means the clause has not closed.
    if words[-1].lower() in DANGLING and unpunctuated == COMPLETE:
        i1_violations += 1
        if i1_violations <= 25:  # minimal reproductions, not the whole cross product
            rec.add("17-I1-dangling-complete", text, HOLD, COMPLETE, **detail)
    # I2: Whisper's own "?" is the strongest closure signal in the pipeline.
    if punctuated == HOLD:
        i2_violations += 1
        if i2_violations <= 25:
            rec.add("17-I2-punctuated-held", text + "?", COMPLETE, HOLD, **pdetail)

rec.note(
    "17-fuzz-summary", "cross product",
    cases=fuzz_cases, i1_violations=i1_violations, i2_violations=i2_violations,
    exceptions=exceptions,
)
print(f"  fuzzed {fuzz_cases} utterances (x2 punctuation variants): "
      f"I1={i1_violations} I2={i2_violations} exceptions={exceptions}")

rec.dump()
