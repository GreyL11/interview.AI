# FIX_MODE_2_REPORT.md — Interview Coach fix pass #2

**Date:** 2026-09-07
**Branch:** `feat/desktop-app-and-llm-routing`
**HEAD:** `c72055d Harden realtime context and retrieval`
**Authority for this pass:** `HARD_CASE_E2E_REPORT.md`
**Mode:** fix only. **Nothing was committed. Nothing was pushed.**

---

## 1. Baseline

### Before

| | |
|---|---|
| Backend suite | **1366 passed, 2 failed, 3 skipped, 1 warning** |
| The 2 failures | `test_secret_storage.py::test_the_backend_is_resolved_directly_when_discovery_finds_nothing`, `test_secret_storage.py::test_a_machine_with_no_backend_at_all_reports_unavailable` — environment-only (`keyring` absent) |
| Flaky | `test_stt_pipeline.py::test_final_diverging_from_its_partial_is_logged` |
| Hard-case | 414 judged checks, 366 pass, **48 fail** |
| Property fuzz | 335,988 detector evaluations, 0 invariant violations, 0 exceptions |

### After

| | |
|---|---|
| Backend suite | **1714 passed, 0 failed, 3 skipped, 1 warning** — two consecutive full runs (174.9s, 164.1s) |
| Red-team suite | **127 checks, 0 fail** (5 probes) |
| Hard-case suite | **402 judged checks, 0 fail**, 51 observations |
| Property fuzz | **335,988 evaluations, 0 invariant violations, 0 exceptions** — unchanged |
| Frontend | typecheck clean, **125/125 tests pass** (no frontend files changed) |

The +348 passing tests are +2 from ENV-001 unblocking two previously-failing
tests and +346 from tests added in this pass. The previously flaky
`test_final_diverging_from_its_partial_is_logged` passed in both full runs;
nothing was done to it, so it should still be treated as timing-sensitive.

The one warning is unchanged and unrelated:
`StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated`.

---

## 2. HC-002 — P1 — ordinary short questions held ~3.1s

### Root cause

Two independent causes, both introduced by the BUG-001 fix in pass #1, plus a
third that fell out of the same audit.

**(a) A word count cannot answer a structural question.**
`_has_content_after` required **two** substantive words after an interrogative
trigger, to tell `"How would you design"` (unfinished) from `"How would you
design a distributed system"` (finished). That distinction is not a count:

```
"How would you design"   one substantive word after the trigger, unfinished
"What is Kafka"          one substantive word after the trigger, finished
```

so every short question of the second shape was marked `sparse` and paid
`question_hold_incomplete_ms` — 3000ms of dead air on the most ordinary
phrasing an interviewer uses, and only when Whisper omitted the `?`, which is
the case this pipeline is built around.

**(b) `_OPEN_PREMODIFIER` matched any determiner + `-ed`/`-ing` word.**
The `-ing` half is not evidence: after a determiner an `-ing` word is usually
a gerund heading a *finished* phrase. Confirmed false positives: `a string`,
`the encoding`, `the padding`, `the meeting`, `a rating`, `a thing`.
`"Reverse a string"` is the single most common coding prompt there is.

**(c) The two modules disagreed about `it`/`this`.**
`is_referentially_empty` treats them as legitimate governed objects (so
`"Explain this."` is accepted as a real request); `_TRIVIAL_TAIL_WORDS`
independently called the same words trivial (so the same utterance was then
marked `sparse` and held 3000ms). Accepted as a question and simultaneously
treated as though nothing had been asked.

### Implementation

- **`_has_content_after` reverted to reason-agnostic** (one substantive word,
  either trigger kind). It is back to the one thing a word count *can* decide:
  did the interviewer say anything at all after naming the kind of request.
- **New `prompt_detector.has_open_clause`** answers the structural question
  instead: *does this clause's predicate have an argument?* It reads only
  **closed** word classes — determiners, auxiliaries, copulas, pronouns,
  prepositions, connectives, interrogatives, complementisers. Those are
  genuinely finite sets in English; every open-class word the interviewer
  might use is simply `CONTENT`, unlisted and unnamed. This is explicitly
  **not** a keyword list, and there is **no LLM call** (asserted by
  `test_an_llm_is_never_consulted_to_decide_finality`).
- **`_OPEN_PREMODIFIER` restricted to past participles**, with a
  morphological guard `_is_participle`: strip `-ed` and require ≥4 characters
  of stem, so `distributed`→`distribut` is a participle and `thread`→`thr`,
  `speed`→`spe`, `head`→`h` are not. `-ing` is gone entirely.
- **`_OBJECT_STANDINS` moved above `_TRIVIAL_TAIL_WORDS` and subtracted from
  it**, so the two readings cannot drift apart again. A trailing `that` is
  still held, by `_DANGLING_WORDS`, which is the right layer for a word that
  is genuinely ambiguous between an object and a complementiser.

The three signals meet in `question_detector._assess`, in the same
provably-incomplete tier as the existing scenario/placeholder/premodifier
checks.

### Verification

All 14 mandated SHOULD-COMPLETE-IMMEDIATELY and all 9 mandated
MUST-REMAIN-INCOMPLETE controls pass. Measured latency, real timing
(`hc_session` §3d, 7/7 — previously 0/7):

| utterance | before | after |
|---|---|---|
| `What is Kafka` | 3140ms | answered inside 200ms |
| `When does it rebalance` | 3155ms | answered inside 200ms |
| `Are there tradeoffs` | 3155ms | answered inside 200ms |
| `Implement this` | 3156ms | answered inside 200ms |
| `Optimize this` | 3156ms | answered inside 200ms |
| `Explain this.` | 3265ms | answered inside 200ms |
| `Optimize it.` | 3141ms | answered inside 200ms |

`hc_detector` §7 referential fragments: 45/45 (was 39/45).
`hc_detector` §5 imperatives: 17/17 (was 12/17).

**The 3000ms hold was not reduced.** `question_hold_incomplete_ms` is still
3000 and `question_stabilization_ms` still 2500; what changed is which
utterances are held at all.

---

## 3. HC-001 — P1 — nested-clause premature finality

### Root cause

The detector's incompleteness evidence was the number of substantive words
and the identity of the final word. A subordinate clause caught mid-flight
defeats both: it is several words long and ends on ordinary content.

```
"Explain how Kafka handles"   3 substantive words, ends on "handles"
```

Nothing said the embedded clause `how Kafka handles …` has a predicate with
no argument, so it was read as complete, answered, and answered again when
the rest of the sentence arrived.

### Implementation

`has_open_clause` (the same function as HC-002) finds the **innermost** clause
— the one still being spoken — and asks whether its predicate is saturated:

```
"how the database"              subject, no predicate      -> open
"how Kafka handles"             predicate, no argument     -> open
"how Kafka handles partitions"  predicate + argument       -> closed
```

Three deterministic distinctions carry it, all structural:

1. **Inversion.** A wh-word followed by an auxiliary is a main-clause
   question (`why does Kafka rebalance`); followed by anything else it is an
   embedded clause (`how Kafka handles`). Standard syntax, not a heuristic.
2. **Subject person.** A bare predicate with a first/second-person subject is
   the interviewer asking the *candidate* to act, and the thing to act on is
   what has not been said (`would you design`). A third-person subject asks
   about the behaviour of a thing, which is whole with no object at all
   (`does it rebalance`).
3. **Terminal punctuation, and only for the one genuinely undecidable case.**
   A predicate with no argument is finished for an intransitive verb (`how
   Kafka works`) and unfinished for a transitive one (`how Kafka handles`).
   No lexicon-free rule separates those, so the tie goes to the speaker: a
   `.`/`!` is Whisper reporting the utterance ended — the same weak-but-real
   closure signal `is_referentially_empty`'s caller already trusts. **The
   unambiguous sub-case (a clause with no predicate at all) does not consult
   punctuation**, so `"Tell me how the database."` is held whether or not
   Whisper terminated it.

A relative `that` is not treated as a clause opener unless a determiner or
pronoun follows it, which is what keeps `"find two numbers that sum to a
target"` (the relative reading, already-protected COMPLETE) apart from
`"ensure that the service …"` (the complementiser reading).

### Verification

Detector level, `hc_detector` §3: **7/7** (was 2/7). End to end with
production hold timing, `hc_session` §3c: **6/6 one question, one
understanding call, one answer call** (was 2/6).

| script | before | after |
|---|---|---|
| `Explain how Kafka handles` + `partitions?` | 2 questions, 2 answers | 1 / 1 |
| `Explain how Kafka handles` + `partitions when brokers fail?` | 2, 2 | 1 / 1 |
| `Tell me how the database` + `handles replication?` | 2, 2 | 1 / 1 |
| `Describe how the system` + `should recover from failure?` | 2, 2 | 1 / 1 |
| `How would you ensure` + `that the service` + `does not lose data?` | 1 / 1 | 1 / 1 |
| `Can you explain why` + `the consumer` + `falls behind?` | 1 / 1 | 1 / 1 |

Every mandated preserved-complete case still fires immediately: `"Explain how
Kafka works."`, `"Tell me how the database works."`, `"Describe how the system
recovers."`, `"Why does Kafka rebalance?"`, `"When does the database
replicate?"`.

**Not all nested questions wait**: `"Can you explain how dependency injection
works in FastAPI"`, `"Explain how you would design this system."` and
`"Tell me how you would debug this."` are all COMPLETE with no hold.

---

## 4. HC-003 — P1 — duplicate identical final

### Root cause

`_is_repeat_of_tail` prevented a redelivered final being *concatenated* onto
itself, and nothing prevented it being *accepted*. So a copy opened a second
turn, spent a second understanding call and a second answer call, and left
the first answer running: 2 questions, 2 answers, 0 cancellations.

### Implementation

`QuestionDetector._is_redelivery`, checked before every other gate. Three
conditions, deliberately not text alone:

1. the text matches the **currently-live accepted turn** exactly, on the words
   (case, internal whitespace and terminal punctuation normalised by
   `_identity`, because two passes over the same audio differ freely on all
   three);
2. a question has actually been accepted for it to be a copy of;
3. it lands within **`question_redelivery_ms` (new, 1500)** of that
   acceptance.

Measured from the one real question rather than from the previous copy, so a
burst cannot hold its own window open. Rejected with
`RejectionReason.NOT_A_QUESTION` and `detail="duplicate_redelivery"` — the
wire-level reason enum is unchanged, so no frontend contract moved.

### Verification

| scenario | result |
|---|---|
| repeated final at +200ms | dropped; **1 question, 1 understanding call, 1 answer call** |
| burst of 7 copies at 200ms intervals | all dropped; the copy at +1.6s is answered |
| punctuation/case drift (`"What is a hash map?"` → `"what is a hash map."`) | dropped |
| duplicate manual submit (`buffer_context=False`) | dropped |
| **the same question asked again 30s later** | **answered** |
| **the same text after a different question** | **answered** |
| **the same text after a topic switch** | **answered** |
| a *different* question at +200ms | answered |
| partial → final → late partial | 1 answer |
| out-of-order finals | both answered, no backward merge |

No global deduplication by question text: every "must still be answered" case
above is a test in `tests/test_turn_identity.py`.

---

## 5. HC-004 — P2 — "Show code" missed

**Root cause.** `_PROMPT_ANYWHERE` only recognises `show me|us`, and `show`
was absent from `_CLAUSE_INITIAL`'s imperative verb group — so an interviewer
who omitted the addressee was not detected at all.

**Implementation.** `show` added to the clause-initial imperative group, where
every other request verb lives. Clause-initial anchoring is what keeps it a
request: matched anywhere, it would fire inside ordinary speech ("the metrics
show latency rising").

**Verification.** `Show code`, `Show Python code`, `Show the implementation`
and `Show me the code.` are all detected and answered; bare `Show` is still
below `question_min_words`; `"The metrics show latency rising after each
deploy."` does not fire as an imperative task.

**One case behaves differently from the mandate's wording:** `Show an example`
is now *detected* (it was rejected outright before) but pays a brief hold,
because `an example` is a placeholder object awaiting its relative clause.
That is pre-existing intentional behaviour protected by
`test_an_incomplete_fragment_accumulates` for `"Walk me through an example"`.
Changing it would weaken an existing test, so it was left alone and is
recorded here.

---

## 6. HC-005 — P2 — split one-word imperative was lost

**Root cause.** The length/referential rejection is an early `return` placed
above the `may_remember` logic, so a short fragment could never become setup
context. `"Explain"` was dropped, and `"Kafka replication."` names no request
of its own, so it was dropped too — the question was lost outright.

**Implementation.** A one-slot `_pending_trigger`:

- **Set** on the length rejection, and only for a fragment that *is* the start
  of a request — pattern extraction matches it and the match is `sparse`,
  meaning the trigger phrase is the whole of what was said. That accepts
  `Explain`, `Optimize`, `Show`, `Describe`; it rejects acknowledgements and
  navigation (extraction returns `None` for both) and referential fragments
  like `"Which one?"` (which match on punctuation, not on a bare trigger).
- **Used** only as a fallback, after `extract_interview_prompt` has already
  failed on the incoming utterance — so an utterance that is a request on its
  own can never be rewritten by an older fragment.
- **Bounded** by `question_context_window_ms`, and cleared the moment any
  question is accepted, so an opener cannot reach across a completed turn.
- Never joined onto an acknowledgement or navigation phrase.

**Verification.** `"Explain"` + `"Kafka replication."` → **one question
(`"Explain Kafka replication?"` — the interviewer's own words, in order), one
understanding call, one answer call**, at gaps of 0.1/0.5/1.5/4.0/7.9s and
with or without the trailing period. Same for `"Optimize"` + `"this query."`
and `"Describe"` + `"the retry policy."`. `"Explain"` alone answers nothing.
`"Okay."`, `"Right."`, `"Mm-hmm."`, `"Next question."`, `"Sure."` never
manufacture a question. A typed question (`buffer_context=False`) neither
holds nor uses an opener.

---

## 7. HC-006 — P2 — NOT_CONFIGURED became UNKNOWN

**Root cause.** `OpenAICompatibleClient.generate_answer` called
`_ensure_client()` from inside `_post()`, inside its own
`except Exception` — so the already-classified `NOT_CONFIGURED` error that
function raises was caught and re-classified as `UNKNOWN` by `_fail`. "No API
key is configured. Add one in Settings" became "The request failed."

**Implementation.** Two layers:

1. `generate_answer` now calls `_ensure_client()` **before** the try, exactly
   as `stream_answer`, `complete_json` and `GroqClient.generate_answer` do.
   `_post` takes the client as a parameter.
2. The general form: `_as_llm_error` in **both** clients returns an `LLMError`
   unchanged instead of running it back through `classify`, which knows
   nothing about this layer's own errors and answers `UNKNOWN`.

The missing-base-url error was also reworded through `app.llm.messages` so it
sends the reader to the right field ("Set its base URL in Settings", not "Add
an API key").

**Verification.** `generate_answer`, `stream_answer` and `complete_json` all
report `NOT_CONFIGURED` with no key; `generate_answer` reports
`NOT_CONFIGURED` with no base URL. A 429 is still classified `RATE_LIMIT`
with `retry_after_seconds == 5.0`, so moving the call did not stop the handler
working. Cancellation still raises `CancelledError` and is never reclassified
(asserted for both clients).

---

## 8. HC-007 — P3 — ungrammatical OpenAI-compatible sentences

**Root cause.** Every template was written around a proper noun ("The {name}
request failed.") and then handed a name carrying its own article, producing
"The the configured endpoint request failed." and "No the configured endpoint
API key is configured."

**Implementation.** Fixed the composition, not the string. `app.llm.messages`
now offers the three grammatical forms English needs, and each template asks
for the one that fits its slot:

```
{bare}  Groq | OpenAI-compatible endpoint       after an article
{name}  Groq | the OpenAI-compatible endpoint   mid-sentence
{Name}  Groq | The OpenAI-compatible endpoint   sentence-initial
```

Which shape a name has is read off the name itself (a leading `"the "`), so
registering a provider stays a one-line change. For a proper noun all three
forms are the same string — which is why **every sentence for Groq, OpenAI
and OmniRouter is byte-for-byte what it was**, pinned literally by
`test_a_proper_noun_provider_keeps_its_exact_wording`. The display name for
`openai_compatible` became `"the OpenAI-compatible endpoint"` (from `"the
configured endpoint"`) so that the word "configured" does not collide with
the MODEL_UNAVAILABLE template.

**Verification.** All 9 kinds × 4 providers = 36 sentences, checked
structurally (no doubled article, no doubled word, sentence-initial capital,
terminal period) rather than against a stored copy, so it keeps holding for a
provider added later. Every one is also on the safe whitelist (§9).

---

## 9. HC-009 — P3 — ANSWER_ERROR forwarded arbitrary exception text

**Root cause.** The WebSocket error event emitted `str(exc)` verbatim, for
both `LLMError` and the catch-all `except Exception`. No real client builds a
message that way, but nothing enforced it.

**Implementation.** A whitelist, not a filter. Every user-facing provider
sentence comes from a fixed template in `app.llm.messages`, so "was this
built there?" is answerable exactly — `messages.is_known_message`. In
`session._client_safe_message`:

- an app-built sentence is shown as written (including MODEL_UNAVAILABLE,
  matched on its fixed head and tail because the model name is configuration);
- anything else is **replaced** by `message_for(exc.kind, "the AI provider")`
  — so the reader still learns what went wrong and what to do, from the
  classification the provider already made — and the substitution is logged;
- `AnswerValidationError` gets its own app-authored sentence;
- the catch-all branch emits a fixed sentence and **never** `str(exc)`; the
  detail is already in `logger.exception`.

Deny by default: a future provider raising a raw `LLMError` gets a correct
generic sentence rather than deciding what reaches the wire.

**Verification.** A provider that raises
`LLMError("401 from https://api.example/v1 key=SUPER_SECRET_TEST_KEY_123456789")`
produces a wire payload containing neither the secret nor the URL, and the
client is shown "The AI provider API key was rejected. Check it in Settings
and save it again." An internal `RuntimeError` carrying the secret reaches the
client as `code="InternalError"` with nothing of itself in the message. The
turn still reaches a terminal state and the next question still works.

`tests/test_live_session.py::test_llm_failure_emits_answer_error` previously
asserted `"exploded" in message` — i.e. that the provider's own words reached
the client, which is exactly the behaviour this finding required be removed.
It now asserts the opposite plus the correct template sentence, with the
reason written at the assertion.

---

## 10. HC-010 — P3 — settings serialisation carried API keys

**Root cause.** `__repr_args__` redacted `repr`/`str`; `model_dump()` and
`model_dump_json()` were untouched, so any future diagnostics endpoint,
support bundle, telemetry payload or config echo would have carried every key
in plain text.

**Implementation.** A `@model_serializer(mode="wrap")` on `Settings` that
replaces every non-empty `*_api_key` with `"***"`. Redaction lives in
serialisation only: **attribute access is untouched**, so
`api_key_for`, `app.core.secrets` and the credential-store round trip all
still see the real value. Reading a key and publishing one are different acts,
and only the second is guarded.

The marker is `"***"` rather than an empty string, so a dump shows *that* a
key is set without revealing it — an empty string would be indistinguishable
from "no key configured", which is a different and misleading claim.

**Verification.** `repr`, `str`, `model_dump()`, `model_dump(mode="json")` and
`model_dump_json()` are all clean with every `*_api_key` field planted;
`settings.groq_api_key` and `api_key_for("groq")` still return the real key;
`secret_field_names()` still discovers all four providers; non-secret fields
survive serialisation unchanged; a field with no key still dumps `""`.
`hc_provider` §16 now reports `contains_secret: false` for both dump forms
(previously `true`), and its 50 other secret-surface checks stay clean.

---

## 11. HC-008 — 3-second boundary

Not prioritised, and the timing system was **not** changed. A characterization
test was added instead —
`test_turn_finalization.py::test_the_hold_boundary_never_costs_a_duplicate_answer`,
6 repetitions — which sends a continuation at exactly the hold and asserts
only what is not negotiable at a boundary no interviewer can hit on purpose:

- one or two questions detected (both orderings legal),
- **no turn answered twice**,
- answers ≤ questions,
- and nothing the interviewer said is lost from what does get answered.

`hc_session` §2-pause reported 9/9 this run and 8/9 in the discovery pass, on
the same code — which is the nondeterminism, observed.

---

## 12. ENV-001 — keyring

`keyring>=25.0` was declared in `requirements.txt` and absent from
`backend/.venv`. Installed:

```
keyring-25.7.0  pywin32-ctypes-0.2.3  jaraco.classes-3.4.0
jaraco.context-6.1.2  jaraco.functools-4.6.0  more-itertools-11.1.0
```

Nothing was removed from `requirements.txt`. `tests/test_secret_storage.py`
went from 31 passed / 2 failed to **33 passed**.

### Full dependency audit

Also declared in `requirements.txt` and **still absent** — deliberately not
installed, because they are the audio/STT/OCR/packaging stack and real
audio/Tauri work is out of scope for this pass:

```
faster-whisper   sounddevice   PyAudioWPatch
pypdfium2        rapidocr-onnxruntime        pyinstaller
```

**A correction to `HARD_CASE_E2E_REPORT.md`:** that report said "the OCR
`slow` tests ran and passed". They did not. `tests/test_pdf_ocr.py` opens
with a module-level `pytest.importorskip("PIL")`, which short-circuits before
its `pypdfium2` and `rapidocr_onnxruntime` guards — so the **entire OCR suite
is skipped**, and has been for both passes. Installing `Pillow` (test-only,
not in `requirements.txt`) plus `pypdfium2` and `rapidocr-onnxruntime` would
unlock it; recommended, but deliberately not done mid-fix-pass because it
opens a previously-unexercised test surface.

---

## 13. New findings discovered during this pass

### NEW-001 (P1, fixed) — a continuation was read as a follow-up

Surfaced by a regression test I wrote for HC-001, not by the inventory.

While a turn was being **held**, a short utterance was treated as a
*follow-up* to the held fragment rather than as its *continuation* — the
follow-up bypass ran before the merge and excluded merging. So
`"Explain how Kafka handles"` + `"partitions?"` produced one answer to a bare
`"partitions?"`, with **every word of the fragment discarded** and no subject
in the question sent to the model. One answer, so it never showed up as a
duplicate — it showed up as a wrong answer.

**Root cause.** A follow-up leans on an answer the candidate has just read. A
held turn has produced no answer to lean on, so the follow-up reading cannot
be right there.

**Fix.** `inspect` now computes `mergeable` and `continuation` before the
length gate. While accumulating, a short mergeable utterance skips the word
floor and merges instead of being read as a follow-up. Granted only when the
merge would actually happen, so the floor still holds for a short utterance
with nothing to attach to. Covered by
`test_a_split_nested_clause_costs_one_question_and_one_answer`.

### NEW-002 (P1, fixed) — the standin-pronoun fix broke referential rejection

Caught by the **existing red-team probe** (`probe_detector`
`9-short_followup_no_context`), which is why it is worth recording: removing
`it`/`this`/`that` from `_TRIVIAL_TAIL_WORDS` also removed them from
`_REFERENTIAL_FILLER`, which is derived from it — so `"What about it?"` stopped
being referentially empty and was accepted with zero context.

**Fix.** `_REFERENTIAL_FILLER` adds `_OBJECT_STANDINS` back explicitly, with
the reason at the definition: the two sets answer different questions and a
standin answers them oppositely. Both readings are now pinned by tests in the
same file so they cannot drift again.

### NEW-003 (P3, not fixed — pre-existing, documented) — trailing "that" is held

`"Answer that."` is held ~3s. `that` stays in `_DANGLING_WORDS` because it is
genuinely ambiguous between an object ("answer that") and a complementiser
("ensure that the service …"), and the second reading is the one where firing
early costs a wrong answer. Pre-existing, unchanged by this pass, and now
pinned as a decision by
`test_a_trailing_that_is_still_held_and_that_is_deliberate`.

### Three QA-tooling defects (fixed; none are production code)

- **`probe_session` `split_becomes_two_answered_turns_at_2500ms_gap`** built
  the harness with a 1ms hold and then scripted a 2.5s gap, so the hold
  expired between the two utterances. It was measuring the harness, not the
  product — and its stored result from the discovery pass was already
  `"ok": false`. Now uses `question_hold_incomplete_ms`, and passes.
- **`probe_detector` `16-duplicate_final`** asserted `det_a2.accepted`, the
  behaviour HC-003 identifies as the bug. Its own note already allowed
  "or deduped upstream"; updated to the new contract, with the reasoning at
  the line.
- **`probe_secret_leak` had been unrunnable** since the provider refactor
  (`GroqClient()` now takes a `ProviderConfig`; `groq_client._message` and
  `settings.groq_model` no longer exist). Repaired — so its 4 secret-leak
  checks now actually run, for the first time since that refactor. This means
  the discovery report's secret-security section did not include this probe;
  it does now, and all 4 pass.

---

## 14. Full regression results

### Backend suite

```
backend/.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider

run 1:  1714 passed, 3 skipped, 1 warning in 174.94s
run 2:  1714 passed, 3 skipped, 1 warning in 164.08s
collected: 1716
```

**0 failures.** Skipped (3, unchanged): `test_pdf_ocr.py:27` (Pillow),
`test_audio_devices.py:125` (sounddevice), `test_embeddings.py:54`
(`RUN_MODEL_TESTS=1`).

### Red-team suite (existing, `backend/qa_redteam/`)

| probe | checks | fail |
|---|---|---|
| `probe_detector.py` | 87 | 0 |
| `probe_session.py` | 24 | 0 |
| `probe_attachments.py` | 6 | 0 |
| `probe_second_pass.py` | 6 | 0 |
| `probe_secret_leak.py` | 4 | 0 |
| **total** | **127** | **0** |

### Hard-case suite (`backend/qa_hardcase/`)

| probe | judged | pass | fail | observations |
|---|---|---|---|---|
| `hc_detector.py` | 148 | 148 | 0 | 27 |
| `hc_session.py` | 88 | 88 | 0 | 18 |
| `hc_provider.py` | 166 | 166 | 0 | 6 |
| **total** | **402** | **402** | **0** | **51** |

Verified deterministic at the post-pass checkpoint: `hc_session` was run four
times and reported 88/88 every time; `hc_detector` and `hc_provider` twice
each, identical.

Per-section, against the discovery pass:

| section | before | after |
|---|---|---|
| §3 nested clause (detector) | 2/7 | **7/7** |
| §3c nested clause (real holds, e2e) | 2/6 | **6/6** |
| §3d over-hold latency | 0/7 | **7/7** |
| §4 interrogative ladder | 45/48 | **48/48** |
| §5 imperative | 12/17 | **17/17** |
| §5 / §5b imperative e2e | 7/11 | **11/11** |
| §7 referential fragments | 39/45 | **45/45** |
| §11 duplicate / race | 7/8 | **8/8** |
| §15 provider failure taxonomy | 13/14 | **14/14** |
| §16 secret surfaces | 50/50 | **50/50** |
| everything else | pass | pass |

The judged total moved from 414 to 402 because twelve rows across two
sections became observations. Both sections squash the hold to 1ms -- a
configuration production never produces, since the real hold is 3000ms
against a 400ms gap -- and under it the outcome races with the continuation.
Measured at the checkpoint, `3-e2e-fast-hold` flipped on 0, 1, 3 and 4 of its
six rows across four runs, and *both* outcomes there are correct by design:
the continuation either merges into one question, or the fragment was already
answered and the short continuation is read as a genuine follow-up to it,
which is what the follow-up bypass is for once an answer exists. A check that
cannot tell a defect from a legal outcome is not a check, so both sections
record observations. §3c runs the identical six scripts on production hold
timing and does state the verdict (6/6 in every run), as does
`tests/test_clause_saturation.py` in the production suite.

`2-pause` at exactly the hold (HC-008) was the other flipping row. It stays a
verdict, but now asserts what is actually guaranteed at a boundary no
interviewer can hit on purpose -- one or two questions, never a turn answered
twice, never a word lost -- instead of guessing which way the race falls.

### Property fuzz

```
utterances fuzzed:        167,994
detector evaluations:     335,988   (each unpunctuated and with a trailing "?")
I1 violations:            0   (a fragment ending on a dangling function word became COMPLETE)
I2 violations:            0   (a "?"-terminated question was left ACCUMULATING)
exceptions:               0
```

Identical to the discovery pass. **Not regressed.**

### Frontend

**No frontend files were changed in this pass** — verified from `git status`.
The wire contract was checked deliberately: no new `RejectionReason` value
(the redelivery drop reuses `not_a_question` with an internal `detail`), no
new or removed event field. Run anyway as insurance:

```
npm run typecheck   -> clean
npm test            -> 125 passed, 0 failed
```

---

## 15. Tests added and changed

### New files (339 tests)

| file | tests | covers |
|---|---|---|
| `backend/tests/test_clause_saturation.py` | 179 | HC-001, HC-002, HC-004, NEW-002, NEW-003 |
| `backend/tests/test_provider_messages.py` | 97 | HC-006, HC-007 |
| `backend/tests/test_turn_identity.py` | 40 | HC-003, HC-005 |
| `backend/tests/test_error_boundary.py` | 23 | HC-009, HC-010 |

`test_clause_saturation.py` is built around two parametrised lists,
`COMPLETE_NOW` (32 utterances) and `STILL_OPEN` (21), which are the contract:
nothing moves between them without a deliberate decision to change what the
app does. It also asserts the transcript is never rewritten for either list,
and that no LLM is consulted to decide finality.

### Changed files

- **`test_question_patterns.py`** —
  `test_a_duplicate_final_is_not_concatenated_onto_itself` asserted
  `repeat.accepted`, which HC-003 identifies as the bug. Split into
  `test_a_duplicate_final_is_dropped_as_a_redelivery` (the new, stronger
  contract — text that never reaches extraction cannot double up) and
  `test_a_late_redelivery_is_still_not_concatenated_onto_itself`, which keeps
  the original guarantee covered on a path that is still reachable: an
  imperative-task accept whose merge window outlives the redelivery window.
  **Coverage was preserved, not deleted.**
- **`test_live_session.py`** — one assertion, see §9.
- **`test_turn_finalization.py`** — added the HC-008 characterization test and
  an `asyncio` import.

No test was deleted, skipped, xfailed, or weakened. Every expectation that
changed did so because the fix mandate names the new behaviour, and the reason
is written at the assertion.

---

## 16. Exact files changed

### Production (8)

```
backend/app/realtime/prompt_detector.py     HC-001, HC-002, HC-004, NEW-002
backend/app/realtime/question_detector.py   HC-001, HC-002, HC-003, HC-005, NEW-001
backend/app/realtime/session.py             HC-009
backend/app/core/config.py                  HC-003 (question_redelivery_ms), HC-010
backend/app/llm/messages.py                 HC-007, HC-009 (whitelist)
backend/app/llm/registry.py                 HC-007 (display name)
backend/app/llm/openai_compatible.py        HC-006, HC-007
backend/app/llm/groq_client.py              HC-006 (LLMError passthrough)
```

### Tests (7)

```
backend/tests/test_clause_saturation.py     new
backend/tests/test_turn_identity.py         new
backend/tests/test_provider_messages.py     new
backend/tests/test_error_boundary.py        new
backend/tests/test_question_patterns.py     one test split in two
backend/tests/test_live_session.py          one assertion
backend/tests/test_turn_finalization.py     one test added, one import
```

### QA tooling (5, plus regenerated result JSON)

```
backend/qa_redteam/probe_detector.py        expectation updated to the HC-003 contract
backend/qa_redteam/probe_session.py         harness hold fixed
backend/qa_redteam/probe_secret_leak.py     repaired (unrunnable since the provider refactor)
backend/qa_hardcase/hc_detector.py          5 expectation labels corrected
backend/qa_hardcase/hc_session.py           §3/§3b reframed, §3e limitation added,
                                            plus the two racy rows made
                                            deterministic at the checkpoint
```

### Environment

```
backend/.venv: + keyring 25.7.0 (and 5 transitive)
```

`git diff --stat` against `HEAD` reports 23 files and ~1,650 insertions, but
that **includes uncommitted work already in the tree before this pass** —
fix pass #1's detector/session changes and the peer session's provider layer
(`app/api/settings_api.py`, `app/core/deps.py`, `app/api/health.py`,
`app/realtime/question_understanding.py`, `README.md`, the frontend files, and
several test files). The lists above are the files **this pass** touched.

---

## 17. Remaining known limitations

1. **A period on an open clause still fires early.** The one genuinely
   undecidable case in HC-001 — a predicate with no argument — is settled by
   terminal punctuation, so if Whisper emits `"Explain how Kafka handles."`
   the fragment is answered and its continuation becomes a second question.
   Measured, recorded as `hc_session` §3e: 2 questions / 2 answers with the
   period, 1/1 without. Closing it needs verb argument structure, which a
   deterministic detector does not have. The unambiguous half (a clause with
   no predicate at all) is robust to punctuation — the same section shows
   `"Tell me how the database."` still reassembling into 1/1.
2. **`"Explain how Kafka works"` unpunctuated waits.** The mirror of the
   above: with no punctuation the intransitive reading cannot be
   distinguished, so it holds. This is a latency cost on a correct question,
   which is the direction the design deliberately errs.
3. **`"Show an example"` pays a brief hold** — pre-existing placeholder-object
   behaviour, protected by an existing test (§5).
4. **`"Answer that."` is held** — NEW-003, pre-existing and deliberate.
5. **HC-008 remains nondeterministic at exactly the hold boundary**, by
   choice. The invariants are pinned; the ordering is not.
6. **The OCR suite is entirely skipped** and has been for both passes (§12).
7. **Real-world exposure of HC-002 and HC-001 is still unmeasured**, because
   both depend on how often real Faster-Whisper omits or supplies terminal
   punctuation. `question_redelivery_ms=1500` is likewise sized from
   reasoning about speech, not from observed STT redelivery intervals.

### Not tested in this pass, as instructed

- WASAPI / device capture
- real Faster-Whisper inference
- real Tauri UI
- packaged desktop app
- real hardware latency
- any real provider network call (every provider result is
  `httpx.MockTransport` or a duck-typed fake SDK)

No provider behaviour was invented: the OpenAI-compatible status-code mapping
and Groq's SDK exception-name mapping were left exactly as they were, and only
the *wording* and *classification plumbing* around them changed.

---

## 18. Against the stated target

| target | status |
|---|---|
| EXACT COMPLETE QUESTION | Transcript never rewritten — asserted for all 53 utterances in the two finality contracts, and the HC-005 join preserves the interviewer's own words in order |
| → EARLIEST SAFE FINALITY | 10 of 25 realistic short questions no longer pay a 3s hold; nothing in the mandated MUST-REMAIN-INCOMPLETE list fires early; 335,988 fuzz evaluations still show 0 invariant violations |
| → ONE UNDERSTANDING CALL | Asserted for the nested-clause chains, the split opener, the redelivery, and all three rapid-fire runs (20/30/50) |
| → CORRECT CONTEXT | §8 contamination and §9 pronoun chains still 4/4 and 3/3; NEW-001 fixed a case where the context was silently *dropped* |
| → ONE ANSWER CALL | Same as above; HC-003 removed the duplicate |
| → NO DUPLICATE ANSWERS | HC-001, HC-003, NEW-001 all closed; HC-008's boundary pinned to "never twice" |
| → NO SECRET LEAKS | 50 secret-surface checks clean, plus `model_dump`/`model_dump_json` now redacted and the WS error boundary whitelisted; 23 provider-error messages clean |

Nothing was committed. Nothing was pushed.
