# HARD_CASE_E2E_REPORT.md — Interview Coach adversarial / hard-case pass

**Date:** 2026-09-07
**Branch:** `feat/desktop-app-and-llm-routing`
**HEAD:** `c72055d Harden realtime context and retrieval`
**Mode:** discovery only. **No production code was modified. Nothing was committed or pushed.**

---

## 0. What this is, and what it explicitly is NOT

Every result below comes from driving the **real** production modules
(`prompt_detector`, `question_detector`, `LiveSession`, `QuestionUnderstander`,
`app.llm.registry`, `GroqClient`, `OpenAICompatibleClient`) with fake
transports, fake LLMs and synthetic transcripts, through `tests/replay_harness.py`
— the same harness the production suite uses.

### NOT TESTED (deferred to the personal laptop)

- **WASAPI** loopback / device capture
- **real Faster-Whisper** inference (accuracy, punctuation behaviour, latency)
- **real Tauri UI**
- **packaged desktop app**
- **real hardware latency** (GPU/CPU inference time, audio buffer jitter)
- any **real provider network call** (Groq, OpenAI, OmniRouter) — all provider
  work below is `httpx.MockTransport` or a duck-typed fake SDK

Consequently every statement about *how often* Whisper omits a `?` is taken
from the codebase's own documented premise, not measured here. Several findings
below (notably HC-002) only bite when Whisper omits the `?`, so their real-world
severity **cannot be finalised until real STT is exercised**.

### Artifacts produced (test-only, isolated, never imported by `app/` or `tests/`)

```
backend/qa_hardcase/hc_common.py            shared scaffolding + settings hygiene
backend/qa_hardcase/hc_detector.py          sections 2-7, 17
backend/qa_hardcase/hc_session.py           sections 2, 3, 5, 8-13, 18
backend/qa_hardcase/hc_provider.py          sections 14-16
backend/qa_hardcase/hc_*_results.json       full machine-readable results
```

Each script mirrors `tests/conftest.py`'s hygiene: `Settings(_env_file=None)` is
copied over the singleton and an `InMemorySecretStore` is installed, so a
developer's `backend/.env` or OS credential store cannot change a verdict.

---

## 1. Baseline

```
Command: backend/.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --tb=no -rf
OS:      Windows 11 Pro 10.0.26200
Python:  3.12.0 (backend/.venv)
```

Two full runs, back to back:

| Run | passed | failed | skipped | warnings | wall |
|-----|--------|--------|---------|----------|------|
| 1   | 1365   | 3      | 3       | 1        | 184.0s |
| 2   | 1366   | 2      | 3       | 1        | 193.5s |

**Collected: 1371.** The run-to-run delta is one flaky test, not a code change
(nothing was edited between the two runs).

### Failing tests, exact names

```
FAILED tests/test_secret_storage.py::test_the_backend_is_resolved_directly_when_discovery_finds_nothing
FAILED tests/test_secret_storage.py::test_a_machine_with_no_backend_at_all_reports_unavailable
FAILED tests/test_stt_pipeline.py::test_final_diverging_from_its_partial_is_logged   (run 1 only)
```

- The two `test_secret_storage` failures are **environment-only** (`ENV-001`):
  both `import keyring` directly and it raises `ModuleNotFoundError`.
  Confirmed: `./.venv/Scripts/python.exe -c "import keyring"` → `ModuleNotFoundError`.
  `keyring>=25.0` is in `requirements.txt` but not installed in `backend/.venv`.
  Production code imports it lazily inside `KeyringSecretStore`, which is why
  every other test in that file passes. Unchanged from `QA_BASELINE.md`.
- `test_final_diverging_from_its_partial_is_logged` is **flaky** (`FLAKE-001`),
  not a product failure: it passes in isolation (`1 passed in 0.35s`) and passed
  in full-suite run 2. It asserts a partial transcript was emitted before the
  final at `partial_interval_ms=200`; under full-suite CPU load the partial can
  miss its window and no divergence metric is recorded.

### Skipped tests (3, exact names)

```
SKIPPED tests/test_pdf_ocr.py:27       Pillow is needed to build the test PDFs
SKIPPED tests/test_audio_devices.py:125 requires sounddevice
SKIPPED tests/test_embeddings.py:54    set RUN_MODEL_TESTS=1 to exercise the real ONNX model
```

All three are environment-gated, not the `slow` marker (the OCR `slow` tests ran
and passed).

### Warning (1)

```
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```

### Adversarial pass totals

| Probe | judged checks | pass | fail | observations |
|-------|---------------|------|------|--------------|
| `hc_detector` | 148 | 129 | 19 | 27 |
| `hc_session`  | 100 | 72  | 28 | 4 |
| `hc_provider` | 166 | 165 | 1  | 6 |
| **total**     | **414** | **366** | **48** | **37** |

`hc_session` was run twice end to end: **27 of its 28 failures reproduced
identically**, and the one that differed is HC-008 (the 3.0s boundary race).
`hc_detector` and `hc_provider` are fully deterministic by construction (no
real sleeps).

Plus **335,988 property-fuzz detector evaluations** (167,994 utterances × 2
punctuation variants) — 0 invariant violations, 0 exceptions (§17).

The 48 raw failures collapse to **8 distinct product findings** plus 2
environment items; the rest are the same root cause observed at different layers
or under different timing (each listed row is kept, so nothing is hidden).

---

## 2. Question finality — fragment growth and the pause matrix

### 2a. Growing fragments, detector level (`2-growth`, 5/5 pass)

| utterance | finality | hold |
|---|---|---|
| `How would you` | ACCUMULATING (sparse) | 3000ms |
| `How would you design` | ACCUMULATING (sparse) | 3000ms |
| `How would you design a` | ACCUMULATING (dangling `a`) | 3000ms |
| `How would you design a distributed` | ACCUMULATING (open premodifier) | 3000ms |
| `How would you design a distributed system` | COMPLETE | 0 |

No premature completion. The BUG-001 fix holds.

### 2b. Continuations merge correctly (`2-merge`, 3/3 pass)

`[held fragment] + "system?"`, `+ "system that processes millions of events"`,
`+ "...per second?"` → each reassembles into **one** COMPLETE question.

### 2c. Lone fragments through the real `LiveSession`, real hold timing (`2b-premature` 6/6, `2b-eventual` 6/6)

For each of the four incomplete prefixes: **no** `QUESTION_DETECTED` and no
answer within 250ms (the hold is genuinely respected), and **every** fragment
is eventually answered once the hold expires — the documented
abandoned-fragment fallback. Nothing is silently lost.

### 2d. Pause matrix, real timing (`2-pause`, 8/9 pass)

`"How would you design a distributed"` … pause … `"system that processes millions of events per second?"`

| pause | result | verdict |
|-------|--------|---------|
| 100ms | 1 question, 1 answer | correct merge |
| 300ms | 1 question, 1 answer | correct merge |
| 500ms | 1 question, 1 answer | correct merge |
| 1.0s  | 1 question, 1 answer | correct merge |
| 1.5s  | 1 question, 1 answer | correct merge |
| 2.0s  | 1 question, 1 answer | correct merge |
| 2.5s  | 1 question, 1 answer | correct merge |
| 3.0s  | run 1: 1 question, 1 answer; run 2: split | **boundary nondeterminism — HC-008** |
| 5.0s  | 2 questions (fragment fired, then the continuation as its own question) | correct: past the 3000ms hold |

3.0s is exactly `question_hold_incomplete_ms`, so the hold expiry and the
continuation race. Two full runs of the probe disagreed on this one row (merged
in run 1, split in run 2) and agreed on all 99 others — which is itself the
evidence that it is a boundary race, not a behaviour. No duplicate answer
either way.

Answers per case: exactly one, in every non-boundary case. No accumulation
leak, no duplicate question, no duplicate answer.

---

## 3. Nested-clause attack — **HC-001 (P1)**

### Detector level (`3-nested`, 2/7 pass)

| utterance | expected | actual |
|---|---|---|
| `Explain how Kafka handles` | HOLD | **COMPLETE** |
| `Tell me how the database` | HOLD | **COMPLETE** |
| `Describe how the system` | HOLD | **COMPLETE** |
| `How would you ensure that the service` | HOLD | **COMPLETE** |
| `Can you explain why the consumer` | HOLD | **COMPLETE** |
| `How would you ensure` | HOLD | HOLD ✓ |
| `Can you explain why` | HOLD | HOLD ✓ (trailing `why` is a dangling word) |

### End to end, with **real** hold timing and 1.5s pauses (`3c-e2e-real-holds`, 2/6 pass)

| script | questions | answers | understanding calls | answer calls |
|---|---|---|---|---|
| `Explain how Kafka handles` + `partitions?` | **2** | **2** | 2 | 2 |
| `Explain how Kafka handles` + `partitions when brokers fail?` | **2** | **2** | 2 | 2 |
| `Tell me how the database` + `handles replication?` | **2** | **2** | 2 | 2 |
| `Describe how the system` + `should recover from failure?` | **2** | **2** | 2 | 2 |
| `How would you ensure` + `that the service` + `does not lose data?` | 1 | 1 | 1 | 1 ✓ |
| `Can you explain why` + `the consumer` + `falls behind?` | 1 | 1 | 1 | 1 ✓ |

The two that pass do so because their **first** fragment ends on a dangling word
and is therefore held; the hold then absorbs the whole chain via the widened
merge window. The four that fail fire the fragment immediately, stream a wrong
answer, and then answer the real question separately — **two questions in the
panel, two answers, 2× provider cost.**

`FIX_MODE_REPORT.md` §3 documents `"Explain how Kafka handles"` as a known
remaining gap. The other four are **new instances of the same root cause** and
were not previously documented.

**Root cause** (`app/realtime/prompt_detector.py`):
`_has_content_after` counts *words*, and `_looks_incomplete` looks only at the
*last* word. `"Explain how Kafka handles"` has three substantive words after the
trigger and ends on `handles`, which is not in `_DANGLING_WORDS` — so neither
check fires, even though the subordinate clause `how Kafka handles …` has no
object. Nothing in the deterministic layer models argument structure.

---

## 4. Interrogative attack (`4-interrogative`, 45/48 pass)

16 interrogative/auxiliary openers × 3 lengths. All ladders behave correctly
except three, all of them the **over-hold** direction (never premature):

| utterance | expected | actual |
|---|---|---|
| `When does it rebalance` | COMPLETE | **ACCUMULATING, 3000ms** |
| `Are there tradeoffs` | COMPLETE | **ACCUMULATING, 3000ms** |
| `Would you choose Kafka` | (I labelled HOLD) | COMPLETE — **not a bug**, see below |

`Would you choose Kafka` is a grammatically complete yes/no question and firing
on it is correct; my expectation was wrong. The session's own fragment debounce
(`question_continuation_ms`) still absorbs a later `"...for this?"`.

The other two are HC-002 (§ below).

---

## 5. Imperative / task attack

### 5a. Detector level (`5-imperative`, 12/17 pass)

| utterance | expected | actual | finding |
|---|---|---|---|
| `Explain` | — | REJECT (`too_short`) | correct-ish, but see HC-005 |
| `Explain Kafka` | COMPLETE | COMPLETE ✓ | |
| `Explain Kafka replication` | COMPLETE | COMPLETE ✓ | |
| `Show code` | COMPLETE | **REJECT (`no_question_pattern`)** | **HC-004** |
| `Show Python code` | COMPLETE | **REJECT (`no_question_pattern`)** | **HC-004** |
| `Implement this` | COMPLETE | **ACCUMULATING, 3000ms** | HC-002 |
| `Optimize this` | COMPLETE | **ACCUMULATING, 3000ms** | HC-002 |
| `Implement this in Python` | COMPLETE | COMPLETE ✓ | |
| `Optimize this query` | COMPLETE | COMPLETE ✓ | |
| `Compare Kafka and RabbitMQ` | COMPLETE | COMPLETE ✓ | |
| `Design a pipeline` / `...for millions of events` | COMPLETE | COMPLETE ✓ | |
| `Show` / `Implement` / `Optimize` / `Compare` / `Design` (bare) | REJECT | REJECT ✓ | below `question_min_words=2` |

The interrogative fix did **not** break imperative tasks: every multi-word
imperative with a real object still resolves immediately.

### 5b. End to end (`5-e2e`, 7/9 pass; `5b-split-imperative`, 0/2 pass)

- `Show code` / `Show Python code` → **no question detected, no answer at all** (HC-004).
  `Show me the code` works, confirming the gap is the missing `me|us`.
- `"Explain"` then `"Kafka replication."` → **both rejected, the question is lost
  entirely** (HC-005). Same for `"Optimize"` then `"this query."`.

---

## 6. Chatter attack (`6-navigation` 11/11, `6-attached` 6/6 — all pass)

Pure navigation rejected, no LLM call, no answer:

```
Next question.  ·  Okay, next question.  ·  Alright, next question.
Let's move to the next question.  ·  Moving on.  ·  Okay, moving on.
One more question.  ·  Last question.  ·  Final question.
We'll go to the next question.  ·  Let's do another question.
```

Attached questions detected and answered:

```
Next question: what is Kafka?
Next question, what is Kafka?
Okay, next question - explain Kafka.
Okay, next question — explain Kafka.      (em dash)
Here's the next question: how does Kafka work?
One more question: would you use Kafka?
```

The BUG-002 fix is intact under adversarial punctuation, including the em dash.
`"Moving on."` is caught by `_ACK_ONLY` rather than `_NAVIGATION_ONLY`
(`is_navigation_chatter("Moving on.")` is `False`) — two mechanisms, one correct
outcome; recorded in `hc_detector_results.json` under `6-navigation-helper`.

---

## 7. Referential fragment attack (`7-*`, 39/45 pass)

Each fragment run in three context states: zero context, a question accepted 1s
earlier, and a pending attachment.

| utterance | zero context | prior question | pending attachment |
|---|---|---|---|
| `Which one?` `Which?` `Can you?` `Can you explain?` | REJECT ✓ | COMPLETE ✓ | COMPLETE ✓ |
| `Show me.` `Show me?` `Then what?` `Why not?` `How so?` | REJECT ✓ | COMPLETE ✓ | COMPLETE ✓ |
| `What about it?` `What about this?` | REJECT ✓ | COMPLETE ✓ | COMPLETE ✓ |
| `What is Kafka?` `Explain Kafka.` | COMPLETE ✓ | COMPLETE ✓ | COMPLETE ✓ |
| `Explain this.` `Optimize it.` | ACCUMULATING 3000ms | ACCUMULATING 3000ms | **ACCUMULATING 3000ms** |

Both context sources behave identically and correctly: a context-dependent
fragment is rejected with nothing to lean on and accepted with either a recent
question **or** pending pasted material. `is_referentially_empty` correctly
separates the two classes.

`Explain this.` / `Optimize it.` are the exception, and they expose an **internal
inconsistency** rather than a wrong verdict: `is_referentially_empty` treats
`this`/`it` as a legitimate governed object (so the utterance is accepted), while
`_has_content_after` counts the same word as a trivial tail (so the utterance is
marked `sparse` and held 3000ms). `"Explain this."` right after a paste is the
single most common attachment flow, and it costs 3.3s of dead air — see HC-002.

---

## 8. Context contamination (`8-*`, 4/4 pass)

The mandated 12-turn sequence (Python problem → follow-ups → `Now switch to
Kafka.` → Kafka questions → `Now switch to SQL.` → SQL question), with the
understanding layer scripted to the relationships a correct classifier would
report.

- 12 utterances → 12 questions detected, 12 answers completed, 12 understanding
  calls, 12 answer calls. One prompt per answered turn.
- The **SQL** prompt contains no `longest substring` and no `rebalanc*`.
- The **Kafka** prompt contains no `longest substring`.

`select_context`'s relationship-driven narrowing does what it claims. Note the
dependency: on the deterministic fallback path (`UnderstandingSource` ≠ `LLM`)
`select_context` deliberately returns the window whole, so **contamination
isolation is only as good as the understanding call**. That is documented,
intentional behaviour (it preserves pre-understanding semantics), not a finding
— but it means a provider outage widens context rather than narrowing it.

---

## 9. Pronoun / referent chains (`9-*`, 3/3 pass)

`Explain Kafka consumer groups.` → `How does it work?` → `What happens when they
rebalance?` → `How would you handle that?` → `Why is this necessary?` → `Can you
optimize it?` → `Show me.` → `Now explain PostgreSQL replication.` → `How does it
work?`

- All 9 utterances reach an answer (9 detected, 9 completed). `Show me.` is
  accepted through the follow-up bypass, exactly as designed.
- `_thread_anchor` moved to the PostgreSQL question after the switch (a new
  thread), so `it` in the final utterance resolves to PostgreSQL.
- The final prompt contains no `consumer group`.

---

## 10. Correction attack (`10-correction`, 3/3 pass)

| script | result |
|---|---|
| `Explain Kafka replication.` → `Actually I mean PostgreSQL replication.` | 2 questions, 2 answers, `REASON_CORRECTION` |
| `How would you design a batch pipeline?` → `Actually make it streaming.` | 2 questions, 2 answers |
| `What database would you use?` → `No, assume we must use PostgreSQL.` | 2 questions, 2 answers |

Each correction is detected as its own answerable turn (the exact wording
preserved, no merge, no context prefix) and `Relationship.CORRECTION` pulls the
revised task from history. Two answers is correct here: the first question was
already answered before the revision arrived.

---

## 11. Duplicate / race attack (`11-race`, 7/8 pass)

| scenario | questions | answers | understanding calls | answer calls | verdict |
|---|---|---|---|---|---|
| **same final twice** | **2** | **2** | **2** | **2** | **HC-003 (P1)** |
| same partial twice | 0 | 0 | 0 | 0 | ✓ partials never reach detection |
| partial → final → late partial | 1 | 1 | 1 | 1 | ✓ |
| final → duplicate final | (same as row 1) | | | | HC-003 |
| out-of-order finals (later speech clock first) | 2 | 2 | 2 | 2 | ✓ no splice, monotonicity guard works |
| empty final | 0 | 0 | 0 | 0 | ✓ |
| whitespace final | 0 | 0 | 0 | 0 | ✓ |
| STT garbage final (`?? -- ...`) | 0 | 0 | 0 | 0 | ✓ rejected `no_words` |
| garbage then a real question | 1 | 1 | 1 | 1 | ✓ garbage did not hijack the turn |

**HC-003**: an identical final delivered twice produces a second turn, a second
understanding call and a second answer, and the **first answer is not cancelled**
(0 `ANSWER_CANCELLED`, 2 `ANSWER_COMPLETED`) — so the candidate sees the same
question and the same answer twice. `QuestionDetector._is_repeat_of_tail` only
prevents the duplicate being *concatenated* onto itself; nothing rejects it as a
redelivery. No dedupe exists in `app/stt/pipeline.py` either (grepped:
no `_last_final` / dedupe of any kind). Whether real Whisper can emit a duplicate
final is **not established by this pass** — but the WS `question.manual` path
(`app/api/ws.py:137`) forwards whatever the client sends, so a double-submit in
the UI reaches the same code path.

No context corruption was observed in any race scenario.

---

## 12. Rapid-fire attack (`12-rapid-fire` 3/3, `12-invariant` 6/6 — all pass)

| run | detected | answered | duplicates (repeated texts) | missed | premature | wrong context | exceptions | understanding calls | answer calls |
|---|---|---|---|---|---|---|---|---|---|
| 20 questions | 20 | 20 | 0 (each text distinct) | 0 | 0 | 0 | 0 | 20 | 20 |
| 30 questions | 30 | 30 | 5 (the pool of 25 wraps) | 0 | 0 | 0 | 0 | 30 | 30 |
| 50 questions | 50 | 50 | 25 (pool wraps twice) | 0 | 0 | 0 | 0 | 50 | 50 |

Mixed coding / Python / SQL / Kafka / system design / data engineering /
follow-ups / corrections / topic switches.

**Hard invariant holds in all three runs:** understanding calls ≤ questions
detected, answer calls ≤ questions detected — exactly one of each per accepted
question. No rejections, no cancellations, no `ANSWER_ERROR`, no exceptions.

---

## 13. Attachment adversarial testing (`13-attachment`, 9/9 pass)

| scenario | pasted material reached the prompt | answers | verdict |
|---|---|---|---|
| fresh paste, then ask | yes | 1 | ✓ |
| ask, then paste **mid-answer** | yes | 1 (re-asked once, first cancelled) | ✓ |
| paste mid-accumulation | yes | 1 | ✓ |
| **stale** paste (window + 30s) + `needs_attachments=true` | yes | 1 | ✓ **intentional** |
| **stale** paste + `needs_attachments=false` | no | 1 | ✓ |
| two pastes, one question | yes | 1 | ✓ |
| attachment-only context (`Explain this.`) | yes | 1 | ✓ (but held 3s — HC-002) |
| follow-up (`Why?`) inherits the paste | yes | 2 | ✓ |
| paste **after** `ANSWER_COMPLETED` | n/a | 1, no re-ask | ✓ **intentional** |

### BUG-003 — confirmed NOT a bug (as instructed, verified before judging)

A stale attachment *is* bound when the understanding call says
`needs_attachments=true`. This is deliberate and load-bearing:

- `tests/test_attachment_lifecycle.py` parametrises the gap up to `gap=90.0`
  with `STALE = context_attachment_window_ms + 30`, and its docstring states the
  point: the failure it pins down came from real use, where a pasted problem was
  dropped "by a clock the interviewer knew nothing about".
- `test_a_pasted_problem_reaches_the_answer_however_long_the_pause` asserts
  exactly this.
- The safety gate is `understanding.source is UnderstandingSource.LLM` — the
  deterministic fallback cannot revive stale material, so it is never revived
  *merely* by a flag.

My probe reproduces both halves (revived with `true`, declined with `false`),
which matches the documented semantics. `FIX_MODE_REPORT.md` §5 already
retracted this finding; this pass independently confirms the retraction.

**Note on my own first attempt:** I initially scored "ask, then paste" as a
failure. It was my scenario that was wrong — I let the answer complete first,
and `LiveSession` clears `_live_question` on `ANSWER_COMPLETED`
(`app/realtime/session.py:957`) specifically so a post-answer paste does not
re-ask. Corrected to a mid-answer paste, which passes.

---

## 14. Provider routing attack (`14-*`, 57/57 pass)

All mocked. Registry: `{groq, openai, openai_compatible, omnirouter}`.

| provider | model | client built | `provider_name` | `model_name` |
|---|---|---|---|---|
| groq | `openai/gpt-oss-120b` | `GroqClient` | groq | as configured |
| groq | `gpt-5.6-terra` | `GroqClient` | groq | as configured |
| openai | `gpt-5.6-terra` | `OpenAICompatibleClient` | openai | as configured |
| openai | `gpt-5.6-luna` | `OpenAICompatibleClient` | openai | as configured |
| openai_compatible | `arbitrary-model-name` | `OpenAICompatibleClient` | openai_compatible | as configured |
| omnirouter | `gpt-5.6-luna` | `OpenAICompatibleClient` | omnirouter | as configured |
| omnirouter | `arbitrary-model-name` | `OpenAICompatibleClient` | omnirouter | as configured |

- **Provider is explicit** — `client.provider_name` equals the configured
  provider in all 7 cases, and it is only ever used to select a branch.
- **Model is configuration** — `gpt-5.6-terra`, `gpt-5.6-luna`,
  `arbitrary-model-name`, `vendor/some_model:v3`, `llama-4-405b-instruct` all
  accepted verbatim. No allow-list. Blank and whitespace-containing models
  rejected at construction with `ProviderConfigError`.
- **API key is configuration** — `api_key_for(provider)` resolves
  `<provider>_api_key` by name for all four providers, and returns `""` for an
  unknown provider (no exception, no leak).
- **Base URL is configuration** — required for `openai_compatible` and
  `omnirouter` (missing → `ProviderConfigError`); defaulted for `openai`;
  `gateway.example.invalid/v1` and `ftp://…` rejected, `https://…` and
  `http://localhost:1234/v1` accepted.
- **No key-prefix inference** — a `gsk_…` key on `openai` still builds
  `OpenAICompatibleClient`; an `sk-…` key on `groq` still builds `GroqClient`;
  `sk-proj-…` on omnirouter and a prefix-less key on openai_compatible route by
  provider only. Confirmed by reading too: nothing in `registry.py`,
  `deps.py` or either client inspects key *shape*.
- **Unknown provider** → `ProviderConfigError` naming the known providers, not
  a crash.

### The same pipeline code on every provider (`14-understanding-portable`, `14-answer-portable`)

Question Understanding produced an **LLM-sourced** `Understanding` (topic
`kafka`, relationship `new_question`) through **all four** providers, and Answer
Generation parsed the identical answer payload through all four — one
`QuestionUnderstander`, one `Answer` schema, four transports. A client without
`complete_json` degrades Understanding to the deterministic fallback rather than
erroring, as documented.

---

## 15. Provider failure attack (`15-*`, 62/63 pass)

### Normalised classification, OpenAI-compatible transport (real `httpx` code path)

| mocked failure | `LLMErrorKind` | user-facing sentence identical to `messages.message_for` |
|---|---|---|
| 401 | `auth` | ✓ |
| 403 | `auth` | ✓ |
| 429 (+`Retry-After: 7`) | `rate_limit` | ✓, `retry_after_seconds == 7.0` |
| 404 | `model_unavailable` | ✓ |
| 400 mentioning "model" | `model_unavailable` | ✓ |
| 500 | `server` | ✓ |
| 503 | `server` | ✓ |
| `httpx.ReadTimeout` | `timeout` | ✓ |
| `httpx.ConnectError` | `network` | ✓ |
| 200 + body missing `choices` | `malformed` | ✓ |
| 200 + empty content | `malformed` | ✓ |
| 200 + non-JSON content | `malformed` | ✓ |
| 200 + wrong answer schema | `malformed` | ✓ |
| **no API key configured** | **`unknown`** ← should be `not_configured` | **HC-006 (P2)** |

### Groq's SDK exception-name mapping (`15-groq-classification`, 10/10 pass)

`AuthenticationError`→auth, `PermissionDeniedError`→auth, `NotFoundError`→
model_unavailable, `RateLimitError`→rate_limit, `APITimeoutError`→timeout,
`APIConnectionError`→network, `InternalServerError`→server,
`APIResponseValidationError`→malformed, an unknown SDK class→unknown, and a
bare wrapper carrying `status_code=503`→server. Groq's no-key path correctly
returns `not_configured`.

### No secret leakage in any error (`15-no-secret-in-error`, 23/23 clean)

Every mocked provider response and SDK exception carried
`SUPER_SECRET_TEST_KEY_123456789` in its body/message. The `LLMError` the app
raises never contains it — the wording comes from `app.llm.messages` templates
and the provider's own string is discarded (only the exception *class name*
reaches the log).

### No retry storm

- 401 → exactly **1** HTTP request.
- 500 → exactly **1** HTTP request at this layer (retries are delegated to the
  transport/SDK via `max_retries`, not layered on top).

### Cancellation stays cancellation (`15-cancellation`, 3/3 pass)

`generate_answer`, `stream_answer` and the Groq path all re-raise
`asyncio.CancelledError` unchanged when cancelled mid-flight. Never reclassified
as a provider error, so a superseded turn cannot be reported as a failure.

### Understanding is isolated from answer health (`15-understanding-isolation`, 2/2 pass)

- A 401 inside `complete_json` leaves `last_error_kind is None` — a classifier
  blip is not shown in Settings as a broken key.
- A provider that is completely down degrades Understanding to the
  deterministic fallback; the turn still gets an answer.

### No duplicate answer on failure

`18-concurrency` "provider failure then the next question": both turns emit
`ANSWER_ERROR`, both questions are still detected, the session stays alive, and
no turn is answered twice.

---

## 16. Secret leak attack (`16-secret`, 50/50 clean)

Probe secret: `SUPER_SECRET_TEST_KEY_123456789`, planted in
`settings.groq_api_key`, `settings.openai_compatible_api_key`, every
`ProviderConfig`, and every mocked provider error body.

| surface | result |
|---|---|
| `repr(ProviderConfig)` / `str()` / f-string / nested in a dict | **clean** (`api_key=set`) |
| `repr(settings)` / `str(settings)` | **clean** (`***`, via `__repr_args__`) |
| `repr(GroqClient)` / `repr(OpenAICompatibleClient)` | **clean** |
| `message_for(kind, provider, model)` — all 9 kinds × 4 providers (36) | **clean** |
| `app` logger output: client init + 401 failure + Groq auth failure + `log_metric` | **clean** |
| full traceback of a 401 (including the `httpx` cause chain) | **clean** — `HTTPStatusError.__str__` carries URL+status, not the body |
| `GET /api/settings` | **clean** |
| `GET /api/health` | **clean** |
| `GET /api/models/status` | **clean** |
| `PUT /api/providers/groq/key` (write) | **clean** — no echo, not even a masked prefix |
| `GET /api/settings` after a key write | **clean** |
| frontend contracts (`frontend/src/api/contracts.ts`) | **clean** — only `configured: boolean` / `llm_key_configured: boolean`; keys are write-only (`client.ts:191`) |

### Two observations, neither a confirmed leak

- **HC-010 (P3, by design):** `settings.model_dump()` and
  `settings.model_dump_json()` **do** contain keys in plaintext — `__repr_args__`
  redacts `repr`/`str` only. Grepped `app/`: nothing dumps `settings` to a log,
  a response or disk, so there is no reachable path today. Worth a guard if
  anyone ever adds a diagnostics dump.
- **HC-009 (P3, defence in depth):** the `ANSWER_ERROR` WebSocket event forwards
  `str(exc)` verbatim. Driving a `LiveSession` with an LLM that *deliberately*
  raises `LLMError("provider echoed the key …")` puts the secret on the wire.
  No production client builds such a message (the 50 checks above prove the
  wording is template-only), so this is not reachable today — but there is no
  sanitiser between a provider message and the client.

---

## 17. Property-style fuzzing (`17-*`, 0 violations)

Cross product of 25 openers × 8 verbs × 4 determiners × 5 modifiers × 6 nouns ×
7 tails (interrogatives, auxiliaries, verbs, determiners, adjectives, nouns,
prepositions, conjunctions, pronouns), each evaluated **unpunctuated and with a
trailing `?`**:

```
utterances fuzzed:        167,994
detector evaluations:     335,988
I1 violations:            0    (a fragment ending on a dangling function word became COMPLETE)
I2 violations:            0    (a "?"-terminated question was left ACCUMULATING)
exceptions:               0
```

- **I1** — no fragment ending on `if/that/because/and/or/the/a/of/to/is/with/for/
  in/on/at/as/than/while/though/since/unless/until/whether/what/how/why/when/
  where/which/who/whom/about` was ever read as COMPLETE.
- **I2** — Whisper's own `?` always wins; nothing punctuated ever accumulated
  forever.
- No exception, no unbounded accumulation, no crash anywhere in the space.

The nested-clause gap (HC-001) is invisible to these invariants precisely
*because* it does not end on a function word — which is the point: the
deterministic layer has no signal for it. HC-002's over-hold is likewise not an
invariant violation, only a latency cost.

Minimal reproductions for every failing case in this pass are the exact strings
in the tables above; all are single-utterance or two-utterance scripts and all
are in `hc_*_results.json`.

---

## 18. Performance / concurrency (`18-*`, 9/9 pass)

| scenario | result |
|---|---|
| two finals dispatched concurrently (`asyncio.gather`, no await between) | ≤2 answers, correct supersede/cancel, no exception |
| partial + final + partial interleaved concurrently | exactly 1 answer |
| accumulation + attachment concurrently | ≥1 answer, material bound, no exception |
| cancellation then a new question | 1 completed, 1 cancelled, 1 answer call |
| provider failure then the next question | 2 `ANSWER_ERROR`, both detected, session alive |
| **task leak**: 10 turns then `close()` | no growth in `asyncio.all_tasks()` |
| **event-loop blocking**: heartbeat during 20 turns | worst stall **< 250ms** |

No deadlocks, no unhandled exceptions, no ordering violations, no task leaks.
The `asyncio.to_thread` wrapping of every synchronous SQLite call inside
`LiveSession` (added in the previous fix pass) shows up here: a 20-turn burst
never stalls the loop.

**Performance caveat:** the harness uses a zero-delay fake LLM and a `MockRetriever`,
so these are event-loop-health numbers, not latency numbers. The only real
latency measured in this pass is the **3.1–3.3s hold** in HC-002, which is
deterministic and provider-independent.

---

## 19. Failure inventory (prioritised)

### P1 — high

---

**HC-001 — Nested-clause fragment read as COMPLETE → duplicate question and duplicate answer**

- **Severity:** P1
- **Layer:** `app/realtime/prompt_detector.py` (`_has_content_after`, `_looks_incomplete`), surfacing in `question_detector._assess`
- **Exact inputs (each followed 1.5s later by its continuation):**
  - `"Explain how Kafka handles"` + `"partitions?"`
  - `"Explain how Kafka handles"` + `"partitions when brokers fail?"`
  - `"Tell me how the database"` + `"handles replication?"`
  - `"Describe how the system"` + `"should recover from failure?"`
  - `"How would you ensure that the service"` (as a single utterance)
  - `"Can you explain why the consumer"` (as a single utterance)
- **Expected:** the fragment holds; one question, one answer, one understanding call, one answer call.
- **Actual:** the fragment fires immediately as COMPLETE; **2 questions, 2 answers, 2 understanding calls, 2 answer calls**. A wrong answer streams into the panel first.
- **Reproduction:** `./.venv/Scripts/python.exe qa_hardcase/hc_session.py` → section `3c-e2e-real-holds`; detector-only: `qa_hardcase/hc_detector.py` → `3-nested`.
- **Deterministic:** yes (4/6 cases fail on every run with real hold timing).
- **Already known:** partially — `FIX_MODE_REPORT.md` §3 documents `"Explain how Kafka handles"` as an accepted remaining gap. The other three deterministic cases are new instances of the same root cause.
- **Likely root cause:** the deterministic layer measures *how many* words follow a trigger and *which* word ends the sentence. A subordinate wh-clause with a subject but no verb (`how Kafka handles …`, `how the database …`) satisfies both checks. Closing this properly needs a signal for an unsaturated verb argument — e.g. "the utterance ends inside a wh-clause opened after the trigger" — not another word list.

---

**HC-002 — Ordinary short unpunctuated questions are held ~3.1s before any answer starts**

- **Severity:** P1 (latency on the critical path; no wrong answer)
- **Layer:** `app/realtime/prompt_detector.py` — two independent causes
- **Exact inputs and measured hold** (`3d-over-hold`, real timing):

  | utterance | measured |
  |---|---|
  | `What is Kafka` | 3140ms |
  | `When does it rebalance` | 3155ms |
  | `Are there tradeoffs` | 3155ms |
  | `Implement this` | 3156ms |
  | `Optimize this` | 3156ms |
  | `Explain this.` | 3265ms |
  | `Optimize it.` | 3141ms |

  A broader sample of 25 realistic short questions: **10 of 25** are held 3000ms
  when unpunctuated — also `What is idempotency`, `What is normalization`,
  `Is it durable`, `Reverse a string`. **0 of 25** are held when Whisper emits `?`.
- **Expected:** a grammatically complete short question is answered immediately.
- **Actual:** `Finality.ACCUMULATING`, `hold_ms=3000`, answer starts ~3.15s late.
- **Reproduction:** `qa_hardcase/hc_session.py` → `3d-over-hold`; detector-only → `4-interrogative`, `5-imperative`, `7-*`.
- **Deterministic:** yes.
- **Already known:** no.
- **Likely root causes (two, both introduced by the BUG-001 fix):**
  1. `_has_content_after` requires **two** substantive words after an
     `INTERROGATIVE` trigger, and `_TRIVIAL_TAIL_WORDS` discards `is/are/does/
     it/this/that/…`. `"What is Kafka"` therefore has one substantive word and
     is marked `sparse`. The same set makes `"Implement this"`, `"Explain this."`,
     `"Optimize it."` sparse under an imperative trigger, which also contradicts
     `is_referentially_empty`, where `this`/`it` explicitly *do* count as a
     governed object.
  2. `_OPEN_PREMODIFIER` is purely morphological: determiner + a word ending in
     `-ed`/`-ing` at the end of the utterance. Confirmed false positives:
     `"Reverse a string"`, `"What is a string"`, `"What is the encoding"`,
     `"Explain the padding"`, `"What is the meeting"`, `"What is a rating"`,
     `"What is a thing"`. `"reverse a string"` is the single most common coding
     prompt there is.
- **Real-world severity depends on real STT**, since a `?` removes the hold
  entirely. Must be re-measured against real Faster-Whisper output.

---

**HC-003 — A duplicate identical final produces a duplicate answer**

- **Severity:** P1
- **Layer:** `app/realtime/question_detector.py` (`inspect`) / `app/realtime/session.py` (`ask`)
- **Exact input:** `"How would you design a rate limiter?"` delivered as a final,
  allowed to answer, then the **identical** final delivered again 200ms later.
- **Expected:** the redelivery is recognised and dropped; 1 question, 1 answer.
- **Actual:** 2 `QUESTION_DETECTED`, 2 `ANSWER_COMPLETED`, **0 `ANSWER_CANCELLED`**,
  2 understanding calls, 2 answer calls. The same question and the same answer
  appear twice.
- **Reproduction:** `qa_hardcase/hc_session.py` → `11-race` / "same final twice".
- **Deterministic:** yes.
- **Already known:** no.
- **Likely root cause:** `_is_repeat_of_tail` exists only to stop a redelivered
  final being *concatenated* onto itself; there is no "this is the same
  utterance I just accepted" rejection. `app/stt/pipeline.py` has no final-level
  dedupe either. Reachable from the WS `question.manual` path
  (`app/api/ws.py:137`) on a UI double-submit; whether real STT can emit a
  duplicate final is **untested** here.

---

### P2 — medium

---

**HC-004 — `"Show code"` / `"Show Python code"` are never detected**

- **Severity:** P2 (missed question — no answer at all)
- **Layer:** `app/realtime/prompt_detector.py` (`_PROMPT_ANYWHERE`, `_CLAUSE_INITIAL`)
- **Exact inputs:** `"Show code"`, `"Show Python code"` (both rejected `no_question_pattern`); `"Show me the code"` works.
- **Expected:** detected and answered (explicitly named in the mandated §5 list).
- **Actual:** `QUESTION_REJECTED`, no turn, no answer.
- **Reproduction:** `qa_hardcase/hc_session.py` → `5-e2e`; detector-only → `5-imperative`.
- **Deterministic:** yes.
- **Already known:** no.
- **Likely root cause:** `_PROMPT_ANYWHERE` only has `show\s+(?:me|us)`, and
  `show` is absent from `_CLAUSE_INITIAL`'s imperative verb group (which does
  contain `write|implement|build|design|…|handle|optimize|answer`). An
  interviewer saying "show" without an addressee matches neither.

---

**HC-005 — A one-word imperative followed by its object loses the question entirely**

- **Severity:** P2
- **Layer:** `app/realtime/question_detector.py` (`inspect`, the `needs_context` early return)
- **Exact inputs:** `"Explain"` then, 500ms later, `"Kafka replication."` — and `"Optimize"` then `"this query."`
- **Expected:** the pair reaches one answer (either by the fragment being remembered as setup, or by the continuation merging onto it).
- **Actual:** **both utterances rejected, nothing answered.** `"Explain"` is rejected `too_short` and — because the `needs_context` branch returns *before* `_remember_as_context` is reachable — is not kept as setup either. `"Kafka replication."` then matches no pattern and is rejected too.
- **Reproduction:** `qa_hardcase/hc_session.py` → `5b-split-imperative`.
- **Deterministic:** yes.
- **Already known:** no.
- **Likely root cause:** the `too_short` / `no_referent` rejection is an early
  `return` placed above the `may_remember` logic, so short fragments can never
  become context. Ordinary rejections (`no_question_pattern`, `low_confidence`)
  *are* remembered.

---

**HC-006 — `OpenAICompatibleClient.generate_answer` reclassifies its own `NOT_CONFIGURED` error as `UNKNOWN`**

- **Severity:** P2 (wrong, unactionable user-facing message)
- **Layer:** `app/llm/openai_compatible.py` (`generate_answer` / `_post` / `_ensure_client`)
- **Exact inputs:** a `ProviderConfig` with `api_key=""` (and, separately, with `base_url=""`), then `await client.generate_answer("prompt")`.
- **Expected:** `LLMErrorKind.NOT_CONFIGURED` — "No … API key is configured. Add one in Settings to enable answers."
- **Actual:** `LLMErrorKind.UNKNOWN` — "The … request failed."
- **Reproduction:** `qa_hardcase/hc_provider.py` → `15-classification` / "no API key configured".
- **Deterministic:** yes.
- **Already known:** no.
- **Likely root cause:** `_ensure_client()` is called from inside `_post()`,
  which is inside `generate_answer`'s `try`, so the `LLMError` it raises is
  caught by `except Exception` and re-classified by `self._fail`. `stream_answer`
  and `complete_json` call `_ensure_client()` **outside** their `try` and are
  correct; `GroqClient.generate_answer` does the same and is correct. Affects
  `Orchestrator` (the typed `/api/question` path) and `SessionSummarizer`, the
  two `generate_answer` callers.

---

### P3 — low

---

**HC-007 — Ungrammatical user-facing sentences for `openai_compatible`**

- **Severity:** P3 (cosmetic, user-visible)
- **Layer:** `app/llm/registry.PROVIDER_DISPLAY_NAMES` × `app/llm/messages._TEMPLATES`
- **Actual strings:**
  - `"The the configured endpoint request failed."`
  - `"The the configured endpoint model 'gpt-5.6-luna' is not available to this account."`
  - `"No the configured endpoint API key is configured. Add one in Settings to enable answers."`
  - `"the configured endpoint's rate limit was reached."`
- **Root cause:** `display_name("openai_compatible")` is the phrase `"the configured endpoint"`, substituted into templates that already supply an article and expect a proper noun. `groq` / `openai` / `omnirouter` all read correctly.
- **Deterministic:** yes. **Known:** no.

---

**HC-008 — Boundary nondeterminism when the pause equals the hold exactly**

- **Severity:** P3
- **Exact input:** `"How would you design a distributed"` + a **3.0s** pause + `"system that processes millions of events per second?"` (3.0s == `question_hold_incomplete_ms`).
- **Expected:** either outcome is legal; the split was predicted.
- **Actual:** merged into one question in the observed run.
- **Deterministic:** **no** — this is a genuine race between hold expiry and the continuation. Every other pause (100ms…2.5s, 5s) is deterministic and correct. No duplicate answer either way.

---

**HC-009 — `ANSWER_ERROR` forwards a provider exception string to the WebSocket unsanitised**

- **Severity:** P3, defence in depth. **Not a confirmed leak.**
- **Layer:** `app/realtime/session.py` error emission
- **Observed:** an LLM client that deliberately raises `LLMError("provider echoed the key SUPER_SECRET_TEST_KEY_123456789")` puts that string on the wire verbatim.
- **Why it is not currently exploitable:** every real client builds its message from `app.llm.messages` templates and discards the provider's own text (50/50 secret checks clean). There is simply no sanitiser between the two.
- **Deterministic:** yes, for the synthetic client. **Known:** no.

---

**HC-010 — `settings.model_dump()` / `model_dump_json()` contain API keys in plaintext**

- **Severity:** P3, by design / latent
- **Layer:** `app/core/config.py` — `__repr_args__` redacts `repr`/`str` only.
- **Observed:** both dumps contain the probe secret.
- **Reachability:** none today — grepped `app/`, nothing dumps `settings` to a log, an HTTP response or disk; the settings API is write-only for keys and the frontend contract carries only booleans.
- **Deterministic:** yes. **Known:** likely intentional; recorded so a future diagnostics dump does not become a leak.

---

### Environment-only

- **ENV-001** — `keyring` is not installed in `backend/.venv` (it is in
  `requirements.txt`). Causes the 2 deterministic `test_secret_storage` failures.
  Fix: `./.venv/Scripts/python.exe -m pip install keyring`.
- **ENV-002** — the `groq` SDK is not installed either. This causes **no** test
  failure (the import is guarded and returns a "missing support files"
  `NOT_CONFIGURED` error), but it means the real Groq transport cannot be
  smoke-tested on this machine. All Groq results above use a duck-typed fake SDK.

### Flaky

- **FLAKE-001** — `tests/test_stt_pipeline.py::test_final_diverging_from_its_partial_is_logged`.
  Passes alone and in full-suite run 2; failed in run 1. Timing-sensitive
  (`partial_interval_ms=200` must beat the final under full-suite CPU load).
  Not a product failure. Fix if it recurs: drive the partial deterministically
  rather than by interval.

### Known / intentional — verified, not findings

| behaviour | evidence |
|---|---|
| Stale attachment bound when the LLM says `needs_attachments=true` | `tests/test_attachment_lifecycle.py` (parametrised to `gap=90.0`); `FIX_MODE_REPORT.md` §5 |
| A paste after `ANSWER_COMPLETED` does not re-ask | `session.py:955-957` + its docstring; confirmed |
| An abandoned fragment is eventually answered after the hold | `2b-eventual` 6/6 |
| `"Would you choose Kafka"` fires as COMPLETE | grammatically complete; continuation absorbed by `question_continuation_ms` |
| Pure navigation chatter rejected, attached questions answered | `6-*` 17/17 |
| Referential fragments rejected with no context, accepted with a prior question **or** a pending paste | `7-*` 39/45 |
| `select_context` returns the window whole on the deterministic fallback | documented; means a provider outage widens context rather than narrowing it |
| One HTTP request per `generate_answer` (retries delegated to the transport) | `15-no-retry-storm` |

---

## 20. Recommended order for the next fix pass

1. **HC-002** — highest user-visible cost per interview (3.1s of dead air on the
   most ordinary questions), and both root causes are narrow: the
   `_TRIVIAL_TAIL_WORDS` / `is_referentially_empty` contradiction over
   `this`/`it`, and `_OPEN_PREMODIFIER` matching `-ing`/`-ed` **nouns**.
   Re-measure against real Faster-Whisper output first — a `?` removes the
   symptom entirely, so the true exposure is unknown until then.
2. **HC-003** — small, self-contained, and it doubles provider cost when it fires.
3. **HC-001** — the largest design question. Worth deciding explicitly whether to
   accept it as a documented gap (as §3 of the previous pass did for one case)
   or to add a single "utterance ends inside a wh-clause opened after the
   trigger" signal.
4. **HC-006**, **HC-004**, **HC-005** — each a few lines, each independently testable.
5. **HC-007**, **HC-009**, **HC-010** — cosmetic / hardening.
6. **ENV-001** — install `keyring` so the baseline is 1368 green.

Nothing in this pass was fixed, committed or pushed.
