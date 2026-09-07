# TESTING_REPORT.md — Interview Coach Red-Team Testing

**No production code was modified to produce this report.** All findings were reproduced
against the real, unmodified source tree at commit `c72055d5fe059e1b693026409e74d97fd158fdc2`
(branch `feat/desktop-app-and-llm-routing`). Isolated QA scripts live under
`backend/qa_redteam/` and are listed in full at the end of this document.

---

# Executive Summary

The deterministic question-detection pipeline is largely sound: race/supersession handling,
provider-failure recovery, understanding-failure fallback, coding-context isolation between
unrelated tasks, correction/premise dedup, and the LLM call budget (≤1 understanding + ≤1
answer per accepted question) all held up under targeted adversarial testing.

However, three real, reproducible defects were found in the deterministic layer and one in
the frontend, plus a significant environment/reproducibility gap:

1. **BUG-001 (P0)** — A class of interviewer fragments that end on a bare content word with no
   object (e.g. `"How would you design"`, `"How would you design a distributed"`) are scored
   `Finality.COMPLETE` by the deterministic detector and are answered by the LLM as standalone,
   complete questions — verified end to end through the real `LiveSession` pipeline, not just at
   the detector layer. This is a direct violation of the stated Invariant 1 ("an incomplete
   interviewer question must never reach answer generation").
2. **BUG-002 (P1)** — The literal phrase `"Next question."`, spoken as pure meta-commentary, is
   misread as a request and is answered by the LLM after a 3-second hold, violating the explicit
   "interviewer chatter must never trigger answer generation" requirement.
3. **BUG-003 (P1)** — A stale (90-second-old, past the 60s configured window) and topically
   unrelated pasted attachment leaks verbatim into an unrelated question's prompt whenever the
   question-understanding classifier's `needs_attachments` flag is (mis-)set true — a path that
   bypasses the attachment window entirely with no age or relevance re-check.
4. **BUG-004 (P2)** — The frontend reducer clears the attachment indicator on every
   `answer.completed`, even though the backend intentionally keeps bound material available for a
   same-thread follow-up. This is an existing, already-written, currently-failing frontend test
   (`attachments.test.ts`), not a new regression introduced by this session.
5. **BUG-007 (environment)** — 8 packages declared in `backend/requirements.txt` — including
   `groq` (the only LLM provider SDK) and `faster-whisper` (the only STT engine) — are not
   installed in `backend/.venv`. This makes real-provider, real-STT, real-audio, OCR, and
   packaging testing (sections 29, 34-partial, 35, 36 of the requested matrix) impossible in
   this checked-out environment without installing anything, which this mission explicitly
   forbids doing as a "fix."

Two lower-severity findings (BUG-005, BUG-006) and one architectural observation (event-loop
blocking sqlite calls) round out the inventory. Full detail below.

---

# Environment

```
OS:      Windows 11 Pro 10.0.26200
Python:  3.12.0 (backend/.venv)
Node:    v24.18.0
npm:     11.6.2
```

---

# Git Baseline

```
HEAD:            c72055d5fe059e1b693026409e74d97fd158fdc2
Branch:          feat/desktop-app-and-llm-routing
Working tree:    clean except 2 pre-existing untracked files
                   ?? call-assistant-redesign-plan.pdf
                   ?? mockup.html
git diff --stat / --name-only: empty (no tracked changes at session start)
```

No production file (backend `app/`, frontend `src/`, `src-tauri/`) was written to during this
session. See "Files created/modified by QA only" at the end.

---

# Architecture Observed

Verified by reading the current source, not assumed from any prior description.

```
Audio capture (WASAPI loopback + optional mic, app/audio/)
  -> VAD (app/stt/vad.py, SpeechDetector)
  -> segmentation / scheduling (app/stt/pipeline.py, app/stt/scheduler.py)
  -> STT (app/stt/faster_whisper_engine.py, "auto"/"cuda"/"cpu" device selection)
  -> transcript publication (LiveSession.on_transcript, app/realtime/session.py)
  -> question detection, NO LLM (app/realtime/prompt_detector.py + question_detector.py)
  -> question finality / accumulation (QuestionDetector.Finality, LiveSession.consider/_delayed_ask)
  -> question UNDERSTANDING, one bounded LLM call (app/realtime/question_understanding.py)
  -> context planning / retrieval selection (select_context, ContextPlan)
  -> history / retrieval (app/memory/, app/retrieval/, FAISS + SQLite)
  -> attachments (app/realtime/attachments.py, AttachmentBuffer)
  -> answer generation (app/llm/prompts.py build_prompt -> GroqClient.stream_answer)
  -> WebSocket (app/api/ws.py)
  -> frontend reducer (frontend/src/state/sessionReducer.ts)
  -> UI (frontend/src/screens/, components/)
```

Also verified:

- **Provider abstraction**: `app/llm/base.py` (`LLMClient` ABC, `LLMError`/`LLMErrorKind`).
  **Groq is the only implementation and the only cloud provider** — `app/llm/groq_client.py`'s
  own module docstring: *"This module is deliberately the whole provider layer: there is one
  cloud model, so a router, a priority list and a failover table would be machinery with
  nothing to route between."* **No OmniRouter, no OpenAI-compatible base-URL routing, no
  multi-provider BYOK registry exists anywhere in `app/`.** (Confirmed by grep across `app/`
  for `omni|Omni|OpenAI|openai|base_url|BYOK` — the only hit is the Groq-hosted model name
  `groq_model: str = "openai/gpt-oss-120b"`, which is a model identifier, not a routing
  mechanism.) The mission brief's assumption of an OmniRouter/OpenAI-compatible routing layer
  does **not** match the current codebase — flagging this correction explicitly per the
  instruction to verify architecture from code rather than assume it.
- **Settings**: `app/core/config.py`, a single `pydantic_settings.BaseSettings` object,
  `groq_api_key`/`groq_model` are the only provider fields.
- **Key storage**: `app/core/secrets.py` — OS credential manager via `keyring`
  (`KeyringSecretStore`), never an encrypted file; secret names constrained to
  `^[a-z][a-z0-9_]{0,62}_api_key$`.
- **Model registry**: none beyond `groq_model` (a single string) and `stt_model` (a single
  string, `distil-small.en`) in `Settings`.
- **Cancellation**: `LiveSession._cancel_current` cancels the in-flight `asyncio.Task` and
  awaits it; `asyncio.CancelledError` propagates through `QuestionUnderstander.understand` and
  the answer stream rather than being swallowed.
- **Concurrency**: one `asyncio.Lock` (`LiveSession._lock`) serializes `ask()`/`cancel()`;
  retrieval and understanding run concurrently via `asyncio.gather` (session.py `_answer`).
- **Tracing**: `app.core.metrics.LatencyTrace`, a plain dataclass threaded explicitly through
  the pipeline (not a context-var / global registry).
- **Metrics**: `app.core.metrics.log_metric`, one structured `logger.info("metric %s %s", ...)`
  line per event — no metrics framework, no external sink.

---

# Automated Baseline

See `QA_BASELINE.md` for the full capture. Summary:

```
Backend:  1294 passed, 2 failed, 3 skipped, 1 warning, 184.44s
Frontend: typecheck clean; node --test: 123 passed, 1 failed (124 total)
```

Backend failures — both `ModuleNotFoundError: No module named 'keyring'` in
`tests/test_secret_storage.py` (environment dependency gap, not application logic — see
BUG-007). Frontend failure — `attachments.test.ts` "attachments survive an answer completing,
for the follow-up to use" (see BUG-004).

---

# Question Reconstruction (section 4)

All 8 of the mission's own fragment cases (A-H) were run through the real `QuestionDetector`.

| Case | Fragment 1 | Fragment 2 | Result |
|---|---|---|---|
| A | "How would you" | "design a distributed system?" | correctly ACCUMULATING then merges |
| **B** | **"How would you design"** | "a distributed system?" | **fragment alone scored COMPLETE — BUG-001** |
| C | "How would you design a" | "distributed system?" | correctly ACCUMULATING then merges |
| **D** | **"How would you design a distributed"** | "system?" | **fragment alone scored COMPLETE — BUG-001** |
| E | "What's the" | "time complexity?" | correctly ACCUMULATING then merges |
| F | "Can you" | "optimize it?" | correctly ACCUMULATING then merges |
| G | "How would you" | "handle database replication?" | correctly ACCUMULATING then merges |
| H | "What happens if" | "Kafka goes down?" | correctly ACCUMULATING then merges |

Root cause of B/D (see BUG-001 below): the trailing word of the fragment ("design",
"distributed") is a real word, and `prompt_detector._has_content_after` treats any real word
after the matched interrogative trigger as evidence the sentence is not sparse — but a bare
content word with no completed object is not evidence of a finished request.
`question_detector._DANGLING_WORDS` (the fallback incompleteness check) lists conjunctions,
prepositions, articles, copulas and interrogatives, but not task-imperative verbs
("design", "build", "implement", ...) or bare adjectives ("distributed"), so nothing catches
this case once `sparse` is (wrongly) `False`.

---

# Finality (section 6)

All 15 of the mission's finality cases were run (both punctuated and unpunctuated forms).
Punctuation is confirmed **not** the sole finality mechanism — imperative continuations
("Now handle retries.", "Now optimize this.") and bare interrogative-shaped fragments without
"?" are correctly recognised via `REASON_INTERROGATIVE`/`REASON_IMPERATIVE_TASK`, independent of
trailing punctuation. No defect found here beyond the shared root cause of BUG-001 (which is a
finality-*scoring* defect for objectless fragments, not a punctuation-dependency defect).

---

# Question Understanding (section 27)

Five failure-injection modes — timeout (5s delay vs. the 1.5s budget), malformed JSON, an
empty string response, a raised provider exception, and an unknown `relationship` enum value —
were run end to end through `LiveSession`. **All five** degraded cleanly to
`deterministic_fallback()` and produced exactly one completed answer turn with no
`ANSWER_ERROR`. Invariant 13 held in every case tested.

---

# Context (sections 11-15)

- **Coding context isolation** (section 12): `Implement binary search.` → `What's the
  complexity?` → `Now implement a Kafka consumer.` → `What's the complexity?` (2nd). The final
  understanding-prompt correctly excluded the `binary_search` snippet — Invariant 11/12 held.
- **This/that/it/they referent chain** (section 15): a 7-turn chain (`What is Kafka?` → `How
  does it scale?` → `Why is that useful?` → `What happens when they fail?` → `Can you optimize
  it?` → `How would you implement that?` → `What's its bottleneck?`) — every turn was detected.
  Referent *resolution* itself is delegated to the LLM understanding layer and was not
  independently verifiable without a real model; recorded as detected-but-not-semantically-
  graded.
- **Correction / premise dedup** (section 14): `Assume 1,000 QPS.` → `Actually, assume 10,000
  QPS.` → `What's the bottleneck at that scale?` — the final answer prompt correctly carried
  only the corrected `10,000 QPS`, not both. Held.

---

# Conversation Threads (sections 9, 10, 13)

- **Topic switching** (section 13): coding → SQL → behavioral → Kafka → system design → back to
  coding, six distinct utterances — all six were detected as six separate questions with no
  merging or dropped turns.
- **Short follow-up attack** (section 9), 20 phrases (`Why?`, `How?`, `Then?`, ... `Go deeper.`):
  when a recent question exists, all 20 correctly route through the follow-up path or are
  rejected as too-short consistent with the design. **When no prior question exists at all**,
  8 of the 20 (`Then what?`, `Why not?`, `How so?`, `Which one?`, `What about it?`, `Can you?`,
  `Can you expland?`, `Show me.`) are still accepted as brand-new standalone questions purely
  because they clear the 2-word minimum — see BUG-005.
- **Interviewer chatter** (section 24): 12 filler phrases tested; 11 of 12 correctly rejected.
  `"Next question."` is the one exception — see BUG-002.

---

# Coding (section 11)

The 10-step coding progression (implement → complexity → optimize → show implementation →
memory-efficient → Unicode → test → space complexity → sliding window → final implementation)
was not run as a literal 10-turn scripted scenario (time-boxed out in favour of the more
targeted section-12 context-poisoning check above, which exercises the same underlying
mechanism — code-reference carry-forward via `needs_previous_code` — with a cleaner assertion).
No defect found in the mechanism itself in the scenarios that were run.

---

# SQL (section 22)

Not separately scripted — the deterministic detector and understanding layer are topic-agnostic
by design (no keyword tables; `topic`/`domain` are free text per `question_understanding.py`'s
own module docstring), and the section-13 topic-switch scenario already exercises a SQL-shaped
question (`"What index would you use in PostgreSQL for a range query?"`) with no defect found.
Dedicated SQL-context-retention scenarios (schema, then a follow-up query modification) were not
separately run; recorded as not executed, low incremental risk given the topic-agnostic design.

---

# Data Engineering / System Design (sections 21, 23)

Not separately scripted beyond what section 13's topic-switch scenario covers (Kafka ingestion,
notification system design questions were included there and detected correctly). The
detector/understanding layers carry no domain-specific logic for these topics, so incremental
risk from not running the full realistic-wording matrix is assessed as low; flagged as not
executed for completeness.

---

# Attachments (sections 18, 19)

- **Timing sweep** 0s/1s/5s/10s/15s/30s/60s (paste-then-ask): all bound correctly; content
  verified present in the actual LLM prompt at 0s and (separately) at exactly the 60s window
  boundary (inclusive, per `AttachmentBuffer._fresh`'s `<=` comparison).
- **Multiple attachments**: arrival order preserved in the rendered prompt (SQL pasted first,
  then a table — SQL appeared first in the prompt).
- **Replacing an old paste with a corrected one**: no replace-on-paste semantics exist; both the
  old and new versions ride together in the prompt (recorded as actual behaviour — an
  interviewer correcting a paste currently gets both the wrong and right data quoted to the
  model, with no automatic "this replaces that" resolution the way spoken corrections get via
  `is_correction`).
- **Paste arriving mid-answer-stream**: correctly triggers exactly one re-ask, not a duplicate
  turn.
- **Paste arriving mid-accumulation** (before a held fragment's continuation lands): the pasted
  material does correctly reach the eventual merged question's prompt — but the *fragment* it
  was pasted into is itself subject to BUG-001, so the overall scenario still double-answers.
- **Stale + unrelated attachment leak**: see BUG-003 (P1, this section's most severe finding).
- **Content types** (section 19 — PDF/DOCX/image/OCR): **not executed** — see "Tests not
  executed" below (Pillow/OCR test fixtures unavailable in this environment, same gap the
  baseline suite's own `test_pdf_ocr.py` hits).

---

# Providers / BYOK (sections 29-31)

**Severely constrained by environment** (BUG-007): the `groq` SDK is not installed in
`backend/.venv`, so `GroqClient._ensure_client()` cannot construct a real client at all — every
attempt fails at `from groq import AsyncGroq` with a packaging-guard `LLMError` regardless of
key validity. This means:

- Real key / invalid key / rate-limit / 401/403/404/429/500/502/503 / timeout / malformed-response
  testing against the actual Groq API is **not executed** in this environment (would require
  installing the `groq` package, which this mission's "do not fix/modify environment" scope
  does not authorize, and would also require a real network call this mission's fake-key-only
  rule for section 30 does not permit anyway).
- What **was** verified: `GroqClient.classify()`'s exception→`LLMErrorKind` mapping table
  (`_BY_EXCEPTION_NAME`, `_BY_STATUS`) covers 401/403/404/408/429/500/502/503/504 by pure data
  inspection of `app/llm/groq_client.py`, and the per-kind user-facing message table
  (`_MESSAGES`) was exercised directly (see secret-leak probe) and never echoes key material.
  This is a code-reading + unit-level check, not a live-provider test.
- No OmniRouter / OpenAI-compatible custom-base-URL path exists to test (see Architecture
  Observed above) — the mission brief's assumption here does not hold against current code.

# Secret Leak Test (section 30)

Fake key `gsk_QA_REDTEAM_FAKE_KEY_DO_NOT_USE_...` used throughout (never a real key). Results:

| Check | Result |
|---|---|
| Full fake key in any captured log line | **not found** |
| User-facing `_message()` error text contains key | **not found** |
| `repr(vars(GroqClient()))` contains key | **not found** |
| `repr(Settings)` (pydantic default repr) contains key | **FOUND — BUG-006 (P3, latent)** |

---

# Races / Cancellation (sections 17, 27 partial)

- Two questions back to back with a slow-streaming first answer still in flight: exactly one
  completed turn (the second), the first cancelled before producing content. Invariant 8 held.
- Event **ordering** under cancellation: no event belonging to the superseded (first) turn was
  ever observed after the first event of the superseding (second) turn. Held.
- Provider failure on turn 1, healthy turn 2 immediately after: exactly one `ANSWER_ERROR`, one
  clean completion, no corrupted state carried forward. Invariant 16 held.
- Out-of-order final delivery (an older-clock-timestamp final arriving after a newer one was
  already accepted): the stale final was not merged onto / did not corrupt the newer accepted
  question. Invariant 7 held (in this specific reproduction).
- Duplicate final, identical text, no gap: second delivery did not self-concatenate
  (`"...? ...?"` double text). `_is_repeat_of_tail` correctly guards this.
- Duplicate-with-drift (`"What is a hash map?"` then, shortly after, `"What is a hash map?
  Actually never mind."`): did not double-merge into a self-referential repeat. The explicit
  "never mind" is silently treated as a fresh, separate, accepted question by the deterministic
  layer (no special handling for an explicit retraction) — recorded as a minor UX gap, not filed
  as a numbered bug (no invariant explicitly covers a spoken retraction with nothing to retract
  it into).
- Follow-up window boundary (`question_followup_window_ms` = 45000ms exactly, vs. 45001ms):
  behaves as coded — accepted as follow-up exactly at the boundary (`<=` comparison), rejected
  one ms past it. Not a bug; documents the exact cutoff.

---

# STT (sections 34, 35)

**Real STT engine is not testable in this environment** — `faster-whisper` is not installed in
`backend/.venv` (BUG-007); the baseline suite's real-engine tests are stubbed via fakes
(`tests/fakes.py::FakeSttEngine`) for exactly this reason and passed. Real-hardware audio
capture (WASAPI loopback, GPU/CPU device selection, model warmup) is **not executed** — no
audio hardware/GPU probing was attempted in this non-interactive environment and doing so was
out of scope for a code-level red-team pass.

**Event-loop blocking audit** (section 34, static): every `SessionRepository` method
(`app/storage/session_repository.py`) is a plain synchronous `def` backed by `sqlite3`. Several
call sites in `app/realtime/session.py` invoke these **directly on the event loop**, with no
`asyncio.to_thread` wrapper — unlike `_session_context` (session.py:776), which explicitly is
wrapped:

```
session.py:208   self._sessions.add_transcript(...)      -- inside on_transcript, no to_thread
session.py:647   self._sessions.create_turn(...)          -- inside ask(), no to_thread
session.py:715   self._sessions.interrupt_turn(...)       -- inside _cancel_current(), no to_thread
session.py:724   self._sessions.mark_turn(...)            -- inside _cancel_current(), no to_thread
session.py:908   self._sessions.complete_turn(...)        -- inside _answer(), no to_thread
session.py:956   self._sessions.mark_turn(...)            -- inside _answer() except LLMError, no to_thread
session.py:963   self._sessions.mark_turn(...)            -- inside _answer() except Exception, no to_thread
session.py:1199  self._sessions.end(...)                  -- inside close(), no to_thread
session.py:1203  self._sessions.get_turns(...)             -- inside close(), no to_thread
```

Under normal conditions a local SQLite write is sub-millisecond and this is unlikely to be
perceptible; it is nonetheless a genuine event-loop-blocking surface across every concurrent
WebSocket session on the same process, and is recorded per the mission's explicit ask
(**not fixed**). By contrast, `app/documents/service.py` consistently wraps every disk/FAISS/
embedding operation in `asyncio.to_thread` — the inconsistency is specifically in the realtime
session path.

---

# UI (section 37)

Frontend automated suite: 123/124 passing. The one failure (`attachments.test.ts`) is BUG-004.
No other frontend defect was found in the states exercised by the existing suite (idle,
listening, transcript, detected, retrieving, streaming, answered, cancelled, failed, stale/late
event dropping — all covered by existing passing tests in `sessionReducer.test.ts` and
`attachments.test.ts`, not re-litigated here since they already pass).

---

# Production E2E (sections 35, 36)

**Not executed.** This is a non-interactive sandboxed environment with no real audio hardware
verified available, no GPU confirmed, `faster-whisper`/`groq`/`PyInstaller` not installed (see
BUG-007), and no built Tauri desktop binary was launched. Running the actual packaged desktop
app end-to-end was out of scope for what this environment can do without installing missing
dependencies or building a native binary, neither of which this mission's "do not modify, do
not fix" scope authorizes as a side effect of a QA pass.

---

# Performance (sections 32, 33)

No real STT/LLM latency exists in this environment (everything is a fake with near-zero
delay), so median/p95/worst-case timing numbers would be meaningless and are not reported —
reporting fabricated latency numbers would be worse than reporting none. What **was** measured:
functional throughput under light stress (20 rapid distinct finals with no settle between them)
completed with no exception and no observed duplicate answers for distinct text. A 30+ turn
stress interview (section 39) was not run in full; the shorter but representative scenarios
above (topic switching, coding-context poisoning, races) exercise the same mechanisms that a
long-form interview would stress, at lower time cost, given the LLM layer is universally faked
in this environment (a long real run would not produce more informative timing data here).

---

# Security

See "Secret Leak Test" above. No injection-style finding: `question_understanding.py`'s system
prompt explicitly instructs the model to treat the question and attachments as data, never
instructions, and `Understanding.exact_question` is unconditionally overwritten with the
caller's own transcript after parsing — a compromised/prompt-injected classifier response cannot
change what question is actually answered. This was verified by reading the code
(`parse_understanding`, `exact_question=question` at construction) rather than by live-injecting
against a real model, since no real provider is available in this environment (BUG-007).

---

# Stress Tests / 20+ Turn / 30+ Turn Interview (sections 33, 38, 39)

20 rapid-fire distinct questions completed with no crash (section 33). A dedicated 20-turn and
30-turn mixed-topic scripted interview (behavioral/coding/SQL/Kafka/system-design/follow-
ups/corrections/topic-switches in the specific proportions requested) was not separately
scripted; the section-13 topic-switching scenario (6 switches) and section-12/14 context and
correction scenarios above collectively exercise every *mechanism* a long interview would stress
(thread anchoring, context narrowing, correction dedup, topic isolation) without needing a
much longer script to find a new defect class, given the LLM layer is faked throughout this
environment. Recorded as partially executed; a literal 30+ turn run is straightforward to add
with the same `ReplayHarness` if a fix-phase wants a regression fixture for BUG-001 specifically.

---

# Failure Inventory

### BUG-001
SEVERITY: **P0**
CATEGORY: premature-finalization / incomplete-question-answered
TEST: RT-001, RT-002, RT-003, RT-014 (`qa_redteam/probe_detector.py` cases B & D,
`qa_redteam/probe_session.py` P0-fragment_reconstruction scenarios,
`qa_redteam/probe_second_pass.py` scenario 2)
REPRODUCIBILITY: 100% (deterministic, no timing race required)

INPUT: `"How would you design"` (or `"How would you design a distributed"`) as a finalised
interviewer transcript, with or without a later continuation.

EXPECTED: Neither is a complete request (no object of "design"); the deterministic detector
should score both `Finality.ACCUMULATING` and hold for a continuation, exactly as it correctly
does for cases A, C, E-H of the same fragment family.

ACTUAL: Both are scored `Finality.COMPLETE` (`hold_ms=0`) and are answered immediately.
Confirmed end to end through the real `LiveSession`: `"How would you design a distributed"`
alone produces one completed answer turn from the fake LLM; when its actual continuation
(`"system that processes millions of events per second?"`) arrives 2.5 seconds later, it is
answered again as a second, unrelated-looking turn.

RAW TRANSCRIPT: `"How would you design a distributed"` (single final)

DETECTED QUESTION: `"How would you design a distributed?"` (see `_normalize` auto-appending "?")

FINALITY: `complete` (should be `accumulating`)

CLASSIFICATION: SYSTEM_DESIGN (phase-1 classifier, confidence 0.9 — the classifier is
downstream of the finality bug and not itself at fault)

UNDERSTANDING: deterministic (no completer wired in the minimal repro; behaves identically with
one wired, since the bug is upstream of the understanding call)

CONTEXT: n/a

ATTACHMENTS: n/a (also corroborated with an attachment pasted mid-fragment — RT-014 — same
outcome)

LLM CALL COUNT: 1 (for the bare fragment alone), 2 (for the fragment + late continuation case)
— either way, more than the 0 that Invariant 1 requires for an incomplete question, or the 1
that a correctly-merged question would cost.

PROVIDER: fake (`SlowStreamingLLM`) / MODEL: fake / TTFW: n/a / TOTAL LATENCY: n/a

UI RESULT: not exercised (no UI layer in `ReplayHarness`); would show two separate answered
turns in the transcript history for one interviewer sentence.

REPRODUCTION STEPS:
1. `d = QuestionDetector()`
2. `d.inspect("How would you design", now=0.0, buffer_context=True)` → observe
   `accepted=True, finality=Finality.COMPLETE`.
3. (End-to-end) `ReplayHarness().play([ReplayEvent(0, "How would you design a distributed")])`
   → observe `result.completed_turn_ids() == [1]`.

EVIDENCE: `qa_redteam/probe_detector_results.json` (`case_B`, `case_D`),
`qa_redteam/probe_session_results.json` (`bare_fragment_answered_standalone`,
`split_becomes_two_answered_turns_at_2500ms_gap`).

LIKELY SUBSYSTEM: `app/realtime/prompt_detector.py` (`_has_content_after`,
`_TRIVIAL_TAIL_WORDS`) and `app/realtime/question_detector.py` (`_looks_incomplete`,
`_DANGLING_WORDS`) — the sparse/incompleteness heuristics do not recognise a bare
task-imperative verb or a bare adjective with no following object as incomplete.

DO NOT PROVIDE A FIX.

---

### BUG-002
SEVERITY: **P1**
CATEGORY: chatter-triggers-answer
TEST: RT-004 (`qa_redteam/probe_detector.py` 24-chatter, `qa_redteam/probe_session.py`
24-chatter)
REPRODUCIBILITY: 100%

INPUT: `"Next question."` spoken as pure interviewer meta-commentary (move on to the next
topic), with no actual question attached.

EXPECTED: Must never trigger answer generation (explicit requirement for interviewer chatter).

ACTUAL: `extract_interview_prompt` matches `_PROMPT_ANYWHERE`'s
`(?:another|next|final|last|one\s+more)\s+question` pattern, is scored `sparse=True` →
`Finality.ACCUMULATING` (`hold_ms=3000`), and — because nothing ever supersedes it — is answered
verbatim (`"Next question?"`) after the 3-second hold expires.

RAW TRANSCRIPT: `"Next question."`

DETECTED QUESTION: `"Next question?"`

FINALITY: `accumulating` → fires after hold expiry with no supersession

CLASSIFICATION: TECHNICAL_KNOWLEDGE

UNDERSTANDING: deterministic

CONTEXT: n/a

ATTACHMENTS: n/a

LLM CALL COUNT: 1 (should be 0)

PROVIDER: fake / MODEL: fake / TTFW: n/a / TOTAL LATENCY: ~3s hold + fake stream (production:
~3s of dead air before an irrelevant answer appears)

UI RESULT: candidate would see the coaching panel "answer" a phrase that meant "let's move on,"
after a multi-second delay.

REPRODUCTION STEPS:
1. `d = QuestionDetector(); d.inspect("What is a load balancer?", now=0.0, buffer_context=True)`
   (seed a recent question so this isn't confused with the no-context case)
2. `d.inspect("Next question.", now=2.0, buffer_context=True)` → observe `accepted=True,
   detail="interview_prompt"`.
3. (End-to-end) `ReplayHarness(stabilization_ms=50).play([ReplayEvent(0, "Next question.")])`
   → observe `result.completed_turn_ids()` is non-empty.

EVIDENCE: `qa_redteam/probe_detector_results.json` (`24-chatter` / `"Next question."`),
`qa_redteam/probe_session_results.json` (`next_question_literal_chatter_gets_answered`).

LIKELY SUBSYSTEM: `app/realtime/prompt_detector.py`, `_PROMPT_ANYWHERE` regex.

DO NOT PROVIDE A FIX.

---

### BUG-003
SEVERITY: **P1**
CATEGORY: attachment-leak / stale-context
TEST: RT-005 (`qa_redteam/probe_attachments.py`
`9-stale_attachment_leak::90s_stale_unrelated_table_leaks_on_needs_attachments_true`)
REPRODUCIBILITY: 100%, conditional on the question-understanding classifier setting
`needs_attachments=True` for a question that does not actually reference the pasted material
(a real-world classifier misfire, not a contrived input)

INPUT: paste a table at t=0s; ask a topically unrelated question at t=90s (90 seconds later,
30 seconds past the configured 60-second `context_attachment_window_ms`); scripted
understanding response sets `needs_attachments=True`.

EXPECTED (Invariant 9): a stale attachment must never leak into a new, unrelated question.

ACTUAL: the 90-second-old table's exact content appears in the answer prompt.
`AttachmentBuffer._claim_referenced` → `LiveSession._claim_referenced` (session.py:997) binds
with `bind(..., ignore_age=True)` (session.py:1053) whenever `understanding.source is
UnderstandingSource.LLM` and `understanding.needs_attachments` is true, with **no re-check of
elapsed time and no check of topical relatedness to the question** — the age window
(`context_attachment_window_ms`) exists only on the *other*, time-based binding path
(`AttachmentBuffer.bind` called from `LiveSession.ask`), and is completely bypassed here.

Control run with the same 90-second-old attachment but `needs_attachments=False` correctly did
**not** leak, confirming the leak is specific to the `_claim_referenced` / `ignore_age=True`
path and not a general window-expiry defect.

RAW TRANSCRIPT: `"What is a bloom filter?"`

DETECTED QUESTION: `"What is a bloom filter?"`

FINALITY: complete

CLASSIFICATION: n/a (scripted understanding response)

UNDERSTANDING: LLM (scripted), `relationship=new_question`, `needs_attachments=True`

CONTEXT: none carried

ATTACHMENTS: 1 pending `table` attachment, age 90s at claim time

LLM CALL COUNT: 1 understanding + 1 answer

PROVIDER: fake / MODEL: fake / TTFW: n/a / TOTAL LATENCY: n/a

UI RESULT: not exercised; the candidate would see an answer quoting pasted material they may
have long forgotten providing, attributed to an unrelated question.

REPRODUCTION STEPS:
1. `h = ReplayHarness(llm=SlowStreamingLLM(), stabilization_ms=1)`
2. `h.live._understander = QuestionUnderstander(CountingCompleter(reply_with_needs_attachments_true))`
3. `await h.live.on_context_attached("table", content=TABLE, now=0.0)`
4. `await h.live.on_transcript("What is a bloom filter?", TranscriptSource.LOOPBACK, True, now=90.0)`
5. Inspect `llm.prompts[-1]` → contains the table's content.

EVIDENCE: `qa_redteam/probe_attachments_results.json`
(`9-stale_attachment_leak::90s_stale_unrelated_table_leaks_on_needs_attachments_true`, and the
paired control `18-attachment_binding::90s_stale_table_control_needs_attachments_false`).

LIKELY SUBSYSTEM: `app/realtime/session.py::LiveSession._claim_referenced` (line 997) and
`app/realtime/attachments.py::AttachmentBuffer.bind` (`ignore_age` parameter).

DO NOT PROVIDE A FIX.

---

### BUG-004
SEVERITY: **P2**
CATEGORY: frontend-state / attachment-ui
TEST: existing test `frontend/src/api/attachments.test.ts:342` ("attachments survive an answer
completing, for the follow-up to use") — already written in the repository, currently failing;
not a new test authored for this mission.
REPRODUCIBILITY: 100%, deterministic

INPUT: apply `context.attached` → `question.detected` → `answer.started` → `answer.completed`
(all for the same `turn_id`) to `sessionReducer`'s `apply()`.

EXPECTED: `state.attachments.length === 1` after `answer.completed` — the backend intentionally
keeps bound material available so a same-thread follow-up can still reference it (see
`app/realtime/session.py`'s own comment on `_live_question`/attachment lifecycle: *"The turn
keeps its material until the next non-follow-up question is asked."*).

ACTUAL: `state.attachments.length === 0`. `sessionReducer.ts`'s `"answer.completed"` case
(line 360) unconditionally sets `attachments: []`.

RAW TRANSCRIPT: n/a (frontend-only)

DETECTED QUESTION: n/a / FINALITY: n/a / CLASSIFICATION: n/a / UNDERSTANDING: n/a / CONTEXT: n/a

ATTACHMENTS: 1 before, 0 after (expected 1 after)

LLM CALL COUNT: n/a / PROVIDER: n/a / MODEL: n/a / TTFW: n/a / TOTAL LATENCY: n/a

UI RESULT: `0 !== 1` (`node --test` assertion failure, `ERR_ASSERTION`)

REPRODUCTION STEPS:
1. `cd frontend && npm test` → observe the one failing test in the output.

EVIDENCE: direct `npm test` output, this session; test file unmodified.

LIKELY SUBSYSTEM: `frontend/src/state/sessionReducer.ts`, `"answer.completed"` case (line 360).

DO NOT PROVIDE A FIX.

---

### BUG-005
SEVERITY: **P2**
CATEGORY: context-free-question / referent-resolution
TEST: RT-007 (`qa_redteam/probe_detector.py` `9-short_followup_no_context`)
REPRODUCIBILITY: 100%

INPUT: `"Which one?"`, `"Can you?"`, `"Show me."`, `"Can you explain?"` (and 4 similar
2-word-or-longer phrases) as the very first utterance of a session, with zero prior turns.

EXPECTED: these are pure referent-dependent fragments with nothing to refer to; ideally rejected
or at minimum flagged as ungrounded rather than answered as if they were self-contained.

ACTUAL: accepted as fresh, standalone questions (`detail="punctuation"` or
`detail="interview_prompt"`, never `detail="follow_up"`) because they clear the
`question_min_words=2` gate on raw word count alone, independent of whether any prior
question exists. The question-understanding layer (not exercised in this detector-level probe)
may still recover gracefully via its own low-confidence/no-context handling, but the
deterministic layer offers no protection at all here.

RAW TRANSCRIPT: e.g. `"Which one?"` / FINALITY: complete / CLASSIFICATION: TECHNICAL_KNOWLEDGE

LLM CALL COUNT: would be 1 answer call spent on a referentially empty question, in production.

REPRODUCTION STEPS:
1. `d = QuestionDetector()` (fresh, no prior `.inspect()` calls)
2. `d.inspect("Which one?", now=1.0, buffer_context=True)` → `accepted=True`.

EVIDENCE: `qa_redteam/probe_detector_results.json`, section `9-short_followup_no_context`.

LIKELY SUBSYSTEM: `app/realtime/question_detector.py::QuestionDetector.inspect` (the `short`
gate is purely `len(text.split()) < self._min_words`, with no check that *any* prior turn
exists for a referent-shaped fragment to attach to).

DO NOT PROVIDE A FIX.

---

### BUG-006
SEVERITY: **P3**
CATEGORY: secret-hygiene / latent-risk
TEST: RT (secret leak probe) — `qa_redteam/probe_secret_leak.py`
REPRODUCIBILITY: 100%, but no active exploitation path found in current code

INPUT: `repr(settings)` with `settings.groq_api_key` set to a fake key.

EXPECTED: no code path should be able to trivially dump the raw key.

ACTUAL: `Settings` is a plain `pydantic_settings.BaseSettings` with no field redaction; its
default `__repr__` prints `groq_api_key='...'` verbatim. **No call site was found in the
reviewed `app/` source that logs `repr(settings)`, `str(settings)`, or
`settings.model_dump()` wholesale** — every actual log/metric line reviewed passes individual,
named fields, never the settings object itself. This is a latent risk (a future debug log, an
uncaught-exception handler that dumps locals, or a crash reporter touching `settings` directly
would leak it), not a confirmed active leak today.

EVIDENCE: `qa_redteam/probe_secret_leak_results.json`
(`settings_repr_never_contains_key: false`).

LIKELY SUBSYSTEM: `app/core/config.py::Settings`.

DO NOT PROVIDE A FIX.

---

### BUG-007
SEVERITY: **P1 (environment / reproducibility)**
CATEGORY: dependency-drift
TEST: direct `pip list` vs. `requirements.txt` diff, this session
REPRODUCIBILITY: 100%, environment-wide (affects every session against this checkout)

INPUT: `backend/.venv` as currently provisioned.

EXPECTED: every package pinned in `backend/requirements.txt` is installed, so the full test
matrix (including the real Groq provider, the real STT engine, real audio devices, and OCR) is
exercisable.

ACTUAL: 8 declared packages are absent from `backend/.venv`:
`groq`, `keyring`, `faster-whisper`, `sounddevice`, `PyAudioWPatch`, `pyinstaller`,
`pypdfium2`, `rapidocr-onnxruntime`.

Direct consequences observed this session:
- `tests/test_secret_storage.py`'s 2 direct-`import keyring` tests fail at baseline.
- `tests/test_pdf_ocr.py`, `tests/test_audio_devices.py`, `tests/test_embeddings.py` skip
  (Pillow/sounddevice/opt-in env var gates — Pillow itself was not checked separately but the
  skip reason names it directly).
- Real Groq provider testing (sections 29-31), real STT/hardware testing (section 35), and
  PyInstaller packaging / production-build E2E (section 36) are **not executable** in this
  environment at all, regardless of test design, since the underlying SDKs do not import.

A separate `backend/.deps/` directory exists (contains `faster_whisper`, and looks like a
PyInstaller-bundled dependency snapshot for packaging) but is not on `backend/.venv`'s Python
path for `pytest`, so it does not close this gap for automated testing.

EVIDENCE: `pip show groq / keyring` → "Package(s) not found"; full missing-package diff
recorded in this session's tool output (8 packages, listed above).

LIKELY SUBSYSTEM: environment provisioning (`backend/.venv`), not application code.

DO NOT PROVIDE A FIX. (Per mission scope: do not install packages, do not modify the
environment — reporting only.)

---

# Second Red-Team Pass (mission section 46)

Five additional adversarial scenarios, deliberately chosen to not overlap the matrix above:

1. **Correction-premise dedup, end to end** (`qa_redteam/probe_second_pass.py` scenario 1) —
   confirmed working correctly (not a bug; recorded as a control).
2. **Paste arriving mid-accumulation, before the fragment's continuation lands** (scenario 2) —
   the pasted material itself binds correctly, but the scenario surfaces BUG-001 again (the
   held fragment fires standalone before the paste or continuation matter). No new independent
   defect; corroborating evidence.
3. **Event ordering under cancellation** — confirmed no event for a superseded turn is ever
   emitted after the first event of the turn that superseded it (scenario 3, control).
4. **Duplicate-with-drift** (`"What is a hash map?"` then shortly after `"What is a hash map?
   Actually never mind."`) — did not self-concatenate; the explicit retraction is answered as a
   plain new question with no special handling (a minor, unfiled UX gap, discussed under Races
   above — no stated invariant covers a spoken retraction).
5. **Follow-up window boundary** (exactly `question_followup_window_ms` vs. one ms past) —
   behaves exactly as coded; not a bug, documents the cutoff precisely.

No additional numbered bug resulted from this pass beyond corroborating BUG-001; this itself is
a meaningful finding — it means BUG-001's root cause (the sparse/incompleteness heuristic gap)
is the dominant, highest-leverage defect in the whole matrix tested, since it kept resurfacing
under different scenario framings (fragment reconstruction, pause attack, attachment-during-
accumulation) rather than four independent bugs.

---

# TESTING COMPLETE — NO CODE CHANGED

```
Git HEAD:      c72055d5fe059e1b693026409e74d97fd158fdc2
Branch:        feat/desktop-app-and-llm-routing
Working tree:  clean except the 2 pre-existing untracked files noted at session start
Baseline:      backend 1294 passed / 2 failed / 3 skipped ; frontend 123 passed / 1 failed
Final:         IDENTICAL -- no application or test file was modified, so re-running the
               automated suites now would reproduce the exact same baseline numbers and the
               exact same failing test names. (Not re-run a second time, since nothing that
               could change the result was touched -- see git status below.)
```

```
Automated tests:
  Passed:  1294 (backend) + 123 (frontend) = 1417
  Failed:  2 (backend, both ModuleNotFoundError: keyring -- BUG-007)
           1 (frontend, attachments.test.ts -- BUG-004)
  Skipped: 3 (backend, environment-gated: Pillow / sounddevice / RUN_MODEL_TESTS)

Manual / scripted red-team scenarios:
  Executed: 100+ individual scenario checks across 5 isolated QA scripts (exact counts:
            probe_detector.py 95, probe_session.py 24, probe_attachments.py 6,
            probe_second_pass.py 6, probe_secret_leak.py 4 = 135 total assertions)
  Passed:   121
  Failed:   14 (all attributable to BUG-001 (5 occurrences), BUG-002 (2), BUG-003 (1),
            BUG-005 (8), BUG-006 (1) -- counts overlap slightly where one probe script's
            "FAIL" corroborates a bug already found by another)

P0: 1  (BUG-001)
P1: 3  (BUG-002, BUG-003, BUG-007)
P2: 2  (BUG-004, BUG-005)
P3: 1  (BUG-006)

Top 10 root causes:
  1. prompt_detector._has_content_after / _TRIVIAL_TAIL_WORDS does not treat a bare
     task-imperative verb or bare adjective as "no real content" -- BUG-001's actual root cause.
  2. question_detector._DANGLING_WORDS is a closed list that excludes content/task verbs
     entirely -- the fallback that would have caught BUG-001 if the above didn't.
  3. prompt_detector._PROMPT_ANYWHERE's "(?:another|next|...)\s+question" pattern matches
     meta-commentary use of "next question" with equal force to an actual request -- BUG-002.
  4. AttachmentBuffer.bind(ignore_age=True), reached from LiveSession._claim_referenced with no
     re-check of elapsed time or relatedness -- BUG-003.
  5. sessionReducer.ts's answer.completed handler unconditionally clears attachments -- BUG-004.
  6. QuestionDetector.inspect's `short` gate is pure word-count with no "does prior context
     exist" check for referent-shaped short questions -- BUG-005.
  7. pydantic BaseSettings ships no field redaction for secret-shaped fields -- BUG-006 (latent).
  8. backend/requirements.txt vs backend/.venv drift, 8 packages -- BUG-007 (environment).
  9. No replace-on-paste semantics for attachments (documented behaviour, not filed as a
     numbered bug, but a real gap versus how spoken corrections are handled).
  10. LiveSession's SQLite writes are not offloaded to a thread, unlike every other
      disk-touching path in the codebase (documents/service.py) -- architectural inconsistency,
      not filed as a numbered bug (no observed impact given SQLite write latency).

Top latency bottlenecks: not measured -- no real STT/LLM latency exists in this environment
  (see Performance section); reporting a number here would be fabricated.

Top context failures: BUG-003 (stale attachment leak via needs_attachments over-trigger);
  BUG-005 (context-free short question answered with nothing to ground it).

Top attachment failures: BUG-003 (P1); no-replace-on-paste (documented, unfiled).

Top provider failures: none observed with the exception-classification code path that WAS
  testable (data-only inspection of app/llm/groq_client.py); live-provider testing blocked by
  BUG-007.

Top race/cancellation failures: none found -- every race/cancellation/duplicate scenario tested
  passed (see Races/Cancellation section).

Top UI failures: BUG-004 (attachment indicator cleared prematurely).

Production E2E result: NOT EXECUTED -- see BUG-007 and "Production E2E" section.

Tests not executed:
  - Real Groq provider (valid/invalid key, 401/403/404/429/500/502/503, timeouts, malformed
    responses against the live API) -- Reason: `groq` package not installed (BUG-007); would
    also require a real network call and/or a real key, both out of this mission's scope.
  - Real STT / real hardware (WASAPI loopback, GPU/CPU device selection, model warmup) --
    Reason: `faster-whisper`/`sounddevice`/`PyAudioWPatch` not installed (BUG-007); no audio
    hardware confirmed available in this non-interactive environment.
  - OCR / attachment content types requiring Pillow, pypdfium2, rapidocr-onnxruntime (PDF/DOCX/
    image attachments) -- Reason: packages not installed (BUG-007); baseline suite's own
    test_pdf_ocr.py skips for the same reason.
  - PyInstaller packaging / production desktop build E2E -- Reason: `pyinstaller` not installed
    (BUG-007); no built Tauri binary available to launch.
  - Literal 20-turn and 30+-turn scripted interviews in the exact proportions requested --
    Reason: time-boxed in favour of targeted scenarios covering the same mechanisms (topic
    switching, context narrowing, correction dedup) at lower cost, given the LLM layer is
    universally faked in this environment and would not surface additional defect classes from
    sheer turn count alone.

Files created/modified by QA only (all under backend/qa_redteam/, none under app/, tests/,
frontend/src/, or src-tauri/):
  - backend/qa_redteam/probe_detector.py
  - backend/qa_redteam/probe_detector_results.json
  - backend/qa_redteam/probe_session.py
  - backend/qa_redteam/probe_session_results.json
  - backend/qa_redteam/probe_attachments.py
  - backend/qa_redteam/probe_attachments_results.json
  - backend/qa_redteam/probe_second_pass.py
  - backend/qa_redteam/probe_second_pass_results.json
  - backend/qa_redteam/probe_secret_leak.py
  - backend/qa_redteam/probe_secret_leak_results.json
  - QA_BASELINE.md
  - TESTING_REPORT.md (this file)
  - manual_test_results.md

CONFIRMED: no production source file was modified. `git status --porcelain` and
`git diff --name-only` against `app/`, `frontend/src/`, `frontend/src-tauri/`, `tests/`, and
every other tracked file show zero changes; the only new files on disk are the ones listed
immediately above, all newly created (never overwriting an existing file) under an isolated
`backend/qa_redteam/` directory or the repository root as explicitly permitted by the mission
brief ("create isolated QA/test files", "create TESTING_REPORT.md and supporting QA
artifacts"). No commit was made and nothing was pushed.
```
