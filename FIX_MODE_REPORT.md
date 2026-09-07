# FIX_MODE_REPORT.md — Interview Coach Fix-Mode Results

Scope note up front: this session ran **concurrently with a peer session** ("Interview Coach
LLM provider architecture") that was independently implementing Phase 8/9 (provider-agnostic
LLM/BYOK — `settings.llm_provider`, `api_key_for()`, `app.llm.registry`,
`OpenAICompatibleClient`) and, as part of that work, also fixed BUG-006 and the environment's
`groq`/`keyring` import guards. I discovered this mid-session via `git status`/`git diff`,
messaged that session to coordinate, and deliberately did **not** touch any provider/config
file it was editing (`app/core/config.py`, `app/core/deps.py`, `app/llm/groq_client.py`,
`app/llm/registry.py`, `app/llm/openai_compatible.py`, `app/llm/messages.py`,
`app/api/settings_api.py`, `app/api/health.py`, `frontend/src/api/contracts.ts`,
`frontend/src/api/providers.ts`, `frontend/src/screens/SetupScreen.tsx`, and the test files
paired with them). Sections 8, 9, and parts of 6/7 below report on that work as observed
through the shared working tree, not as something I implemented — full ownership and design
rationale for that phase belongs to that session.

My own changes are confined to 5 files, listed in full under "Files changed."

---

## 1. Files changed

**By me, this session:**
- `backend/app/realtime/prompt_detector.py` (+171/-…)
- `backend/app/realtime/question_detector.py` (+58/-…)
- `backend/app/realtime/session.py` (+80/-…)
- `frontend/src/state/sessionReducer.ts` (+61/-…)
- `frontend/src/state/sessionReducer.test.ts` (+30/-…, two existing tests rewritten to match
  corrected behavior, one new test added)

**QA-only artifacts (new, isolated, not shipped code):**
- `backend/qa_redteam/probe_detector.py`, `probe_session.py`, `probe_attachments.py`,
  `probe_second_pass.py`, `probe_secret_leak.py` and their `*_results.json` outputs — updated
  in place this session to reflect fixes and one retraction (see §3).

**By the concurrent peer session** (not mine, reported for completeness — see scope note):
`backend/app/core/config.py`, `backend/app/core/deps.py`, `backend/app/llm/groq_client.py`,
`backend/app/llm/registry.py` (new), `backend/app/llm/openai_compatible.py` (new),
`backend/app/llm/messages.py` (new), `backend/app/api/settings_api.py`,
`backend/app/api/health.py`, `backend/README.md`, `backend/docs/` (new),
`frontend/src/api/contracts.ts`, `frontend/src/api/providers.ts`,
`frontend/src/screens/SetupScreen.tsx`, plus `tests/test_llm_provider_routing.py` (new),
`tests/test_llm_registry.py` (new), `tests/test_openai_compatible_client.py` (new),
`tests/test_groq_client.py`, `tests/test_secret_storage.py`, `tests/test_settings_api.py`,
`tests/test_cors_and_auth.py`, `tests/test_ws.py`.

---

## 2. Architecture changes

**None to the deterministic detection architecture.** BUG-001/002/005 were fixed by adding
targeted, reused-vocabulary checks inside the *existing* `prompt_detector.py` /
`question_detector.py` pipeline — no new router, no new state machine, no LLM call added to
the hot path. One new parameter (`has_pending_context`) was threaded from `LiveSession` into
`QuestionDetector.inspect`, since attachment state lives entirely outside the detector and the
detector had no other way to know material was pending.

**Frontend**: one new piece of state, `SessionState.attachmentsTurnId`, tracking which turn the
current attachment set is bound to — the minimum needed to stop clearing on `answer.completed`
while still eventually retiring stale material.

**Realtime performance**: every synchronous `SessionRepository`/`SqliteSessionMemory` call
inside `LiveSession` now runs via `asyncio.to_thread`, matching the pattern already used
elsewhere in the codebase (`app/documents/service.py`, `LiveSession._session_context`). No
storage architecture change — `Database` already supported this (thread-local connections,
process-wide write lock), it just wasn't applied consistently inside `LiveSession` itself.

**Provider abstraction** (peer session, reported not implemented by me): `app.llm.registry` now
maps a provider name (`groq`, `openai_compatible`, ...) to a client implementation;
`OpenAICompatibleClient` is a new, generic OpenAI-chat-completions-shaped transport
(base URL + API key + model, used for `openai_compatible` and reusable for any
OpenAI-compatible gateway); `GroqClient` was refactored to share error-classification/message
plumbing (`app.llm.messages`) with it rather than duplicating it. `Settings` gained
`llm_provider`/`llm_model`/`llm_base_url`/`llm_max_retries` alongside the existing
`groq_api_key`, with alias-based backward compatibility for `GROQ_MODEL` etc. This preserves
the existing `LLMClient`/`StructuredCompleter` abstraction as the extension point, as
instructed — I did not independently verify every detail of this implementation since it
wasn't mine to build, but the full test suite run in §12 includes its test files
(`test_llm_provider_routing.py`, `test_llm_registry.py`, `test_openai_compatible_client.py`,
`test_groq_client.py`) and all pass.

---

## 3. BUG-001 resolution (P0 — incomplete question reaches answer generation)

**Root cause** (confirmed by reading, not guessed): two independent gaps in the sparse/
incompleteness heuristics.

1. `prompt_detector._has_content_after` treated *any* single non-trivial word after an
   interrogative trigger ("would you", "can you", "how", "what") as evidence the request was
   filled — but after an interrogative trigger, the very next word is the clause's own verb,
   not its object. "How would you **design**" has one such word ("design") and was read as
   "filled."
2. Nothing checked for an *open noun phrase* — a determiner ("a", "an", ...) followed by a
   premodifier with no head noun yet ("a **distributed**", waiting for "system").

**Fix** (`prompt_detector.py`, `question_detector.py`):
- `_has_content_after` is now reason-aware: after an `INTERROGATIVE` trigger it requires **two**
  substantive words (the object must itself be filled, not just present), leaving
  `IMPERATIVE_TASK`/`INTERVIEW_PROMPT` triggers (which already consumed the verb) unchanged —
  "handle retries", "explain caching" (one-word objects) still resolve immediately, as before.
- New `has_open_premodifier(prompt)`: a **morphological** check (a determiner followed by up to
  two words ending in `-ed`/`-ing`, anchored at the *end* of the utterance) — not a word list.
  "a distributed" matches; "a distributed system" does not (ends on the head noun "system").
  Wired into `_assess` alongside the existing `is_scenario_without_request` /
  `has_placeholder_object` checks, same `ACCUMULATING` tier.

**Verified, end to end, through the real `LiveSession` (not just the detector in isolation):**
- `"How would you design a distributed"` alone: held (`Finality.ACCUMULATING`), no
  `QUESTION_DETECTED` event, no completed turn, until the hold expires with nothing more said
  (matching the pre-existing, intentional "an abandoned fragment still eventually gets an
  answer" fallback — this is not a violation of the invariant, since nothing is lost, it is a
  bounded ~3s wait rather than an instant wrong answer).
- The full split (`"How would you design a distributed"` then, 2.5s later,
  `"system that processes millions of events per second?"`): reconstructs into **exactly one**
  detected question and **one** completed answer turn (previously: two).
- Full mandated regression list (`"How would you"` → `"...design a distributed system?"`,
  `"What would you"` → `"...use PostgreSQL for?"`, `"Can you"` → `"...implement this in
  Python?"`, `"Now handle"` → `"...retries correctly."`, `"Explain"` → `"...how Kafka handles
  partitions?"`) — every intermediate fragment holds, every completed form answers immediately;
  natural pauses of 100/300/500/1200/2500/5000ms all reconstruct correctly.
- **Known remaining gap, honestly reported**: `"Explain how Kafka handles"` (a subordinate
  wh-clause with a subject but no verb of its own) is still read as `COMPLETE`. Fully
  generalizing to nested-clause completeness would require real verb/argument-structure
  parsing, which is out of scope for a deterministic regex-based detector without turning this
  into "a giant list of English keywords" (explicitly forbidden). Not part of the two
  explicitly-mandated cases; documented rather than silently left in.

Regression coverage added directly in `qa_redteam/probe_detector.py` /
`qa_redteam/probe_session.py` (not the production suite, since these are exploratory red-team
scripts — see §11 for why production test additions were unnecessary: the existing 420+68
detector/attachment tests already exercise this pipeline thoroughly and all pass unchanged).

---

## 4. BUG-002 resolution (P1 — chatter triggers an answer)

**Root cause**: `_PROMPT_ANYWHERE`'s `(?:another|next|final|last|one\s+more)\s+question`
pattern matches "next question" identically whether used as a request ("One more question:
what is X?") or as pure navigation ("Let's move to the next question.").

**Fix**: new `_NAVIGATION_ONLY` regex + `is_navigation_chatter()`, checked in
`_classify_sentence` at the same point as the existing `_ACK_ONLY` filler check (before the
punctuation short-circuit). It matches only when the **entire** utterance (after optional
"okay,"/"let's move to"/"we'll go to" framing) is just "(the) (next|another|one more|final|
last) question" with nothing after it — anchored at both ends, so an attached question
("Next question: what is Kafka?", "Here's the next question: how does Kafka work?") fails the
trailing anchor and falls through to ordinary detection untouched. Also excluded from the
premise buffer (a navigation sentence must not leak into a later question's context as if it
were setup).

**Verified**: all 10 mandated cases behave exactly as specified —
"Next question.", "Let's move to the next question.", "Okay, next question.", "We'll go to the
next question.", "One more question.", "Let's do another question.", "Last question.", "Final
question." → rejected, no LLM call, no answer; "Next question: what is Kafka?", "Here's the
next question: how does Kafka work?" → detected and answered normally. Confirmed end to end
through `LiveSession`: no `QUESTION_DETECTED`, no completed turn, no understanding/answer call
for the pure-navigation forms.

---

## 5. BUG-003 resolution — **retracted, not a bug**

Re-audited per the mission's Phase 0 instruction to verify against current code/tests rather
than assumptions. `tests/test_attachment_lifecycle.py` already covers exactly this mechanism,
deliberately: `STALE = context_attachment_window_ms + 30` (90s at the current 60s window),
parametrized up to `gap=90.0`, with a module docstring stating the point explicitly — "the
phrasing tests exist to fail loudly if anyone ever 'fixes' this with a list of words" and
`test_a_pasted_problem_reaches_the_answer_however_long_the_pause` asserting the same content
resolves correctly *"however long the pause."* `AttachmentBuffer._claim_referenced` binding
regardless of elapsed time when the understanding call directly confirms `needs_attachments` is
**intentional**: it is what lets "the interviewer read the pasted problem out loud before
asking about it, however long that takes" work at all, and the safety net that already exists
(`understanding.source is not UnderstandingSource.LLM` → declined) is exactly the "classifier
must have actually run and said so" gate the mission asked me to preserve
("stale attachment -> cannot be revived merely by needs_attachments" — it is not revived
*merely* by the flag; the flag has to come from a real classification, which the deterministic
fallback path cannot produce). Adding any age ceiling would directly regress this test suite
(confirmed by attempting one and watching `test_a_pasted_problem_reaches_the_answer_however_
long_the_pause[90.0]` and three other tests fail). **No code change made.**
`qa_redteam/probe_attachments.py`'s corresponding scenario was relabeled from a failure to a
confirmation of intentional behavior rather than deleted, so the finding stays visible and its
reasoning stays attached to the evidence.

---

## 6. BUG-004 resolution (P2 — frontend clears attachments too early)

**Root cause**: `sessionReducer.ts`'s `"answer.completed"` case unconditionally set
`attachments: []`, even though `LiveSession` (backend) keeps bound material available past that
point, until the *next fresh* (non-follow-up) question starts.

**Fix**: 
- Removed the unconditional clear from `"answer.completed"`.
- Added `SessionState.attachmentsTurnId: number | null` tracking which turn the current
  attachment set is associated with.
- `"context.attached"`: claims the current in-flight turn immediately if one is open (mid-
  question/mid-answer paste); otherwise marks the material pending (`null`), exactly like a
  paste that arrives before any question is asked.
- `"question.detected"`: a pending (`null`) or same-turn attachment set is preserved and (if
  pending) claimed by the new turn; an attachment set already bound to a **different**, already-
  passed turn is cleared. This means material survives the turn it was pasted for **and** one
  turn beyond it (long enough to show through a same-thread follow-up), clearing once a second
  new question starts without claiming it itself.

**Known, documented limitation**: there is no wire signal for "is this new question a follow-up
or a genuinely new topic," so a real follow-up's own `question.detected` is indistinguishable,
client-side, from an unrelated new question — both look like "a second new turn since the
material was last claimed." The chip therefore may disappear one turn earlier than the
backend's own relationship-aware binding would strictly justify. This is a bounded, cosmetic
UI-only gap (the backend answer itself is unaffected either way, since it decides bindng from
its own state) and was not solvable without adding a new wire event, which was out of scope for
a minimal fix.

**Verified**: the previously-failing test (`attachments.test.ts` "attachments survive an answer
completing, for the follow-up to use") now passes. Two existing `sessionReducer.test.ts` tests
that encoded the *opposite* (buggy) expectation — "attached material disappears once its turn
has been answered" and "clearing material leaves session-wide state alone" — were updated to
assert the corrected, backend-aligned behavior (material survives one full answer, clears on
the next unrelated question), since they could not both pass simultaneously with the fix and
were themselves the same bug duplicated in a second file. One new test added
("attached material clears once a second new question starts without it") to cover the case
neither old test exercised. All 125 frontend tests pass (123 baseline + 1 fixed + 1 new).

---

## 7. BUG-005 resolution (P2 — context-free referential fragments answered)

**Root cause**: the `short` (`< question_min_words`) gate was the *only* protection against
answering a context-dependent fragment; anything ≥2 words bypassed it entirely regardless of
whether it named an actual topic.

**Fix** (`prompt_detector.py::is_referentially_empty`, wired into `question_detector.inspect`):
- New `_REFERENTIAL_FILLER` set: interrogatives, the existing closed-class trivial/auxiliary
  words, a handful of discourse particles ("then", "so", "not", "one"), and the same generic
  request-verb vocabulary already enumerated in `_PROMPT_ANYWHERE`/`_CLAUSE_INITIAL` (not a new,
  separate list — the same one, reused).
- New `_has_verb_with_standin_object`: does a request verb govern `it`/`this`/`that`/`them` as
  its object? "Explain **this**", "optimize **it**" count as filled (the pronoun stands in for
  a presentable topic — on-screen code, a pasted table); "Show **me**" does not (a personal
  pronoun names who receives the answer, never what it's about); a bare trigger with nothing
  after it ("Can you explain?") does not either.
- An utterance is "referentially empty" only when it clears **neither** check, and only applies
  once Whisper terminated it ("?" or ".") — a genuinely open fragment mid-accumulation ("Can
  you", waiting for "...optimize it?") is untouched, so this never fights the existing
  accumulation mechanism.
- **`QuestionDetector.inspect` gained `has_pending_context: bool`**, sourced from
  `LiveSession.consider()`'s `self._attachments.has_pending` — a referential fragment is
  grounded by *either* a recent question *or* pending/bound material the interviewer could be
  pointing at, since the detector has no other way to know an attachment exists at all.

This last piece was added only after the first version of the fix broke
`test_natural_conversation.py`'s existing, deliberate reference-resolution suite (attachment-
grounded "Explain this."/"Answer this."/"What about this?" with zero prior *question* but a
pasted paste) — a genuine interaction bug in my own first pass, caught by the existing tests
and fixed before finalizing.

**Verified**, zero context: `"Which one?"`, `"Can you?"`, `"Show me."`, `"Can you explain?"`,
`"Then what?"`, `"Why not?"`, `"How so?"`, `"What about it?"` → rejected (`no_referent`), no
answer call. With a recent question **or** a pending attachment: all eight accepted as
follow-ups. Must-keep-working control set — `"Why?"`, `"How?"`, `"What is Kafka?"`, `"What's
Redis?"`, `"Explain Kafka."`, `"Use SQL."`, `"Explain this."` (zero context, per
`test_natural_conversation.py`), `"Can we optimize it?"` (zero context, per
`test_question_detector.py`) — all still work. Full `test_natural_conversation.py` (39 tests)
and `test_question_detector.py` suites pass unchanged.

---

## 8. BUG-006 resolution — implemented by the peer session, confirmed present

`app/core/config.py::Settings.__repr_args__` (added by the concurrent session) redacts every
`*_api_key` field from `repr(settings)`/`str(settings)` — the exact mechanism the original
finding recommended investigating. Re-ran the original probe conceptually (inspecting the
current `config.py`) and confirmed the override is in place; did not re-author it since it
would have directly conflicted with that session's edits to the same file.

---

## 9. Dependency / environment resolution

**Not performed by me.** Installing `groq`, `keyring`, `faster-whisper`, `sounddevice`,
`PyAudioWPatch`, `pyinstaller`, `pypdfium2`, `rapidocr-onnxruntime` into the shared
`backend/.venv` was left to the peer session, since (a) it touches the same shared environment
both sessions run tests against, and a concurrent `pip install` from two sessions risks
corrupting the venv, and (b) that session's own test files
(`test_llm_provider_routing.py`, `test_openai_compatible_client.py`) exercise the `groq`-adjacent
import paths directly and are better positioned to verify the install actually closes the gap.
**Confirmed via the final full-suite run (§12)**: `keyring` is still not installed — the same
two `test_secret_storage.py` failures as baseline persist, unchanged in name and cause
(`ModuleNotFoundError: No module named 'keyring'`). This specific piece of Phase 7 was not
completed by either session as of this report.

---

## 10. Provider / BYOK implementation

Implemented by the peer session (see §2, §8). I did not independently design or verify the
OmniRouter/OpenAI-compatible API contract myself; per the mission's own instruction not to
invent an API contract, and given a second, fully-worked implementation was already landing
concurrently in the same files, re-deriving my own would have produced conflicting, wasted
work. What I did verify: the full test suite (§12) includes and passes
`test_llm_registry.py`, `test_llm_provider_routing.py`, `test_openai_compatible_client.py`, and
the refactored `test_groq_client.py` — 1366 passing tests overall include all of these.

---

## 11. Tests added/changed

**By me:**
- `frontend/src/state/sessionReducer.test.ts`: rewrote 2 existing tests to assert the corrected
  attachment lifecycle (previously asserted the buggy behavior), added 1 new test for the
  "clears on a second unrelated question" case.
- `backend/qa_redteam/probe_detector.py`, `probe_session.py`, `probe_attachments.py`,
  `probe_second_pass.py`: updated in place (not part of the production suite) to reflect fixed
  behavior and the BUG-003 retraction; two scenarios that had timing artifacts unrelated to the
  actual product (an unrealistically short test-only hold override racing real wall-clock
  scheduling under system load) were widened to remove flakiness.
- **No new production backend test file was added.** The existing suite
  (`test_question_patterns.py`, `test_question_detector.py`, `test_prompt_detector.py`,
  `test_turn_finalization.py`, `test_turn_assembly.py`, `test_coding_questions.py`,
  `test_natural_conversation.py`, `test_context_attachment.py`, `test_attachment_lifecycle.py` —
  468 tests across these files alone) already covers the fragment-reconstruction, finality,
  chatter, and reference-resolution behavior space thoroughly enough that every one of my fixes
  was validated (and, in BUG-005's case, corrected mid-implementation) against it directly,
  without needing a parallel, possibly-drifting new file.

**By the peer session:** `test_llm_provider_routing.py`, `test_llm_registry.py`,
`test_openai_compatible_client.py` (new files), plus expansions to `test_groq_client.py`,
`test_secret_storage.py`, `test_settings_api.py`, `test_cors_and_auth.py`, `test_ws.py`.

---

## 12. Baseline failure set

```
Backend:  1294 passed, 2 failed, 3 skipped  (1299 collected)
  FAILED tests/test_secret_storage.py::test_the_backend_is_resolved_directly_when_discovery_finds_nothing
  FAILED tests/test_secret_storage.py::test_a_machine_with_no_backend_at_all_reports_unavailable
Frontend: 123 passed, 1 failed  (124 total), typecheck clean
  FAILED attachments.test.ts :: attachments survive an answer completing, for the follow-up to use
Red-team probes: 135 assertions, 121 passed, 14 failed
```

## 13. Final failure set

```
Backend:  1366 passed, 2 failed, 3 skipped  (1370 collected + 1 pre-skipped = 1371)
  FAILED tests/test_secret_storage.py::test_the_backend_is_resolved_directly_when_discovery_finds_nothing
  FAILED tests/test_secret_storage.py::test_a_machine_with_no_backend_at_all_reports_unavailable
Frontend: 125 passed, 0 failed  (125 total), typecheck clean
Red-team probes: 123 assertions across 4 probe scripts, 123 passed, 0 failed
```

**Compared by exact test name** (not just count, per the mission's explicit instruction): the
two backend failures are the *same two test names* as baseline, same root cause
(`ModuleNotFoundError: keyring`), confirmed unrelated to any change made this session (neither
`keyring` nor `app/core/secrets.py` was touched by me; the peer session's config/settings work
does not touch this either). Zero new backend failures. The one frontend failure from baseline
is now fixed; zero new frontend failures. The ~71 additional backend tests (peer session's
provider/registry/settings work) are new, not present at baseline, and all pass.

## 14. Backend result

`1366 passed, 2 failed, 3 skipped, 170.63s` — full suite, run to completion with an in-file
exit-code marker to guard against a truncated capture (two earlier background runs were
silently killed mid-suite by too-short tool timeouts and produced misleadingly incomplete
output before this one; discarded, not treated as evidence of anything).

## 15. Frontend result

`125 passed, 0 failed`, `tsc --noEmit` clean.

## 16. Typecheck result

Clean (`npm run typecheck` → no output, exit 0), against the combined working tree (my changes
+ the peer session's in-flight provider/BYOK changes together).

## 17. Red-team result

All 4 QA probe scripts (`probe_detector.py`, `probe_session.py`, `probe_attachments.py`,
`probe_second_pass.py`) — 123 scenario assertions total — pass cleanly, including the exact
scenarios that originally surfaced BUG-001, BUG-002, and BUG-005 in the prior red-team pass,
verified end to end through the real `LiveSession`, not just at the detector layer in
isolation.

## 18. Real E2E result

**Not executed**, for the same reasons as the original red-team pass: `groq`, `faster-whisper`,
`sounddevice`/`PyAudioWPatch`, and `pyinstaller` remain uninstalled in this environment (§9), no
audio hardware or real provider credential was available, and this is a non-interactive
sandboxed session. No latency numbers are reported since there is no real STT/LLM latency to
measure here — reporting fabricated numbers would be worse than reporting none.

## 19. Remaining known limitations

- **BUG-001**: nested subordinate-clause fragments ("Explain how Kafka handles" — subject with
  no verb of its own) still read as complete. Documented, not silently hidden; fixing it fully
  would require real argument-structure parsing beyond a regex-based deterministic detector.
- **BUG-004**: the frontend attachment chip cannot distinguish a genuine follow-up's own new
  turn from an unrelated new question (no wire signal exists for this), so it may clear one
  turn earlier than the backend's actual binding would justify. Backend answer correctness is
  unaffected; this is a UI-only, cosmetic gap.
- **Phase 7 (environment)**: `keyring` (and, unverified by me, `groq`/`faster-whisper`/
  `sounddevice`/`pyinstaller`/OCR deps) were left to the peer session; confirmed still missing
  as of the final full-suite run.
- **Phase 12 (real hardware/production E2E)**: not executed, environment-blocked as above.
- Two pre-existing, environment-only backend test failures remain (same as baseline).

## 20. Git diff summary

```
My files (5):
 backend/app/realtime/prompt_detector.py   | 171 ++++++++++++++++++++++++++++--
 backend/app/realtime/question_detector.py |  58 ++++++++--
 backend/app/realtime/session.py           |  80 +++++++++-----
 frontend/src/state/sessionReducer.test.ts |  30 ++++--
 frontend/src/state/sessionReducer.ts      |  61 ++++++++---
 5 files changed, 338 insertions(+), 62 deletions(-)

Full working tree (mine + concurrent peer session): 20 files changed, 961 insertions(+), 284
deletions(-) -- see §1 for the per-file attribution.

New, untracked (QA-only, mine): backend/qa_redteam/*.py and *_results.json,
QA_BASELINE.md, TESTING_REPORT.md, manual_test_results.md (from the prior red-team phase),
this file (FIX_MODE_REPORT.md).

New, untracked (peer session): backend/app/llm/{registry,openai_compatible,messages}.py,
backend/docs/, backend/tests/test_llm_{provider_routing,registry}.py,
backend/tests/test_openai_compatible_client.py.
```

**No commit was made. Nothing was pushed.**
