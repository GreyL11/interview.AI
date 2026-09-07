# QA Baseline

Captured before any red-team testing. No production code touched to produce this.

## Git

```
HEAD:            c72055d5fe059e1b693026409e74d97fd158fdc2
Branch:          feat/desktop-app-and-llm-routing
Working tree:    clean except 2 untracked files present at session start
                   ?? call-assistant-redesign-plan.pdf
                   ?? mockup.html
git diff --stat: (empty)
git diff --name-only: (empty)
```

## Environment

```
OS:      Windows 11 Pro 10.0.26200
Python:  3.12.0 (backend/.venv)
Node:    v24.18.0
npm:     11.6.2
```

## Backend automated suite

Command: `./.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --tb=no -rf`

```
1294 passed, 2 failed, 3 skipped, 1 warning in 184.44s
```

### Failing tests (exact names)

```
FAILED tests/test_secret_storage.py::test_the_backend_is_resolved_directly_when_discovery_finds_nothing
FAILED tests/test_secret_storage.py::test_a_machine_with_no_backend_at_all_reports_unavailable
```

Root cause (confirmed, not fixed): both tests do `import keyring` directly and it
raises `ModuleNotFoundError: No module named 'keyring'`. `keyring>=25.0` is listed
in `backend/requirements.txt` but is **not installed** in `backend/.venv` —
an environment/dependency drift, not an application logic bug. Every other
test in `test_secret_storage.py` passes because the production code
(`app/core/secrets.py`) imports `keyring` lazily inside `KeyringSecretStore`
methods and those paths are otherwise exercised through fakes/monkeypatching;
only these two tests import the real module directly.

### Skipped tests (3, exact names)

```
SKIPPED [1] tests/test_pdf_ocr.py:27: Pillow is needed to build the test PDFs
SKIPPED [1] tests/test_audio_devices.py:125: requires sounddevice
SKIPPED [1] tests/test_embeddings.py:54: set RUN_MODEL_TESTS=1 to exercise the real ONNX model
```

All three are environment-gated (missing optional dependency / opt-in env var),
not the `slow` OCR marker registered in `pytest.ini` — that marker ran and
passed as part of the 1294.

## Frontend

Not yet run at this checkpoint — see TESTING_REPORT.md for frontend
typecheck/test results captured later in the same session.

## Notes on architecture (verified from current code, not assumed)

- Provider layer is **Groq only**. `app/llm/groq_client.py`'s own docstring:
  "Groq is reached through the official SDK. This module is deliberately the
  whole provider layer: there is one cloud model." `app/core/config.py` has a
  single `groq_api_key` / `groq_model` pair. No OmniRouter, no OpenAI-compatible
  base-URL routing, no BYOK multi-provider registry exists anywhere in
  `app/`. (Grepped for `omni|Omni|OpenAI|openai|base_url|BYOK` across `app/` —
  the only hits are the `groq_model` default value `"openai/gpt-oss-120b"`
  itself, which is a Groq-hosted model name, not an OpenAI route.)
- Secrets: `app/core/secrets.py` — OS credential store via `keyring`, never an
  encrypted file. Names constrained to `^[a-z][a-z0-9_]{0,62}_api_key$`.
- Question pipeline (verified by reading, not assumed):
  `app/realtime/prompt_detector.py` (deterministic sentence-level pattern
  extraction, no LLM) → `app/realtime/question_detector.py` (`Finality`:
  ACCUMULATING / POTENTIALLY_COMPLETE / COMPLETE, hold timers) →
  `app/realtime/session.py` (`LiveSession.consider` / `_delayed_ask` /
  `ask` — accumulation, cancellation, attachment binding) →
  `app/realtime/question_understanding.py` (one bounded LLM classification
  call per accepted turn, `deterministic_fallback` on any failure) →
  `select_context` (relationship-driven history narrowing) → `build_prompt`
  → `GroqClient.stream_answer`.
