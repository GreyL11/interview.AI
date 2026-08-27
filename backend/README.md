# Interview Coach — Backend (Phase 1)

Local-first interview practice engine. Phase 1 is the core intelligence pipeline only:

```
Question -> Classifier -> Router -> Retrieval Interface -> Gemini -> Answer Validator -> Structured Response
```

No audio, no speech-to-text, no frontend, no FAISS yet — those are later phases.

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in GEMINI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

## Endpoints

- `GET /health` — liveness check.
- `POST /question` — `{"question": "...", "session_id": "optional"}` -> structured classification + answer.

## Architecture

- `intelligence/classifier.py` — rule-based classification (no LLM call; fast, deterministic, testable offline).
- `intelligence/router.py` — maps a question category to a route (RAG, REASONING, CODING, SQL, FOLLOW_UP).
- `retrieval/` — `Retriever` interface; `MockRetriever` is the Phase 1 implementation (returns no context). Swap in a FAISS-backed retriever later without touching callers.
- `llm/` — `LLMClient` interface; `GeminiClient` is the Phase 1 implementation. Swap providers without touching the orchestrator.
- `memory/session_memory.py` — in-process, per-session conversation history for FOLLOW_UP questions. Not persisted; resets on restart.
- `intelligence/orchestrator.py` — wires the pipeline together end to end.
- `intelligence/answer_validator.py` — rejects empty answers, flags unsupported first-person-past claims when no personal context was retrieved.
