# Call Assistant — Backend

Local-first interview practice. Everything except LLM reasoning runs on your machine:
audio, speech-to-text, documents, embeddings, vector search, and session history never
leave the device. Groq receives only the current question, retrieved context, and a
bounded slice of conversation memory — never raw audio, never your document library.

> **Product boundary.** This is for mock interviews, practice, and settings where AI
> assistance is explicitly permitted. It is not built to conceal assistance from an
> interviewer.

## Status

| Milestone | Scope | State |
|---|---|---|
| Phase 1 | Classifier, router, orchestrator, Groq, validator | Complete |
| M1 | Document ingestion, ONNX embeddings, FAISS, SQLite, retrieval | Complete |
| M2 | Session persistence, bounded memory, summarization | Complete |
| M3 | WebSocket realtime, streaming, cancellation | Complete |
| M4 | Audio capture, VAD, faster-whisper | Complete, **hardware unverified** |
| M5 | Live wiring, settings/devices/models API | Complete, **hardware unverified** |
| M6–M7 | Tauri + React frontend, Windows packaging | Not started |

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then set `GROQ_API_KEY` in `.env` (the packaged app uses the OS credential store instead).

Live audio needs two extra packages that are **not** in the base install:

```bash
pip install faster-whisper sounddevice
```

Without them everything still works except live capture — practise by typing questions
(`question.manual` over the WebSocket, or `POST /question`).

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```

272 tests, fully offline and deterministic: fake embedder, fake LLM, replayable fake
audio source, scripted VAD. No network, no API key, no microphone required.

To exercise the real ONNX embedding model (downloads ~90MB on first run):

```bash
set RUN_MODEL_TESTS=1 && pytest tests/test_embeddings.py
```

## Architecture

```
Documents ─ parse ─ chunk ─ embed ─┬─ FAISS (vectors)
                                   └─ SQLite (text, metadata, vector ids)

Audio ─ VAD ─ Whisper ─ final transcript ─ classify ─ route ─┬─ retrieve
                                                             └─ Groq ─── validate ─ WS events
```

Key interfaces, each with one real implementation — swap without touching callers:
`Retriever`, `EmbeddingProvider`, `VectorStore`, `DocumentParser`, `Chunker`,
`AudioSource`, `SttEngine`, `SessionMemory`, `LLMClient`.

STT inference scheduling, partial-transcript policy, latency metrics, and
recommended per-hardware configuration are covered in
[PERFORMANCE.md](PERFORMANCE.md).

### Decisions worth knowing

**SQLite is the source of truth; FAISS only holds vectors.** Retrieval joins vector ids
back to chunks and requires the parent document to be `READY`. Partially-ingested,
failed, or deleted content is therefore invisible *by construction* rather than by
filtering — no soft-delete column, no index rebuild.

**No torch anywhere.** Embeddings run on ONNX Runtime and STT on CTranslate2. That keeps
the eventual installer at roughly 250–400MB instead of 1.5–2.5GB.

**Only final transcripts reach the LLM.** Partial transcripts are display-only, which is
the entire debounce strategy — no amount of interim churn can trigger a provider call.

**Device attribution, not diarization.** Mic and loopback are captured as separate
streams. Question detection runs only on loopback (the interviewer); the microphone is
transcribed for your own review and never answered.

**Cancellation is two-sided.** Each answer carries a `turn_id`; a new question cancels
the previous task and both server and client drop events from superseded turns.

**`context_found`, not `requires_personal_context`.** The fabrication warning fires when
retrieval actually returned nothing — so an invented "I led the migration" against an
empty knowledge base gets flagged, which is exactly when it matters most.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness |
| POST | `/question` | One-shot question → structured answer |
| POST | `/documents?filename=&knowledge_type=` | Upload (raw body, not multipart) |
| POST | `/documents/{id}/ingest` | Parse → chunk → embed → store |
| GET/DELETE | `/documents`, `/documents/{id}` | List, fetch, delete |
| POST/GET | `/sessions`, `/sessions/{id}` | Create, list, detail |
| POST | `/sessions/{id}/end` | End a session |
| GET/PUT | `/settings` | Config (the API key is write-only) |
| GET | `/audio/devices`, `/models/status` | Setup screen data |
| WS | `/ws/session/{id}?token=&since_seq=` | Live session |

### WebSocket events

Server → client: `session.started`, `session.status`, `transcript.partial`,
`transcript.final`, `question.detected`, `question.rejected`, `answer.started`,
`answer.retrieving`, `answer.delta`, `answer.completed`, `answer.cancelled`,
`answer.error`, `session.ended`, `error`.

Client → server: `question.manual`, `answer.cancel`, `audio.start`, `audio.stop`,
`session.stop`, `ping`.

Reconnect with `?since_seq=N` to replay missed events — the session lives in the
backend, so a dropped socket pauses the view, not the run.

## Storage

```
data/
  documents/{document_id}/original_file
  faiss/index.faiss
  metadata/app.db
  models/hf/         ONNX embedding model
  models/whisper/    CTranslate2 STT model
```

Nothing under `data/` is committed.

## Pending hardware verification

These paths are implemented and unit-tested behind their interfaces, but have **not**
been run against real hardware in this environment, because `sounddevice` and
`faster-whisper` are not installed here:

- Opening a real microphone through PortAudio
- **WASAPI loopback capture** — the path that hears the interviewer. Device naming
  varies by driver and this is the most likely thing to need adjustment
- Real Whisper transcription accuracy and latency, and the CUDA-vs-CPU probe
- Silero VAD threshold tuning against real room noise (`VAD_THRESHOLD`,
  `VAD_SILENCE_MS`)

To verify: install the two packages above, `GET /audio/devices` to confirm a loopback
device appears, then start a session and send `audio.start`.
