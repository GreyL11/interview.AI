# Building and packaging Interview Coach (Windows)

> None of this has been executed. It was written against the official Tauri v2,
> Vite and PyInstaller project layouts, and the parts that could be verified
> without installing anything were verified — see "What is already proven".

## Prerequisites

| Tool | Why | Notes |
|---|---|---|
| Python 3.12+ | Backend | Already used by the backend venv |
| Node 20+ | Frontend build | Node 24 verified here |
| Rust (stable, MSVC toolchain) | Tauri shell | `rustup default stable-x86_64-pc-windows-msvc` |
| VS Build Tools — "Desktop development with C++" | Links the Rust binary | ~3–6GB, may need admin |
| WebView2 Runtime | Renders the UI | Preinstalled on Windows 11 |
| PyInstaller | Freezes the backend | `pip install pyinstaller` |

## First-time setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install faster-whisper sounddevice pyinstaller
copy .env.example .env

cd ..\frontend
npm install
```

## Development

Two processes. The Tauri shell spawns the backend itself in debug builds
(`python -m app` from `backend/`), so normally you only need:

```bash
cd frontend
npm run tauri dev
```

To run them separately — useful when iterating on the backend:

```bash
cd backend && .venv\Scripts\uvicorn app.main:app --port 8000 --reload
cd frontend && npm run dev
```

With no `API_TOKEN` set, the token check is disabled and the UI falls back to
port 8000 with a fixed dev token, so the two halves line up automatically.

## Production build

```bash
# 1. Freeze the backend into a onedir tree
cd backend
pyinstaller packaging/interview-coach-backend.spec
#   -> backend/dist/interview-coach-backend/

# 2. Build the app; tauri.conf.json copies that tree in as a resource
cd ..\frontend
npm run tauri build
#   -> src-tauri/target/release/bundle/nsis/Interview Coach_0.1.0_x64-setup.exe
```

## How the two processes fit together

```
Tauri shell (Rust)
  ├─ picks a free port + mints a 32-char token
  ├─ spawns the backend with --port --token --parent-pid --data-dir
  ├─ blocks reading stdout until {"ready":true,...}   (no sleeps)
  ├─ injects window.__BACKEND__ = {port, token} before page scripts run
  └─ on close: POST /shutdown → 5s grace → kill → kill reported PID
```

Three independent shutdown layers, because each covers a case the others miss:

1. **POST `/shutdown`** — the normal path; closes sessions, then exits.
2. **Kill after grace** — for a wedged backend. Also kills the *reported* PID,
   because a venv `python.exe` on Windows is a launcher stub whose real
   interpreter is a grandchild (measured: child 55980 vs actual 53976). Without
   this, dev-mode shutdown orphans the server.
3. **Win32 Job Object** (`KILL_ON_JOB_CLOSE`) — for when the shell is killed and
   never gets to ask. Windows does not kill children with their parent, so
   without this a force-quit leaves a backend holding the microphone and the
   SQLite database.

The backend additionally runs a **parent watchdog**: if `--parent-pid`
disappears, it exits on its own. Verified working — an orphaned backend
self-terminated 3.4s after its parent was killed.

## Data and model layout

Everything user-specific lives under `%LOCALAPPDATA%\com.interviewcoach.desktop\`,
passed to the backend as `--data-dir`:

```
documents/{document_id}/original_file
faiss/index.faiss
metadata/app.db
models/hf/        ONNX embedding model   (~87MB, first use)
models/whisper/   CTranslate2 STT model  (~250MB, first live session)
```

**Models are not bundled.** They download on first use. That keeps the installer
at roughly 250–400MB instead of well over a gigabyte, and lets the user change
the speech model without reinstalling. It also means the first run needs
network access — behind a proxy that blocks Hugging Face, point the settings at
a pre-downloaded directory instead.

Writing under `%LOCALAPPDATA%` rather than the install directory is deliberate:
a per-user NSIS install can land in a location the app cannot write to, which
would break ingestion in a way that is confusing to diagnose.

## Unsigned builds

The MVP ships unsigned. On first run users see **"Windows protected your PC"** —
they must click **More info → Run anyway**. This is expected; document it
wherever the installer is distributed.

To sign later, add a `windows.certificateThumbprint` (and optionally
`digestAlgorithm`, `timestampUrl`) under `bundle` in `tauri.conf.json`. Do not
commit the certificate or its password.

Related notes:
- The spec uses **onedir, not onefile**. onefile re-extracts to `%TEMP%` on every
  launch, which is slow and is exactly what antivirus heuristics flag.
- **UPX is disabled.** It trips antivirus far more often than the space is worth.
- First launch may be slow while Defender scans the freshly installed tree. The
  readiness timeout is 90s for this reason.

## What is already proven

Verified on this machine without installing anything:

- Backend starts via `python -m app --port --token --parent-pid --data-dir`
- Readiness line is emitted on stdout only once the socket is accepting
- `/health` stays open; every other route rejects a missing or wrong token
- WebSocket rejects a bad token and accepts a good one
- `question.detected → answer.started → …` flows over the socket
- `POST /shutdown` exits cleanly in 0.3s
- Orphaned backend self-terminates 3.4s after its parent dies
- 295 backend tests, 39 frontend reducer tests

## What is NOT proven

Everything requiring an install:

- `npm install`, `tsc --noEmit`, `vite build` — no typecheck or bundle has run
- No React component has been rendered
- `cargo build` — the Rust shell has never been compiled
- PyInstaller has never been run; the spec's hidden imports and data files are
  the usual set for this stack but are unverified
- No NSIS installer has been produced or installed
- WASAPI loopback capture and real Whisper transcription (pending since M4)
