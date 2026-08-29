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

One command from `frontend/`:

```bash
cd frontend
npm run desktop:build
```

That runs the three steps in the only order that works -- the backend must be
frozen *before* Tauri bundles it, because `tauri.conf.json` lists
`backend/dist/interview-coach-backend` as a bundled resource, so a stale or
missing tree silently produces an installer with no backend inside it:

```bash
# 1. Freeze the backend into a onedir tree
cd backend
pyinstaller --noconfirm packaging/interview-coach-backend.spec
#   -> backend/dist/interview-coach-backend/

# 2. Typecheck + bundle the frontend
cd ..rontend
npm run build
#   -> frontend/dist/

# 3. Build the shell and the installer
npm run tauri:build
#   -> frontend/src-tauri/target/release/bundle/nsis/Interview Coach_0.1.0_x64-setup.exe
```

## How the two processes fit together

```
Tauri shell (Rust)
  ├─ refuses to run twice (single-instance; focuses the existing window)
  ├─ picks a free port + mints a 32-char token
  ├─ spawns the backend with --port --token --parent-pid --data-dir
  ├─ drains stderr into a 40-line ring buffer (an undrained pipe fills
  |    and blocks the backend mid-startup -- a hang that looks exactly
  |    like a slow model load)
  ├─ blocks reading stdout until {"ready":true,...}   (no sleeps)
  ├─ then polls GET /health until the API actually answers
  ├─ emits backend://startup and backend://status to the UI per stage
  ├─ injects window.__BACKEND__ = {port, token} before page scripts run
  └─ on close: POST /shutdown → 5s grace → kill → kill reported PID
```

Startup reaches the UI as **named stages**, never a percentage: the shell
cannot know how long model loading will take, and a fake bar that stalls at
80% is worse than an honest sentence. On failure the UI shows the reason plus
**Retry** and **Open Logs Folder** -- the log path comes from the backend's own
readiness line, so the two can never point at different directories.

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
logs/interview-coach.log   rotating, 2MB x 3 -- what "Open Logs Folder" opens
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

Run on this machine (an office laptop, so no audio and no provider calls):

- Backend starts via `python -m app --port --token --parent-pid --data-dir`
- Readiness line is emitted on stdout only once the socket is accepting, and
  now also carries `logs_dir` / `data_dir`
- Rotating file log is created under `data_dir/logs/`
- `/health` stays open; every other route rejects a missing or wrong token
- `POST /shutdown` exits cleanly
- Orphaned backend self-terminates after its parent dies
- Backend test suite, frontend reducer tests, `tsc --noEmit`, `vite build`

## What is NOT proven

**The Rust shell has never been compiled.** Rust is not installed on the
machine this was written on, so `cargo check` could not run. Every change in
`src-tauri/src/` — single-instance, stderr draining, `/health` confirmation,
startup events, `retry_backend`, `open_logs_folder` — is unverified by a
compiler. Expect to fix compile errors on the first `npm run desktop:build`.

Also unproven here, by design (office laptop):

- PyInstaller has never been run; the spec's hidden imports and data files are
  the usual set for this stack but are unverified
- No NSIS installer has been produced or installed
- The startup screen has never rendered against a real Tauri event stream —
  only the browser fallback path (which skips it) was exercised
- WASAPI loopback capture and real Whisper transcription
- Groq / Gemini provider calls from the packaged app
