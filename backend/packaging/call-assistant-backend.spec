# PyInstaller spec for the backend sidecar.
#
#   cd backend && python -m PyInstaller --noconfirm packaging/call-assistant-backend.spec
#
# onedir, not onefile: onefile re-extracts the whole tree to %TEMP% on every
# launch, which is slow and is exactly the behaviour antivirus heuristics flag.
# The resulting folder is shipped as a Tauri resource.
#
# Models are deliberately NOT bundled. They are downloaded on first use into the
# per-user data directory, which keeps the installer at a few hundred MB instead
# of well over a gigabyte, and lets the user swap models without a reinstall.
#
# ---------------------------------------------------------------------------
# Every hidden import below exists because something is imported by *name at
# runtime* and PyInstaller's static analysis cannot see it. Nothing is here
# "just in case": an unexplained hidden import is indistinguishable from a
# forgotten one, and the pile grows until nobody dares remove any of it.

import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# ---------------------------------------------------------------------------
# Build-environment guard.
#
# This is the single most important thing in this file. `collect_submodules`
# and `collect_data_files` return an EMPTY LIST for a package that is not
# installed -- they warn, they do not fail. So building with the wrong
# interpreter silently produces an installer that is missing whole features,
# and the first evidence is a user reporting that their API key does not
# survive a restart.
#
# That is exactly what shipped: the build ran on an interpreter without
# `keyring`, so the frozen app logged "No module named 'keyring'" and reported
# secret_store_available=False. Fail the build instead.
REQUIRED = [
    ("groq", "the only cloud LLM provider"),
    ("keyring", "persists the API key in Windows Credential Manager"),
    ("keyring.backends.Windows", "the Credential Manager backend itself"),
    ("win32ctypes.core", "what the Windows keyring backend calls into"),
    ("fastapi", "the HTTP API"),
    ("uvicorn", "the HTTP server"),
    ("faiss", "document search index"),
    ("onnxruntime", "embedding model runtime"),
    ("tokenizers", "embedding tokenizer"),
    ("huggingface_hub", "downloads both local models on first use"),
    ("faster_whisper", "speech to text"),
    ("ctranslate2", "what faster-whisper runs on"),
    ("pypdf", "PDF document parsing"),
    ("docx", "DOCX document parsing"),
    ("pypdfium2", "renders scanned PDF pages so they can be OCR'd"),
    ("rapidocr_onnxruntime", "reads scanned PDFs"),
    ("certifi", "CA bundle for HTTPS to Groq and Hugging Face"),
]

import importlib.util  # noqa: E402

missing = [
    f"  - {name}  ({why})"
    for name, why in REQUIRED
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "\n\nRefusing to build a backend with missing dependencies.\n\n"
        f"Interpreter: {sys.executable}\n\n"
        "Not importable here:\n" + "\n".join(missing) + "\n\n"
        "Each of these would produce a package that starts fine and then fails\n"
        "at runtime in a way that looks like an application bug.\n\n"
        "Fix: build with the project virtualenv, e.g.\n"
        "    ..\\venv\\Scripts\\python.exe -m PyInstaller --noconfirm "
        "packaging/call-assistant-backend.spec\n"
    )

hidden = [
    # uvicorn resolves these by name at runtime, so static analysis misses them.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "app.__main__",
]

# faster-whisper reaches for its feature-extraction and VAD modules lazily.
hidden += collect_submodules("faster_whisper")

# keyring finds its backend through package *entry points*, which PyInstaller
# does not follow. Two things are needed and they fix different halves:
#   - the submodules, so the backend classes exist in the bundle at all
#   - the metadata, so keyring's own entry-point discovery still finds them
# app.core.secrets also imports keyring.backends.Windows directly as a fallback
# for when discovery still comes up empty in a frozen build.
hidden += collect_submodules("keyring")
hidden += ["keyring.backends.Windows", "keyring.backends.fail"]

# The Windows keyring backend picks its FFI layer by name at import time
# (ctypes or cffi), so neither branch is statically visible.
hidden += collect_submodules("win32ctypes")

# faiss loads its compiled extension through a try/except import chain in
# faiss/loader.py, none of which is statically resolvable.
hidden += collect_submodules("faiss")

# ---------------------------------------------------------------------------
# Data vs. binaries: the distinction that broke the shipped build.
#
# A native library collected as *data* is copied verbatim. PyInstaller does not
# scan it, does not follow its dependencies, and does not relocate it. The
# result loads far enough to import and then fails with
#
#   DLL load failed while importing onnxruntime_pybind11_state:
#   A dynamic link library (DLL) initialization routine failed.
#
# which is what the shipped build did: `collect_data_files("onnxruntime")` swept
# up onnxruntime.dll, so document ingestion failed on every document, and the
# Silero VAD that runs on the same runtime failed with it. Collecting the same
# file through both pipelines is worse still -- two copies race to the same
# destination and the unprocessed one can win.
#
# The rule, enforced below rather than remembered:
#   native libraries  -> collect_dynamic_libs (binaries)
#   everything else   -> collect_data_files   (datas), with natives excluded
NATIVE = ["*.dll", "*.pyd", "*.so", "*.dylib"]

datas = []
# The Silero VAD model ships inside faster-whisper as an ONNX asset; without
# this the packaged app has no voice activity detection.
datas += collect_data_files("faster_whisper", includes=["assets/*"])
# python-docx cannot create or open a document without its bundled default
# template, which is package data rather than an import.
datas += collect_data_files("docx", excludes=NATIVE)
# The CA bundle httpx uses to verify TLS. Without it every HTTPS call --
# Groq, and both model downloads -- fails with a certificate error.
datas += collect_data_files("certifi", excludes=NATIVE)
# keyring's entry-point metadata, per the note above.
datas += copy_metadata("keyring")
# RapidOCR ships its detection/recognition ONNX models and its YAML config as
# package data. Without them OCR raises on first use -- and because OCR only
# runs on scanned PDFs, that would not surface until a user uploaded one.
datas += collect_data_files("rapidocr_onnxruntime", excludes=NATIVE)

binaries = []
# Every native library the app loads, through the one pipeline that actually
# processes them. onnxruntime is deliberately absent: PyInstaller ships an
# official hook for it, and hand-collecting it is what broke the build.
binaries += collect_dynamic_libs("faiss")
binaries += collect_dynamic_libs("tokenizers")
# Includes libiomp5md.dll, CTranslate2's OpenMP runtime. Two copies of an
# OpenMP runtime in one process is its own flavour of initialisation failure,
# which is the other reason this must not also be swept up as data.
binaries += collect_dynamic_libs("ctranslate2")
# pypdfium2 links the PDF renderer as a separate native library.
binaries += collect_dynamic_libs("pypdfium2_raw")

excluded = [
    # Nothing here should ever pull in a deep-learning stack. If torch appears
    # in the bundle, an embedding or STT backend has regressed to a torch path
    # and the installer is about to grow by well over a gigabyte.
    "torch",
    "torchvision",
    "tensorflow",
    "matplotlib",
    "tkinter",
    "PyQt5",
    "PySide6",
    "pytest",
    # Removed with the second provider. Excluded rather than merely unimported
    # so a stale install in the build environment cannot quietly add ~40MB of
    # Google API client back into the bundle.
    "google.genai",
    "google_genai",
]

a = Analysis(
    ["../app/__main__.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excluded,
    noarchive=False,
)

# ---------------------------------------------------------------------------
# Drop the bundled Microsoft C++ standard library.
#
# PyInstaller collects whatever msvcp140.dll it finds next to the interpreter,
# and Windows resolves DLLs from the application directory *before* System32 --
# so that copy shadows the system one for every native module in the process.
# When it is older than the one a native extension was built against, the
# extension loads and then fails in its initialiser:
#
#   DLL load failed while importing onnxruntime_pybind11_state:
#   A dynamic link library (DLL) initialization routine failed.
#
# Measured here: bundled msvcp140.dll 567,328 bytes vs system 553,552 bytes,
# and onnxruntime could not initialise against the bundled one. That took out
# every onnxruntime consumer at once -- document embeddings, scanned-PDF OCR,
# and the Silero VAD -- which is why the shipped app failed to ingest any
# document at all.
#
# Verified by bisection: removing only msvcp140*.dll fixes it; vcruntime140*
# is fine and stays, because Python itself is built against the copy it ships.
#
# Safe because the app already requires the VC++ 2015-2022 redistributable to
# run at all (CPython, WebView2 and CTranslate2 all link it), and Windows 10/11
# ship with it. If a target machine somehow lacks it, the fix is to install the
# redistributable rather than to ship a mismatched copy of half of it.
_SHADOWING_RUNTIME = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll")

a.binaries = [
    entry
    for entry in a.binaries
    if os.path.basename(entry[0]).lower() not in _SHADOWING_RUNTIME
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="call-assistant-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX-packed binaries trip antivirus far more often than they save space
    console=True,  # the shell reads the readiness line from stdout
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="call-assistant-backend",
)
