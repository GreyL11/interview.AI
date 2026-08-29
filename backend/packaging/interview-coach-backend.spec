# PyInstaller spec for the backend sidecar.
#
#   cd backend && pyinstaller packaging/interview-coach-backend.spec
#
# onedir, not onefile: onefile re-extracts the whole tree to %TEMP% on every
# launch, which is slow and is exactly the behaviour antivirus heuristics flag.
# The resulting folder is shipped as a Tauri resource.
#
# Models are deliberately NOT bundled. They are downloaded on first use into the
# per-user data directory, which keeps the installer at a few hundred MB instead
# of well over a gigabyte, and lets the user swap models without a reinstall.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
hidden += collect_submodules("faster_whisper")
# keyring finds its backend through entry points, which PyInstaller does not
# follow. Without these the frozen app falls back to keyring's fail backend
# and reports "no credential store", so saved API keys would not persist.
hidden += collect_submodules("keyring")
hidden += ["keyring.backends.Windows", "win32ctypes.core"]

datas = []
# The Silero VAD model ships inside faster-whisper as an ONNX asset; without
# this the packaged app has no voice activity detection.
datas += collect_data_files("faster_whisper", includes=["assets/*"])
# CTranslate2 and onnxruntime carry native libraries that must come along.
datas += collect_data_files("ctranslate2", include_py_files=False)
datas += collect_data_files("onnxruntime", include_py_files=False)

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
]

a = Analysis(
    ["../app/__main__.py"],
    pathex=[".."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excluded,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="interview-coach-backend",
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
    name="interview-coach-backend",
)
