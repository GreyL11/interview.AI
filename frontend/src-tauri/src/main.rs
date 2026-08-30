// Release builds must not open a console window behind the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Emitter, Manager, RunEvent, WindowEvent};

use backend::{spawn, BackendState};

#[derive(Serialize, Clone)]
struct BackendInfo {
    port: u16,
    token: String,
}

/// What the startup screen renders. The shell owns backend lifecycle, so it is
/// the only thing that can honestly say whether the engine is up.
#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct StartupState {
    /// "starting" | "ready" | "failed"
    status: String,
    stage: String,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logs_dir: Option<String>,
}

impl StartupState {
    fn starting() -> Self {
        Self {
            status: "starting".into(),
            stage: "launching".into(),
            label: "Opening Call Assistant".into(),
            detail: None,
            logs_dir: None,
        }
    }
}

struct StartupStatus(Mutex<StartupState>);

/// The UI asks for this on load to learn where the backend is and how to
/// authenticate. Injecting it via a command rather than baking it into the
/// bundle keeps the token out of anything written to disk.
#[tauri::command]
fn backend_info(state: tauri::State<'_, BackendState>) -> Result<BackendInfo, String> {
    let guard = state.0.lock().map_err(|e| e.to_string())?;
    guard
        .as_ref()
        .map(|process| BackendInfo {
            port: process.handle.port,
            token: process.handle.token.clone(),
        })
        .ok_or_else(|| "backend is not running".to_string())
}

#[tauri::command]
fn backend_running(state: tauri::State<'_, BackendState>) -> bool {
    state.0.lock().map(|g| g.is_some()).unwrap_or(false)
}

/// Current startup state, for a UI that mounted after the events were emitted.
/// Without this, a slow first paint would miss them and sit on "Opening…".
#[tauri::command]
fn startup_status(state: tauri::State<'_, StartupStatus>) -> StartupState {
    state
        .0
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| StartupState::starting())
}

/// Reveal the backend's log directory in Explorer. The path comes from the
/// backend's own readiness line, so the two cannot disagree.
#[tauri::command]
fn open_logs_folder(app: tauri::AppHandle, state: tauri::State<'_, BackendState>) -> Result<(), String> {
    let from_backend = state
        .0
        .lock()
        .ok()
        .and_then(|g| g.as_ref().and_then(|p| p.handle.logs_dir.clone()));

    // Falls back to the per-user data dir: if the backend never started, its
    // log directory is exactly what the user needs to look at.
    let path = match from_backend {
        Some(dir) => std::path::PathBuf::from(dir),
        None => app
            .path()
            .app_local_data_dir()
            .map_err(|e| e.to_string())?
            .join("logs"),
    };
    let _ = std::fs::create_dir_all(&path);

    tauri_plugin_opener::open_path(path.to_string_lossy().to_string(), None::<&str>)
        .map_err(|e| e.to_string())
}

/// Reveal the per-user data directory: documents, session history, the vector
/// index and downloaded models all live here.
#[tauri::command]
fn open_data_folder(app: tauri::AppHandle) -> Result<(), String> {
    let path = app
        .path()
        .app_local_data_dir()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::create_dir_all(&path);
    tauri_plugin_opener::open_path(path.to_string_lossy().to_string(), None::<&str>)
        .map_err(|e| e.to_string())
}

/// Excludes the window from anything capturing the screen — Meet/Zoom/Teams
/// screen-share, OBS, etc. The window keeps rendering normally on the rep's
/// own display; only the captured frame omits it, with no placeholder box.
/// Windows-only: there is no equivalent OS API on macOS/Linux.
#[cfg(windows)]
fn hide_from_screen_capture(window: &tauri::WebviewWindow) {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::UI::WindowsAndMessaging::{SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE};
    match window.hwnd() {
        // tauri's HWND (from the `windows` crate) and windows-sys's HWND are both
        // just *mut c_void — grab the raw pointer via `.0`, skip a second crate.
        Ok(hwnd) => {
            if unsafe { SetWindowDisplayAffinity(hwnd.0, WDA_EXCLUDEFROMCAPTURE) } == 0 {
                // The error code is the whole diagnosis: 87 (ERROR_INVALID_PARAMETER)
                // means the OS refused WDA_EXCLUDEFROMCAPTURE — it needs Windows 10
                // 2004 (build 19041) or later. Without it there is nothing to go on.
                log::warn!(
                    "screen-capture exclusion rejected by the OS (GetLastError={})",
                    unsafe { GetLastError() },
                );
            } else {
                log::info!("window excluded from screen capture");
            }
        }
        Err(e) => log::warn!("could not get window handle to exclude from capture: {e}"),
    }
}

#[cfg(not(windows))]
fn hide_from_screen_capture(_window: &tauri::WebviewWindow) {}

/// Start (or restart) the backend and publish the outcome.
fn start_backend(app: &tauri::AppHandle) {
    let publish = |state: StartupState| {
        if let Some(slot) = app.try_state::<StartupStatus>() {
            if let Ok(mut guard) = slot.0.lock() {
                *guard = state.clone();
            }
        }
        let _ = app.emit("backend://status", state);
    };

    publish(StartupState {
        status: "starting".into(),
        stage: "launching".into(),
        label: "Opening Call Assistant".into(),
        detail: None,
        logs_dir: None,
    });

    match spawn(app) {
        Ok(process) => {
            let info = BackendInfo {
                port: process.handle.port,
                token: process.handle.token.clone(),
            };
            let logs_dir = process.handle.logs_dir.clone();

            // Order matters. The state has to be published *before* READY is
            // emitted, because `backend_info` reads it and the UI calls that
            // the instant it sees READY. Publishing after would leave a window
            // where the UI is told to go and has nowhere to go to.
            if let Some(slot) = app.try_state::<BackendState>() {
                if let Ok(mut guard) = slot.0.lock() {
                    *guard = Some(process);
                }
            }

            publish(StartupState {
                status: "starting".into(),
                stage: "configuring_runtime".into(),
                label: "Preparing the interface".into(),
                detail: None,
                logs_dir: logs_dir.clone(),
            });

            // Convenience, not the contract. `eval` reaches only the page that
            // is loaded right now, so it does not survive a webview reload and
            // races a page that has not finished loading. The frontend's
            // `ensureBackendRuntime()` asks `backend_info` for the same values
            // and is what actually guarantees they are known before the first
            // request.
            if let Some(window) = app.get_webview_window("main") {
                let script = format!(
                    "window.__BACKEND__ = {{ port: {}, token: {} }};",
                    info.port,
                    // serde_json, not {:?}: Rust's debug escaping is not
                    // JavaScript's, and the token is injected as source.
                    serde_json::to_string(&info.token).unwrap_or_else(|_| "\"\"".into()),
                );
                let _ = window.eval(&script);
            }

            publish(StartupState {
                status: "ready".into(),
                stage: "ready".into(),
                label: "Ready".into(),
                detail: None,
                logs_dir,
            });
        }
        Err(message) => {
            log::error!("backend failed to start: {message}");
            publish(StartupState {
                status: "failed".into(),
                stage: "failed".into(),
                label: "Something prevented Call Assistant from starting".into(),
                detail: Some(message),
                logs_dir: None,
            });
        }
    }
}

/// True while a start attempt is in flight.
///
/// Startup takes up to 90 seconds, and the Retry button is on screen for all of
/// it. Without this, an impatient second click spawns a second backend that
/// fights the first over the SQLite database and the audio device -- and the
/// losing one is never shut down, because only the last handle is stored.
static STARTING: AtomicBool = AtomicBool::new(false);

/// Retry after a failed start. Tears down any half-started process first so a
/// second attempt cannot leave the first one orphaned.
#[tauri::command]
fn retry_backend(app: tauri::AppHandle) {
    // compare_exchange, not load-then-store: two clicks can land on two threads
    // and both would see `false`.
    if STARTING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        log::info!("retry ignored: a start attempt is already running");
        return;
    }

    if let Some(slot) = app.try_state::<BackendState>() {
        if let Ok(mut guard) = slot.0.lock() {
            if let Some(process) = guard.as_mut() {
                process.shutdown();
            }
            *guard = None;
        }
    }
    let handle = app.clone();
    // Off the UI thread: spawn() blocks until the backend is ready, and
    // blocking here would freeze the window it is supposed to be updating.
    std::thread::spawn(move || {
        start_backend(&handle);
        STARTING.store(false, Ordering::SeqCst);
    });
}

fn shutdown_backend(app: &tauri::AppHandle) {
    if let Some(state) = app.try_state::<BackendState>() {
        if let Ok(mut guard) = state.0.lock() {
            if let Some(process) = guard.as_mut() {
                process.shutdown();
            }
            *guard = None;
        }
    }
}

fn main() {
    tauri::Builder::default()
        // Two copies would both spawn a backend and fight over the SQLite
        // database and the audio device. Focus the existing window instead.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        // Installs an actual logger behind the `log` facade. Without one every
        // log::warn!/error! in this file is a silent no-op, which is how a
        // failed capture-exclusion and a failed backend start both vanished.
        // Writes to the same folder the "Open logs folder" button reveals.
        .plugin(tauri_plugin_log::Builder::new().build())
        .manage(BackendState(Mutex::new(None)))
        .manage(StartupStatus(Mutex::new(StartupState::starting())))
        .invoke_handler(tauri::generate_handler![
            backend_info,
            backend_running,
            startup_status,
            retry_backend,
            open_logs_folder,
            open_data_folder,
        ])
        .setup(|app| {
            let handle = app.handle().clone();

            if let Some(window) = app.get_webview_window("main") {
                hide_from_screen_capture(&window);
            }
            // Claimed here, not inside start_backend, so a Retry click during
            // the first boot is refused rather than racing it.
            STARTING.store(true, Ordering::SeqCst);
            // Off the setup thread so the window paints the startup screen
            // immediately instead of appearing frozen for the whole boot.
            std::thread::spawn(move || {
                start_backend(&handle);
                STARTING.store(false, Ordering::SeqCst);
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the application")
        .run(|app_handle, event| match event {
            // Closing the window and exiting are separate events; the backend
            // must be stopped on whichever happens.
            RunEvent::WindowEvent {
                event: WindowEvent::CloseRequested { .. },
                ..
            }
            | RunEvent::ExitRequested { .. } => shutdown_backend(app_handle),
            _ => {}
        });
}
