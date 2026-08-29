// Release builds must not open a console window behind the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

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
            label: "Opening Interview Coach".into(),
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
        label: "Opening Interview Coach".into(),
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
            // Injected before the page's own fetches run, so the very first
            // request already knows the port and token.
            if let Some(window) = app.get_webview_window("main") {
                let script = format!(
                    "window.__BACKEND__ = {{ port: {}, token: {:?} }};",
                    info.port, info.token
                );
                let _ = window.eval(&script);
            }
            if let Some(slot) = app.try_state::<BackendState>() {
                if let Ok(mut guard) = slot.0.lock() {
                    *guard = Some(process);
                }
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
                label: "Something prevented Interview Coach from starting".into(),
                detail: Some(message),
                logs_dir: None,
            });
        }
    }
}

/// Retry after a failed start. Tears down any half-started process first so a
/// second attempt cannot leave the first one orphaned.
#[tauri::command]
fn retry_backend(app: tauri::AppHandle) {
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
    std::thread::spawn(move || start_backend(&handle));
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
            // Off the setup thread so the window paints the startup screen
            // immediately instead of appearing frozen for the whole boot.
            std::thread::spawn(move || start_backend(&handle));
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
