// Release builds must not open a console window behind the app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use std::sync::Mutex;

use serde::Serialize;
use tauri::{Manager, RunEvent, WindowEvent};

use backend::{BackendState, spawn};

#[derive(Serialize, Clone)]
struct BackendInfo {
    port: u16,
    token: String,
}

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

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(BackendState(Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![backend_info, backend_running])
        .setup(|app| {
            let handle = app.handle().clone();
            match spawn(&handle) {
                Ok(process) => {
                    let info = BackendInfo {
                        port: process.handle.port,
                        token: process.handle.token.clone(),
                    };
                    // Inject before the page scripts run, so the very first
                    // fetch already knows the port and token.
                    if let Some(window) = app.get_webview_window("main") {
                        let script = format!(
                            "window.__BACKEND__ = {{ port: {}, token: {:?} }};",
                            info.port, info.token
                        );
                        let _ = window.eval(&script);
                    }
                    let state = app.state::<BackendState>();
                    *state.0.lock().unwrap() = Some(process);
                }
                Err(message) => {
                    // Surface the reason instead of showing an app that
                    // silently cannot do anything.
                    log::error!("backend failed to start: {message}");
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.eval(&format!(
                            "document.body.innerHTML = {:?};",
                            format!(
                                "<pre style='padding:24px;font:14px sans-serif;color:#f85149'>\
                                 Interview Coach could not start its local backend.\n\n{message}</pre>"
                            )
                        ));
                    }
                }
            }
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
            | RunEvent::ExitRequested { .. } => {
                if let Some(state) = app_handle.try_state::<BackendState>() {
                    if let Ok(mut guard) = state.0.lock() {
                        if let Some(process) = guard.as_mut() {
                            process.shutdown();
                        }
                        *guard = None;
                    }
                }
            }
            _ => {}
        });
}
