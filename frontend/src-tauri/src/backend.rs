//! Lifecycle for the Python backend sidecar.
//!
//! The shell owns the backend process: it picks the port, mints the token,
//! spawns it, waits for it to actually be listening, and makes sure it dies.
//!
//! Shutdown is three layers because no single one covers every case:
//!   1. POST /shutdown        — the normal path, lets the backend close cleanly
//!   2. kill() after a grace  — for a backend that is wedged
//!   3. Win32 Job Object      — for when *we* are killed and never get to ask
//!
//! Without (3), force-quitting the app on Windows leaves an orphaned
//! python.exe holding the microphone and the SQLite database, because Windows
//! does not kill child processes with their parent.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rand::Rng;
use serde::Deserialize;
use tauri::{AppHandle, Manager};

/// How long to wait for the readiness line. Generous: the first launch loads an
/// ONNX model and may be fighting an antivirus scan of a freshly extracted
/// PyInstaller directory.
const READY_TIMEOUT: Duration = Duration::from_secs(90);
/// Grace period between asking the backend to stop and killing it.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct BackendHandle {
    pub port: u16,
    pub token: String,
    /// PID the backend reports for *itself*.
    ///
    /// Not always the same as the spawned child's PID: on Windows a virtualenv
    /// `python.exe` is a launcher stub that runs the real interpreter as a
    /// grandchild, so killing the child handle would leave the actual server
    /// running. Measured on this repo's venv: child 55980, real backend 53976.
    /// Production spawns the PyInstaller exe directly and the two match, but
    /// dev mode needs this to shut down reliably.
    pub reported_pid: Option<u32>,
}

pub struct BackendProcess {
    child: Option<Child>,
    pub handle: BackendHandle,
}

pub struct BackendState(pub Mutex<Option<BackendProcess>>);

#[derive(Deserialize)]
struct ReadyLine {
    ready: bool,
    #[allow(dead_code)]
    port: u16,
    #[serde(default)]
    pid: Option<u32>,
}

/// Ask the OS for a free port, then release it.
///
/// Racy in principle — another process could take it in the gap — but the
/// alternative is a hardcoded port that collides with whatever else the user
/// runs, which fails far more often in practice.
fn free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

fn mint_token() -> String {
    let mut rng = rand::thread_rng();
    (0..32)
        .map(|_| {
            const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
            CHARS[rng.gen_range(0..CHARS.len())] as char
        })
        .collect()
}

/// Where the backend executable lives.
///
/// Packaged: a PyInstaller onedir tree shipped as a Tauri resource.
/// Dev: `python -m app` from the repo, so there is no build step in the loop.
fn resolve_command(app: &AppHandle) -> Result<(String, Vec<String>), String> {
    if cfg!(debug_assertions) {
        let python = if cfg!(windows) { "python" } else { "python3" };
        return Ok((python.to_string(), vec!["-m".into(), "app".into()]));
    }

    let resource_dir = app
        .path()
        .resource_dir()
        .map_err(|e| format!("cannot resolve resource dir: {e}"))?;
    let exe: PathBuf = resource_dir
        .join("backend")
        .join(if cfg!(windows) {
            "interview-coach-backend.exe"
        } else {
            "interview-coach-backend"
        });

    if !exe.exists() {
        return Err(format!("backend executable missing at {}", exe.display()));
    }
    Ok((exe.to_string_lossy().into_owned(), vec![]))
}

/// Working directory for the dev backend: the repo's `backend/` folder.
fn dev_working_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest.parent()?.parent()?.join("backend");
    candidate.exists().then_some(candidate)
}

/// Per-user data root. Keeps documents, the vector index, session history and
/// downloaded models out of Program Files, which is read-only for a per-user
/// install and would break ingestion.
fn data_dir(app: &AppHandle) -> Option<String> {
    app.path()
        .app_local_data_dir()
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

pub fn spawn(app: &AppHandle) -> Result<BackendProcess, String> {
    let port = free_port().map_err(|e| format!("no free port: {e}"))?;
    let token = mint_token();
    let (program, base_args) = resolve_command(app)?;

    let mut command = Command::new(&program);
    command
        .args(&base_args)
        .arg("--port")
        .arg(port.to_string())
        .arg("--token")
        .arg(&token)
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(dir) = data_dir(app) {
        command.arg("--data-dir").arg(dir);
    }
    if cfg!(debug_assertions) {
        if let Some(dir) = dev_working_dir() {
            command.current_dir(dir);
        }
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // CREATE_NO_WINDOW: without it the packaged app flashes a console.
        command.creation_flags(0x0800_0000);
    }

    let mut child = command
        .spawn()
        .map_err(|e| format!("failed to start backend ({program}): {e}"))?;

    #[cfg(windows)]
    job_object::assign_current_process_job(&child);

    let reported_pid = wait_until_ready(&mut child)?;

    Ok(BackendProcess {
        child: Some(child),
        handle: BackendHandle {
            port,
            token,
            reported_pid,
        },
    })
}

/// Block until the backend prints its readiness line.
///
/// Reading stdout rather than sleeping or polling: startup time depends on
/// model loading and antivirus, so any fixed delay is either wasteful or wrong.
fn wait_until_ready(child: &mut Child) -> Result<Option<u32>, String> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "backend stdout unavailable".to_string())?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<Option<u32>, String>>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if let Ok(parsed) = serde_json::from_str::<ReadyLine>(&line) {
                if parsed.ready {
                    let _ = tx.send(Ok(parsed.pid));
                    // Keep draining: a full stdout pipe would block the backend.
                    continue;
                }
            }
            log::debug!("backend: {line}");
        }
        let _ = tx.send(Err("backend exited before signalling readiness".into()));
    });

    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            return Err(format!("backend exited during startup with {status}"));
        }
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Ok(pid)) => return Ok(pid),
            Ok(Err(message)) => return Err(message),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err("backend stdout closed".into())
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    return Err("backend did not become ready in time".into());
                }
            }
        }
    }
}

impl BackendProcess {
    /// Ask nicely, then insist.
    pub fn shutdown(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };

        let url = format!("http://127.0.0.1:{}/shutdown", self.handle.port);
        let token = self.handle.token.clone();
        let _ = std::thread::spawn(move || {
            let _ = ureq::post(&url)
                .set("Authorization", &format!("Bearer {token}"))
                .timeout(Duration::from_secs(2))
                .call();
        })
        .join();

        let deadline = Instant::now() + SHUTDOWN_GRACE;
        loop {
            match child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(100));
                }
                _ => break,
            }
        }

        log::warn!("backend did not stop gracefully; killing it");
        let _ = child.kill();
        let _ = child.wait();

        // Killing the child handle is not always enough (see reported_pid): if
        // the spawned process was a launcher stub, the real server is a
        // grandchild and is still running.
        if let Some(pid) = self.handle.reported_pid {
            kill_by_pid(pid);
        }
    }
}

/// Last-resort kill of the process the backend reported as itself.
fn kill_by_pid(pid: u32) {
    if pid == 0 {
        return;
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .creation_flags(0x0800_0000) // CREATE_NO_WINDOW
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(windows))]
    {
        let _ = Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
    }
}

#[cfg(windows)]
mod job_object {
    //! Kill the backend if this process dies without getting to ask.
    //!
    //! Windows has no parent-kills-child semantics, so a crash or a Task
    //! Manager kill would otherwise leak the backend. A Job Object with
    //! KILL_ON_JOB_CLOSE makes the OS clean up for us.

    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use std::sync::OnceLock;

    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    static JOB: OnceLock<usize> = OnceLock::new();

    fn job_handle() -> HANDLE {
        *JOB.get_or_init(|| unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle != 0 {
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const core::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
            }
            handle as usize
        }) as HANDLE
    }

    pub fn assign_current_process_job(child: &Child) {
        let job = job_handle();
        if job == 0 {
            log::warn!("could not create job object; backend may outlive a crash");
            return;
        }
        unsafe {
            if AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) == 0 {
                log::warn!("could not assign backend to job object");
            }
        }
    }
}
