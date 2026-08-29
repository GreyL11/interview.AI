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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::Rng;
use serde::Deserialize;
use tauri::{AppHandle, Manager};

/// How long to wait for the readiness line. Generous: the first launch loads an
/// ONNX model and may be fighting an antivirus scan of a freshly extracted
/// PyInstaller directory.
const READY_TIMEOUT: Duration = Duration::from_secs(90);
/// How long /health may take to answer once the socket is up. Short: by this
/// point the process has already loaded its models and bound its port.
const HEALTH_TIMEOUT: Duration = Duration::from_secs(20);
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
    /// Where the backend is writing its rotating log, as the backend itself
    /// reports it. Read from the readiness line rather than re-derived here,
    /// so the two can never disagree about which directory to open.
    pub logs_dir: Option<String>,
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
    #[serde(default)]
    logs_dir: Option<String>,
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
            "call-assistant-backend.exe"
        } else {
            "call-assistant-backend"
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

    progress(app, "launching_engine", "Starting the local engine");

    let mut child = command
        .spawn()
        .map_err(|e| format!("failed to start backend ({program}): {e}"))?;

    #[cfg(windows)]
    job_object::assign_current_process_job(&child);

    // stderr MUST be drained. It is a piped handle with a fixed OS buffer, so a
    // backend that logs enough to fill it blocks on write and never reaches
    // readiness -- a hang indistinguishable from a slow model load. Draining
    // into a bounded tail also gives the failure screen something concrete to
    // show instead of "did not become ready in time".
    let stderr_tail = drain_stderr(&mut child);

    // The stages below are what the user reads while they wait. They are named
    // after what is actually happening, and each one only appears once the
    // previous step genuinely completed -- a stage that lies about progress is
    // worse than no stage at all.
    progress(app, "waiting_for_process", "Waiting for the local engine");
    let ready = wait_until_ready(&mut child, &stderr_tail)?;

    // The readiness line is printed once uvicorn's socket accepts. That is not
    // the same as the API being usable, so it is never treated as READY.
    progress(app, "checking_health", "Checking the local engine");
    confirm_health(port, &token, &mut child, &stderr_tail)?;

    Ok(BackendProcess {
        child: Some(child),
        handle: BackendHandle {
            port,
            token,
            reported_pid: ready.pid,
            logs_dir: ready.logs_dir,
        },
    })
}

/// Emit a startup stage to the UI. Named stages, never a fabricated
/// percentage: the shell genuinely cannot know how long model loading takes.
fn progress(app: &AppHandle, stage: &str, label: &str) {
    use tauri::Emitter;
    let _ = app.emit(
        "backend://startup",
        serde_json::json!({ "stage": stage, "label": label }),
    );
}

/// Last few stderr lines, kept for the failure screen.
type StderrTail = Arc<Mutex<Vec<String>>>;
const STDERR_TAIL_LINES: usize = 40;

fn drain_stderr(child: &mut Child) -> StderrTail {
    let tail: StderrTail = Arc::new(Mutex::new(Vec::new()));
    let Some(stderr) = child.stderr.take() else {
        return tail;
    };
    let sink = Arc::clone(&tail);
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            log::warn!("backend stderr: {line}");
            if let Ok(mut buffer) = sink.lock() {
                if buffer.len() == STDERR_TAIL_LINES {
                    buffer.remove(0);
                }
                buffer.push(line);
            }
        }
    });
    tail
}

fn tail_text(tail: &StderrTail) -> String {
    tail.lock()
        .map(|buffer| buffer.join("\n"))
        .unwrap_or_default()
}

fn with_tail(message: String, tail: &StderrTail) -> String {
    let text = tail_text(tail);
    if text.is_empty() {
        message
    } else {
        format!("{message}\n\n{text}")
    }
}

/// The origin the packaged WebView serves the frontend from on Windows.
///
/// Every request the UI makes is cross-origin to `127.0.0.1:<port>` and is
/// therefore subject to CORS. Sending this header during the readiness check is
/// the entire point of `confirm_health`: without it the check is a `curl`, and a
/// `curl` cannot detect the failure this app actually shipped with.
const WEBVIEW_ORIGIN: &str = "http://tauri.localhost";

/// Prove the backend is usable *by the frontend*, not merely listening.
///
/// A running process, an open port and a 200 from `curl` were all true while
/// the UI showed "Cannot reach the backend". Readiness here means all three of
/// the things the UI depends on, in the same request it will make:
///
///   1. the token is accepted            (Authorization header)
///   2. the origin is allowed            (Origin header -> CORS response header)
///   3. the body is what the client expects
///
/// Anything less lets the shell declare READY for a backend the UI cannot
/// actually talk to, which is the failure this whole path exists to prevent.
fn confirm_health(
    port: u16,
    token: &str,
    child: &mut Child,
    tail: &StderrTail,
) -> Result<(), String> {
    let url = format!("http://127.0.0.1:{port}/health");
    let deadline = Instant::now() + HEALTH_TIMEOUT;
    // Kept so a timeout can explain *how* the last attempt failed rather than
    // just that time ran out.
    let mut last_reason = String::from("no response yet");

    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(with_tail(
                format!("backend exited during startup with {status}"),
                tail,
            ));
        }

        match probe_health(&url, token) {
            Ok(()) => return Ok(()),
            Err(reason) => last_reason = reason,
        }

        if Instant::now() > deadline {
            return Err(with_tail(
                format!("the local engine started but is not answering: {last_reason}"),
                tail,
            ));
        }
        std::thread::sleep(Duration::from_millis(150));
    }
}

/// One readiness attempt. `Err` carries a reason specific enough to act on.
fn probe_health(url: &str, token: &str) -> Result<(), String> {
    let response = ureq::get(url)
        .set("Authorization", &format!("Bearer {token}"))
        .set("Origin", WEBVIEW_ORIGIN)
        .timeout(Duration::from_secs(2))
        .call();

    let response = match response {
        Ok(response) => response,
        // ureq treats any 4xx/5xx as an error. Distinguish them: a 401 is a
        // token mismatch we can name, while a transport error is just "not up
        // yet" and will be retried.
        Err(ureq::Error::Status(401, _)) => {
            return Err("it rejected the shell's access token".into())
        }
        Err(ureq::Error::Status(code, _)) => {
            return Err(format!("it returned HTTP {code} from /health"))
        }
        // A wildcard rather than naming `Transport`: this arm exists to keep the
        // match total whatever the http crate adds to its error enum, since a
        // readiness probe has nothing useful to do differently per transport
        // failure -- it retries either way.
        Err(other) => return Err(other.to_string()),
    };

    // The browser discards a response without this header, so a backend that
    // omits it is unusable by the UI no matter what status it returned.
    if response.header("access-control-allow-origin").is_none() {
        return Err(
            "it answered but refused the app's own origin, so the interface \
             cannot read the reply"
                .into(),
        );
    }

    // Read the body rather than trusting the status: a proxy or a stale process
    // on a recycled port can return 200 for something that is not this backend.
    let body = response
        .into_string()
        .map_err(|e| format!("its reply could not be read: {e}"))?;
    if !body.contains("healthy") {
        return Err("something other than the local engine answered on its port".into());
    }
    Ok(())
}

/// Block until the backend prints its readiness line.
///
/// Reading stdout rather than sleeping or polling: startup time depends on
/// model loading and antivirus, so any fixed delay is either wasteful or wrong.
fn wait_until_ready(child: &mut Child, tail: &StderrTail) -> Result<Ready, String> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "backend stdout unavailable".to_string())?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<Ready, String>>();
    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines() {
            let Ok(line) = line else { break };
            if let Ok(parsed) = serde_json::from_str::<ReadyLine>(&line) {
                if parsed.ready {
                    let _ = tx.send(Ok(Ready {
                        pid: parsed.pid,
                        logs_dir: parsed.logs_dir,
                    }));
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
            return Err(with_tail(
                format!("backend exited during startup with {status}"),
                tail,
            ));
        }
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(Ok(ready)) => return Ok(ready),
            Ok(Err(message)) => return Err(with_tail(message, tail)),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(with_tail("backend stdout closed".into(), tail))
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if Instant::now() > deadline {
                    let _ = child.kill();
                    return Err(with_tail(
                        "backend did not become ready in time".into(),
                        tail,
                    ));
                }
            }
        }
    }
}

/// What the readiness line told us about the running backend.
struct Ready {
    pid: Option<u32>,
    logs_dir: Option<String>,
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
    #[cfg(windows)]
    use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject,
    CreateJobObjectW,
    SetInformationJobObject,
    JobObjectExtendedLimitInformation,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };



    static JOB: OnceLock<usize> = OnceLock::new();

    fn job_handle() -> HANDLE {
        *JOB.get_or_init(|| unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if !handle.is_null() {
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
        if job.is_null() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener as StdTcpListener;

    #[test]
    fn free_port_returns_a_port_that_can_actually_be_bound() {
        let port = free_port().expect("should find a free port");
        assert!(port > 0);
        // The port is released before being handed back, so the backend must be
        // able to take it. If this fails, the release is not happening.
        StdTcpListener::bind(("127.0.0.1", port)).expect("port should be bindable");
    }

    #[test]
    fn free_port_does_not_hand_out_the_same_port_twice_in_a_row() {
        let (a, b) = (free_port().unwrap(), free_port().unwrap());
        // Not a guarantee the OS makes, but a regression here would mean the
        // listener is being leaked rather than dropped.
        assert!(a != b || StdTcpListener::bind(("127.0.0.1", a)).is_ok());
    }

    #[test]
    fn a_minted_token_is_long_and_url_safe() {
        let token = mint_token();
        assert_eq!(token.len(), 32);
        assert!(token.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn minted_tokens_differ_between_launches() {
        assert_ne!(mint_token(), mint_token());
    }

    #[test]
    fn the_readiness_line_is_parsed() {
        let line = r#"{"ready":true,"port":8123,"pid":4242,"logs_dir":"C:\logs"}"#;
        let parsed: ReadyLine = serde_json::from_str(line).expect("should parse");
        assert!(parsed.ready);
        assert_eq!(parsed.pid, Some(4242));
        assert_eq!(parsed.logs_dir.as_deref(), Some("C:\logs"));
    }

    #[test]
    fn a_readiness_line_without_the_optional_fields_still_parses() {
        // Older backends, and any future one that drops a field, must not make
        // the shell hang waiting for a line it already received.
        let parsed: ReadyLine =
            serde_json::from_str(r#"{"ready":true,"port":1}"#).expect("should parse");
        assert!(parsed.ready);
        assert_eq!(parsed.pid, None);
    }

    #[test]
    fn ordinary_log_output_is_not_mistaken_for_readiness() {
        assert!(serde_json::from_str::<ReadyLine>("INFO starting up").is_err());
        assert!(serde_json::from_str::<ReadyLine>("{}").is_err());
    }

    #[test]
    fn a_probe_against_a_closed_port_fails_rather_than_hanging() {
        // Bind and drop, so the port is almost certainly closed.
        let port = free_port().unwrap();
        let error = probe_health(&format!("http://127.0.0.1:{port}/health"), "t")
            .expect_err("a closed port must not report healthy");
        assert!(!error.is_empty());
    }

    #[test]
    fn the_webview_origin_is_the_one_the_backend_allows() {
        // Kept in lockstep with the backend's CORS regex and its test suite. If
        // this constant drifts, every request from the packaged UI is rejected
        // while curl keeps working -- the exact failure this app shipped with.
        assert_eq!(WEBVIEW_ORIGIN, "http://tauri.localhost");
    }
}
