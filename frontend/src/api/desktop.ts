/**
 * Bridge to the Tauri shell.
 *
 * The shell owns the backend process, so it is the only thing that can
 * honestly report whether the engine is up. In a plain browser (dev, `npm run
 * dev`) none of this exists, and the app falls back to assuming a
 * hand-started backend on the dev port -- which is exactly what runtime.ts
 * already does.
 *
 * Everything here is dynamically imported and guarded: importing the Tauri API
 * eagerly would put it in the browser bundle and throw on load outside Tauri.
 */

export type StartupStatus = "starting" | "ready" | "failed";

export interface StartupState {
  status: StartupStatus;
  /** Machine-readable stage, for tests and telemetry. */
  stage: string;
  /** Human sentence for the startup screen. Written by the shell. */
  label: string;
  /** Concise technical reason, only on failure. */
  detail?: string;
  logsDir?: string;
}

declare global {
  interface Window {
    __TAURI_INTERNALS__?: unknown;
  }
}

/** True when running inside the packaged desktop shell. */
export function isDesktop(): boolean {
  return typeof window !== "undefined" && window.__TAURI_INTERNALS__ !== undefined;
}

async function core() {
  return await import("@tauri-apps/api/core");
}

async function event() {
  return await import("@tauri-apps/api/event");
}

/**
 * Current startup state.
 *
 * Polled once on mount rather than relying only on the event: the shell starts
 * the backend before the webview finishes its first paint, so a listener alone
 * would miss the stages that already fired.
 */
export async function getStartupState(): Promise<StartupState> {
  if (!isDesktop()) {
    // Browser dev: the backend is started by hand, so there is nothing to wait
    // for. App.tsx's own /health check decides whether it is actually up.
    return { status: "ready", stage: "ready", label: "Ready" };
  }
  try {
    const { invoke } = await core();
    return await invoke<StartupState>("startup_status");
  } catch (cause) {
    return {
      status: "failed",
      stage: "bridge",
      label: "Something prevented Interview Coach from starting",
      detail: String(cause),
    };
  }
}

/** Subscribe to startup transitions. Returns an unsubscribe function. */
export async function onStartupChange(
  handler: (state: StartupState) => void,
): Promise<() => void> {
  if (!isDesktop()) return () => {};
  try {
    const { listen } = await event();
    const stops = await Promise.all([
      listen<StartupState>("backend://status", (e) => handler(e.payload)),
      // Intermediate progress carries no status field; it only refines the
      // label while the state stays "starting".
      listen<{ stage: string; label: string }>("backend://startup", (e) =>
        handler({ status: "starting", stage: e.payload.stage, label: e.payload.label }),
      ),
    ]);
    return () => stops.forEach((stop) => stop());
  } catch {
    return () => {};
  }
}

/** Ask the shell to start the backend again after a failure. */
export async function retryBackend(): Promise<void> {
  if (!isDesktop()) {
    window.location.reload();
    return;
  }
  const { invoke } = await core();
  await invoke("retry_backend");
}

/**
 * Make sure the page knows where the backend is.
 *
 * The shell injects `window.__BACKEND__` before page scripts run, which covers
 * the normal case. It does not survive a webview reload, and it is written
 * from a background thread once the backend is ready -- so a page that loaded
 * early could read it before it exists. Without this the app would silently
 * fall back to the *development* port and token and report a confusing
 * "cannot reach the backend at :8000".
 *
 * Asking the shell is authoritative: it is the process that chose the port.
 */
export async function ensureBackendRuntime(): Promise<boolean> {
  if (!isDesktop()) return true;
  if (window.__BACKEND__ !== undefined) return true;
  try {
    const { invoke } = await core();
    window.__BACKEND__ = await invoke<{ port: number; token: string }>("backend_info");
    return true;
  } catch {
    return false;
  }
}

/** Reveal the backend log directory. No-op outside the desktop shell. */
export async function openLogsFolder(): Promise<void> {
  if (!isDesktop()) return;
  const { invoke } = await core();
  await invoke("open_logs_folder");
}

/** Reveal the per-user data directory. No-op outside the desktop shell. */
export async function openDataFolder(): Promise<void> {
  if (!isDesktop()) return;
  const { invoke } = await core();
  await invoke("open_data_folder");
}

/**
 * Installed application version, or null in a browser.
 *
 * Read from the shell rather than hardcoded in the bundle so it can never
 * disagree with what the installer actually put on disk.
 */
export async function appVersion(): Promise<string | null> {
  if (!isDesktop()) return null;
  try {
    const { getVersion } = await import("@tauri-apps/api/app");
    return await getVersion();
  } catch {
    return null;
  }
}
