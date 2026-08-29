/**
 * Where the backend lives, and how we authenticate to it.
 *
 * Packaged: the Tauri shell picks a free port and a random token, spawns the
 * Python sidecar with them, and hands both to the page — first by injecting
 * `window.__BACKEND__`, and authoritatively via the `backend_info` command
 * (see `ensureBackendRuntime`).
 * Dev: fixed port 8000 and a fixed dev token, with uvicorn started by hand.
 *
 * The token exists because binding to localhost does not stop other local
 * processes from reaching the backend.
 *
 * The one rule this module exists to enforce: **the development fallback must
 * never apply inside the desktop app.** Silently falling back to :8000 there
 * produces "Cannot reach the backend at http://127.0.0.1:8000" while the real
 * backend sits healthy on its actual port — an error that points at the wrong
 * problem and cost real debugging time.
 */

const DEV_PORT = 8000;
const DEV_TOKEN = "dev-token";

declare global {
  interface Window {
    __BACKEND__?: unknown;
  }
}

/** A validated port/token pair. */
export interface BackendConfig {
  port: number;
  token: string;
}

export type BackendRuntime =
  | {
      /** The backend's location is known and requests may be made. */
      readonly kind: "ready";
      readonly baseUrl: string;
      readonly wsBase: string;
      readonly token: string;
      /** True when these values came from the desktop shell. */
      readonly packaged: boolean;
    }
  | {
      /** Inside the desktop shell, but the shell has not supplied a port yet.
       * Requests must fail with this reason rather than guessing. */
      readonly kind: "unavailable";
      readonly reason: string;
    };

/**
 * Is this page running inside the packaged desktop shell?
 *
 * Deliberately not "did `__BACKEND__` get injected". Those are different
 * questions, and answering the first with the second is what produced the
 * silent dev-port fallback: a desktop launch where injection had not landed
 * yet looked exactly like a browser, so the app confidently used :8000.
 */
export function isPackaged(): boolean {
  return typeof window !== "undefined" && window.__TAURI_INTERNALS__ !== undefined;
}

/**
 * Narrow whatever the shell injected into a config, or null.
 *
 * `window.__BACKEND__` is written by an `eval` from the Rust side, so it is
 * typed as `unknown` and checked rather than trusted: a malformed value must
 * produce a clear failure, not a `NaN` port that yields "failed to fetch".
 */
export function parseBackendConfig(value: unknown): BackendConfig | null {
  if (typeof value !== "object" || value === null) return null;
  const candidate = value as { port?: unknown; token?: unknown };

  const { port, token } = candidate;
  if (typeof port !== "number" || !Number.isInteger(port) || port <= 0 || port > 65535) {
    return null;
  }
  if (typeof token !== "string" || token === "") return null;

  return { port, token };
}

/** Store a config the shell handed us. Rejects anything malformed. */
export function setBackendConfig(value: unknown): boolean {
  const config = parseBackendConfig(value);
  if (config === null) return false;
  window.__BACKEND__ = config;
  return true;
}

function ready(config: BackendConfig, packaged: boolean): BackendRuntime {
  return {
    kind: "ready",
    baseUrl: `http://127.0.0.1:${config.port}`,
    wsBase: `ws://127.0.0.1:${config.port}`,
    token: config.token,
    packaged,
  };
}

export function backendRuntime(): BackendRuntime {
  const injected =
    typeof window !== "undefined" ? parseBackendConfig(window.__BACKEND__) : null;

  if (injected !== null) return ready(injected, isPackaged());

  if (isPackaged()) {
    // The port was chosen at random by the shell; there is nothing to guess.
    // `ensureBackendRuntime()` is expected to have filled this in before any
    // request — see App.tsx — so reaching here means that failed.
    return {
      kind: "unavailable",
      reason:
        typeof window !== "undefined" && window.__BACKEND__ !== undefined
          ? "The local engine reported an address this app could not read."
          : "The local engine has not reported its address yet.",
    };
  }

  // Browser development only: uvicorn started by hand on the documented port.
  return ready({ port: DEV_PORT, token: DEV_TOKEN }, false);
}

/** The dev-mode defaults, exported so tests and diagnostics agree with them. */
export const DEV_RUNTIME = { port: DEV_PORT, token: DEV_TOKEN } as const;
