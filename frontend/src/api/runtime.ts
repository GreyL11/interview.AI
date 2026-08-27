/**
 * Where the backend lives, and how we authenticate to it.
 *
 * Packaged: the Tauri shell picks a free port and a random token, spawns the
 * Python sidecar with them, and injects both into the page before React runs.
 * Dev: fixed port 8000 and a fixed dev token, with uvicorn started by hand.
 *
 * The token exists because binding to localhost does not stop other local
 * processes from reaching the backend.
 */

const DEV_PORT = 8000;
const DEV_TOKEN = "dev-token";

declare global {
  interface Window {
    __BACKEND__?: { port: number; token: string };
  }
}

export interface BackendRuntime {
  baseUrl: string;
  wsBase: string;
  token: string;
  packaged: boolean;
}

export function backendRuntime(): BackendRuntime {
  const injected = typeof window !== "undefined" ? window.__BACKEND__ : undefined;
  const port = injected?.port ?? DEV_PORT;
  const token = injected?.token ?? DEV_TOKEN;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    wsBase: `ws://127.0.0.1:${port}`,
    token,
    packaged: injected !== undefined,
  };
}
