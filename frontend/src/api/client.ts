import type {
  AudioDevice,
  DeleteResponse,
  DocumentRecord,
  DocumentStatus,
  IngestResponse,
  KnowledgeType,
  ModelStatus,
  ProviderKeyResult,
  SessionDetail,
  SessionListItem,
  SettingsUpdate,
  SettingsView,
  UploadResponse,
} from "./contracts.ts";
import { backendRuntime } from "./runtime.ts";

/**
 * Why a request failed, in the terms a user could act on.
 *
 * `fetch` collapses several genuinely different problems into one opaque
 * "Failed to fetch" — the backend being down, a CORS rejection, and a DNS-level
 * failure are indistinguishable from the promise alone. These names are what
 * the UI switches on so it can say something better than "request failed".
 */
export type ApiFailure =
  | "unconfigured" // the shell has not told us where the backend is
  | "unreachable" // nothing answered on the port
  | "unauthorized" // the token was rejected
  | "server" // the backend answered, with a failure
  | "provider" // the backend reached Groq and Groq failed
  | "unknown";

export class ApiError extends Error {
  readonly status: number;
  readonly failure: ApiFailure;
  /** Technical context for the log. Never shown to the user as-is. */
  readonly diagnostic: string | undefined;

  constructor(
    status: number,
    message: string,
    failure: ApiFailure = "unknown",
    diagnostic?: string,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.failure = failure;
    this.diagnostic = diagnostic;
  }
}

function failureForStatus(status: number): ApiFailure {
  if (status === 401 || status === 403) return "unauthorized";
  // 502 is what the backend returns when the LLM provider itself failed, and
  // the detail it carries is already a user-facing sentence from the provider
  // layer — so it is passed through rather than replaced.
  if (status === 502) return "provider";
  if (status >= 500) return "server";
  return "server";
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const runtime = backendRuntime();

  if (runtime.kind === "unavailable") {
    // Deliberately not a guessed dev-port request: that produced a confident
    // error about :8000 while the real backend was healthy elsewhere.
    throw new ApiError(
      0,
      runtime.reason,
      "unconfigured",
      `backend runtime unavailable for ${path}`,
    );
  }

  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${runtime.token}`);

  let response: Response;
  try {
    response = await fetch(`${runtime.baseUrl}${path}`, { ...init, headers });
  } catch (cause) {
    // `fetch` rejects identically for a refused connection and a blocked
    // cross-origin request, so this cannot honestly distinguish them. It says
    // the thing that is true of both and keeps the technical text for the log.
    const diagnostic = cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);
    console.error(`api ${path} failed:`, diagnostic);
    throw new ApiError(
      0,
      "Could not reach the local engine.",
      "unreachable",
      `${runtime.baseUrl}${path} — ${diagnostic}`,
    );
  }

  if (!response.ok) {
    const detail = await errorDetail(response);
    const failure = failureForStatus(response.status);
    throw new ApiError(
      response.status,
      failure === "unauthorized"
        ? "The local engine rejected this app's access token. Restart Call Assistant."
        : detail,
      failure,
      `${path} -> ${response.status}`,
    );
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    // An object detail is a validation error shape: useful in the log, but not
    // a sentence, so the user gets a plain one instead of raw JSON.
    console.error("api error detail:", body.detail ?? body);
    return `The local engine reported an error (${response.status}).`;
  } catch {
    return `The local engine reported an error (${response.status} ${response.statusText}).`;
  }
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  // -------------------------------------------------------------- documents

  listDocuments: (filters: { knowledge_type?: KnowledgeType; status?: DocumentStatus } = {}) =>
    request<DocumentRecord[]>(`/documents${query(filters)}`),

  getDocument: (id: string) => request<DocumentRecord>(`/documents/${id}`),

  /** Upload is a raw body, not multipart: the backend drops the
   * python-multipart dependency and the client just posts the file blob. */
  uploadDocument: (file: File, knowledgeType: KnowledgeType) =>
    request<UploadResponse>(
      `/documents${query({ filename: file.name, knowledge_type: knowledgeType })}`,
      {
        method: "POST",
        body: file,
        headers: { "Content-Type": "application/octet-stream" },
      },
    ),

  ingestDocument: (id: string) =>
    request<IngestResponse>(`/documents/${id}/ingest`, { method: "POST" }),

  deleteDocument: (id: string) =>
    request<DeleteResponse>(`/documents/${id}`, { method: "DELETE" }),

  // --------------------------------------------------------------- sessions

  createSession: (title = "") =>
    request<{ session_id: string }>(`/sessions${query({ title })}`, { method: "POST" }),

  listSessions: (limit = 50, offset = 0) =>
    request<SessionListItem[]>(`/sessions${query({ limit, offset })}`),

  getSession: (id: string) => request<SessionDetail>(`/sessions/${id}`),

  endSession: (id: string) => request<unknown>(`/sessions/${id}/end`, { method: "POST" }),

  deleteSession: (id: string) => request<unknown>(`/sessions/${id}`, { method: "DELETE" }),

  // --------------------------------------------------------------- settings

  getSettings: () => request<SettingsView>("/settings"),

  updateSettings: (update: SettingsUpdate) =>
    request<SettingsView>("/settings", {
      method: "PUT",
      body: JSON.stringify(update),
      headers: { "Content-Type": "application/json" },
    }),

  /** Save or replace a provider key. The key is sent once and never read back;
   * the response reports status only. */
  setProviderKey: (provider: string, apiKey: string) =>
    request<ProviderKeyResult>(`/providers/${provider}/key`, {
      method: "PUT",
      body: JSON.stringify({ api_key: apiKey }),
      headers: { "Content-Type": "application/json" },
    }),

  removeProviderKey: (provider: string) =>
    request<ProviderKeyResult>(`/providers/${provider}/key`, { method: "DELETE" }),

  audioDevices: () => request<AudioDevice[]>("/audio/devices"),

  modelStatus: () => request<ModelStatus[]>("/models/status"),
};
