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

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const runtime = backendRuntime();
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${runtime.token}`);

  let response: Response;
  try {
    response = await fetch(`${runtime.baseUrl}${path}`, { ...init, headers });
    } catch (cause) {
      const detail =
        cause instanceof Error
          ? `${cause.name}: ${cause.message}`
          : String(cause);

      throw new ApiError(
        0,
        `Cannot reach the backend at ${runtime.baseUrl}. ${detail}`,
      );
    }

  if (!response.ok) {
    throw new ApiError(response.status, await errorDetail(response));
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

async function errorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `${response.status} ${response.statusText}`;
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
