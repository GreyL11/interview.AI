/**
 * Mirrors the backend Pydantic models and WebSocket event contracts.
 *
 * Hand-written rather than generated: the WebSocket protocol is not described
 * by OpenAPI, and splitting the types across a generator and a hand-written
 * union would make drift harder to spot, not easier. Union types are used
 * instead of TypeScript enums so every file here stays erasable syntax, which
 * is what lets Node run and test this logic with no build step.
 */

// ---------------------------------------------------------------- documents

export const KNOWLEDGE_TYPES = [
  "RESUME",
  "PERSONAL",
  "EXPERIENCE",
  "PROJECT",
  "BEHAVIORAL_STORY",
  "TECHNICAL",
  "REFERENCE",
] as const;
export type KnowledgeType = (typeof KNOWLEDGE_TYPES)[number];

/** Knowledge types describing things the user actually did. Mirrors
 * PERSONAL_KNOWLEDGE_TYPES in the backend. */
export const PERSONAL_KNOWLEDGE_TYPES: readonly KnowledgeType[] = [
  "RESUME",
  "PERSONAL",
  "EXPERIENCE",
  "PROJECT",
  "BEHAVIORAL_STORY",
];

export type FileType = "PDF" | "DOCX" | "MARKDOWN" | "TXT";
export type DocumentStatus = "UPLOADED" | "PROCESSING" | "READY" | "FAILED";

export interface DocumentRecord {
  document_id: string;
  filename: string;
  file_type: FileType;
  knowledge_type: KnowledgeType;
  title: string;
  source: string;
  created_at: string;
  ingested_at: string | null;
  status: DocumentStatus;
  error: string | null;
  chunk_count: number;
  /** What a slow ingest is currently doing ("Reading scanned page 3 of 12..."),
   * or null. Only set while PROCESSING. */
  progress: string | null;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: DocumentStatus;
}

export interface IngestResponse {
  document_id: string;
  status: DocumentStatus;
  chunk_count: number;
  error: string | null;
}

export interface DeleteResponse {
  document_id: string;
  deleted: boolean;
  chunks_removed: number;
  vectors_removed: number;
}

// ----------------------------------------------------------------- answers

export interface Complexity {
  time: string;
  space: string;
}

export interface AnswerSection {
  heading: string;
  content: string;
}

export interface Answer {
  summary: string;
  key_points: string[];
  detailed_answer: string;
  approach: string[] | null;
  code: string | null;
  complexity: Complexity | null;
  edge_cases: string[] | null;
  // Debugging / SQL / system design / behavioral structure, e.g.
  // Likely Cause / Diagnosis / Fix, or Situation / Task / Action / Result.
  sections: AnswerSection[] | null;
  warnings: string[];
}

export type Category =
  | "PERSONAL_EXPERIENCE"
  | "RESUME"
  | "PROJECT"
  | "BEHAVIORAL"
  | "TECHNICAL_KNOWLEDGE"
  | "SYSTEM_DESIGN"
  | "SCENARIO"
  | "CODING"
  | "SQL"
  | "DEBUGGING"
  | "ARCHITECTURE"
  | "FOLLOW_UP"
  | "UNKNOWN";

export interface Classification {
  is_question: boolean;
  category: Category;
  domain: string;
  requires_personal_context: boolean;
  requires_rag: boolean;
  requires_reasoning: boolean;
  requires_code: boolean;
  confidence: number;
}

export interface RetrievalHitView {
  chunk_id: string;
  document_id: string;
  score: number;
  title: string;
}

// ---------------------------------------------------------------- sessions

export type SessionStatus = "ACTIVE" | "ENDED";
export type TurnStatus = "PENDING" | "ANSWERED" | "CANCELLED" | "FAILED";
export type TranscriptSource = "MIC" | "LOOPBACK" | "MANUAL";

export interface SessionListItem {
  session_id: string;
  started_at: string;
  ended_at: string | null;
  status: SessionStatus;
  title: string;
  turn_count: number;
}

export interface StoredTurn {
  turn_id: number | null;
  session_id: string;
  seq: number;
  question: string;
  category: string;
  domain: string;
  confidence: number;
  answer: Answer | null;
  context_found: boolean;
  status: TurnStatus;
  latency_ms: number | null;
  created_at: string;
}

export interface StoredTranscript {
  id: number | null;
  session_id: string;
  turn_id: number | null;
  source: TranscriptSource;
  is_final: boolean;
  text: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface SessionDetail {
  session: {
    session_id: string;
    started_at: string;
    ended_at: string | null;
    status: SessionStatus;
    title: string;
    config: Record<string, unknown>;
  };
  turns: StoredTurn[];
  transcript: StoredTranscript[];
  summary: {
    session_id: string;
    summary: string;
    topics: string[];
    covered_through_seq: number;
    updated_at: string;
  } | null;
}

// ---------------------------------------------------------------- settings

/** How a provider request failed, as classified by the backend. Mirrors
 * `LLMErrorKind`. Never carries the provider's own error text. */
export type ProviderErrorKind =
  | "not_configured"
  | "auth"
  | "model_unavailable"
  | "rate_limit"
  | "timeout"
  | "network"
  | "server"
  | "malformed"
  | "unknown";

/** The cloud LLM provider as Settings sees it. No key material, by design. */
export interface ProviderStatus {
  name: string;
  model: string;
  configured: boolean;
  /** Wired into the running engine this launch. */
  active: boolean;
  /** How the last request failed, or null if the last one succeeded. */
  last_error_kind: ProviderErrorKind | null;
}

/** Result of setting or removing a provider key. Carries no key material and
 * nothing derived from it. */
export interface ProviderKeyResult {
  provider: string;
  configured: boolean;
  /** False when the machine has no OS credential store: the key works now but
   * will not survive a restart. Never assume true. */
  persisted: boolean;
  detail: string;
}

export interface SettingsView {
  providers: ProviderStatus[];
  /** Whether a saved key would actually survive a restart on this machine. */
  secure_storage_available: boolean;
  /** False today: changes apply to the running engine but are not written
   * back to .env. Settings says so rather than implying they survive. */
  settings_persist: boolean;
  groq_key_configured: boolean;
  groq_model: string;
  embedding_model: string;
  stt_model: string;
  stt_device: string;
  stt_compute_type: string;
  chunk_size: number;
  chunk_overlap: number;
  rag_top_k: number;
  rag_min_similarity: number;
  data_dir: string;
  audio_capture_mic: boolean;
  audio_capture_loopback: boolean;
  audio_available: boolean;
}

/** Non-secret settings only. Keys go through setProviderKey/removeProviderKey,
 * which is the single path that also persists them. */
export interface SettingsUpdate {
  groq_model?: string;
  stt_model?: string;
  stt_device?: string;
  rag_top_k?: number;
  rag_min_similarity?: number;
  audio_capture_mic?: boolean;
  audio_capture_loopback?: boolean;
}

export interface AudioDevice {
  index: number;
  name: string;
  channel: "MIC" | "LOOPBACK";
  sample_rate: number;
  is_default: boolean;
}

/** Where a local model is in its download lifecycle. Mirrors the backend's
 * `app.models.status`. */
export const MODEL_STATES = [
  "not_downloaded",
  "downloading",
  "downloaded",
  "loading",
  "ready",
  "failed",
] as const;
export type ModelState = (typeof MODEL_STATES)[number];

export interface ModelStatus {
  name: string;
  kind: string;
  state: ModelState;
  /** Derived from `state` by the backend. Kept because it is what older code
   * read; `state` is the field to render. */
  downloaded: boolean;
  path: string;
  /** Present when `state` is "failed": what went wrong and what to do. */
  detail: string | null;
  /** What the model actually runs on. Null until it has loaded. */
  device: "cpu" | "cuda" | null;
}

// ------------------------------------------------------------ ws: server->client

export const SERVER_EVENTS = [
  "session.started",
  "session.status",
  "session.ended",
  "transcript.partial",
  "transcript.final",
  "question.detected",
  "question.rejected",
  "answer.started",
  "answer.retrieving",
  "answer.delta",
  "answer.completed",
  "answer.cancelled",
  "answer.error",
  "context.attached",
  "context.rejected",
  "error",
  "pong",
] as const;
export type ServerEventType = (typeof SERVER_EVENTS)[number];

export type RejectionReason = "not_a_question" | "too_short" | "low_confidence";

/**
 * Interviewer-pasted material. Mirrors `app.realtime.attachments`
 * (AttachmentKind / RejectReason) -- these strings are the wire contract and
 * must not be invented client-side.
 */
export const ATTACHMENT_KINDS = ["text", "code", "sql", "table", "image"] as const;
export type AttachmentKind = (typeof ATTACHMENT_KINDS)[number];
export type AttachmentRejectReason = "empty" | "too_large" | "unreadable_image";

/** Backend limits, from `Settings.context_attachment_*`. Mirrored so the UI can
 * refuse an oversized paste before encoding it; the backend stays
 * authoritative and rejects independently. */
export const ATTACHMENT_MAX_CHARS = 20_000;
export const ATTACHMENT_MAX_ITEMS = 6;
export type CancelReason = "superseded" | "user_stop" | "session_ended";

/**
 * Every frame the server sends. `seq` is monotonic per session and drives
 * reconnect replay; `turn_id` identifies which question an event belongs to,
 * so events from a superseded turn can be discarded.
 */
export interface ServerEvent {
  type: ServerEventType;
  seq: number;
  ts: string;
  turn_id: number | null;
  data: Record<string, unknown>;
}

// ------------------------------------------------------------ ws: client->server

export type ClientMessage =
  | { type: "question.manual"; data: { text: string } }
  /**
   * Interviewer-provided material for the current turn. `content` is the
   * pasted text verbatim; `image_base64` carries a pasted screenshot instead,
   * which the backend decodes and OCRs. Exactly the shape
   * `app/api/ws.py` reads.
   */
  | {
      type: "context.attach";
      data: {
        kind: AttachmentKind;
        content: string;
        name?: string;
        image_base64?: string;
      };
    }
  | { type: "answer.cancel" }
  | { type: "audio.start" }
  | { type: "audio.stop" }
  | { type: "session.stop" }
  | { type: "ping" };

// ------------------------------------------------------------------ helpers

/** Narrow an unknown JSON frame to a ServerEvent, or null if it isn't one.
 * The socket is local, but a malformed frame must not crash the UI. */
export function parseServerEvent(raw: unknown): ServerEvent | null {
  if (typeof raw !== "object" || raw === null) return null;
  const candidate = raw as Partial<ServerEvent>;
  if (typeof candidate.type !== "string") return null;
  if (!(SERVER_EVENTS as readonly string[]).includes(candidate.type)) return null;
  return {
    type: candidate.type as ServerEventType,
    seq: typeof candidate.seq === "number" ? candidate.seq : 0,
    ts: typeof candidate.ts === "string" ? candidate.ts : "",
    turn_id: typeof candidate.turn_id === "number" ? candidate.turn_id : null,
    data:
      typeof candidate.data === "object" && candidate.data !== null
        ? (candidate.data as Record<string, unknown>)
        : {},
  };
}

export function str(data: Record<string, unknown>, key: string, fallback = ""): string {
  const value = data[key];
  return typeof value === "string" ? value : fallback;
}

export function num(data: Record<string, unknown>, key: string, fallback = 0): number {
  const value = data[key];
  return typeof value === "number" ? value : fallback;
}

export function bool(data: Record<string, unknown>, key: string, fallback = false): boolean {
  const value = data[key];
  return typeof value === "boolean" ? value : fallback;
}
