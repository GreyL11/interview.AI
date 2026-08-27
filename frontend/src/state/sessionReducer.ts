/**
 * Pure reduction of WebSocket events into practice-session UI state.
 *
 * All the awkward logic lives here rather than in components: stale-turn
 * dropping, replay de-duplication, and progressive answer assembly. Keeping it
 * a pure function of (state, event) means it can be tested directly with the
 * Node test runner, no DOM and no build step.
 */

import type {
  Answer,
  CancelReason,
  Classification,
  RejectionReason,
  RetrievalHitView,
  ServerEvent,
  TranscriptSource,
} from "../api/contracts.ts";
import { bool, num, str } from "../api/contracts.ts";

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed";

export type AudioState = "off" | "starting" | "ok" | "stopped" | "error";

export type TurnPhase =
  | "detected"
  | "retrieving"
  | "streaming"
  | "answered"
  | "cancelled"
  | "failed";

export interface TranscriptLine {
  id: string;
  source: TranscriptSource;
  text: string;
  ts: string;
}

export interface TurnView {
  turnId: number;
  question: string;
  classification: Classification | null;
  phase: TurnPhase;
  /** Progressive summary from answer.delta, replaced by the real answer on
   * completion. Shown while the model is still typing. */
  streamingSummary: string;
  answer: Answer | null;
  contextFound: boolean;
  hits: RetrievalHitView[];
  latencyMs: number | null;
  cancelReason: CancelReason | null;
  errorMessage: string | null;
}

export interface RejectedQuestion {
  text: string;
  reason: RejectionReason | null;
}

export interface SessionState {
  connection: ConnectionState;
  sessionId: string | null;
  /** Highest seq applied. Sent as ?since_seq= on reconnect and used to ignore
   * frames the reducer has already seen. */
  lastSeq: number;
  audio: AudioState;
  audioMessage: string | null;
  channels: string[];
  /** Live partial per source, cleared when that source produces a final. */
  partials: Partial<Record<TranscriptSource, string>>;
  transcript: TranscriptLine[];
  current: TurnView | null;
  history: TurnView[];
  lastRejected: RejectedQuestion | null;
  error: string | null;
  ended: boolean;
}

export const initialSessionState: SessionState = {
  connection: "idle",
  sessionId: null,
  lastSeq: 0,
  audio: "off",
  audioMessage: null,
  channels: [],
  partials: {},
  transcript: [],
  current: null,
  history: [],
  lastRejected: null,
  error: null,
  ended: false,
};

export type Action =
  | { kind: "connection"; state: ConnectionState }
  | { kind: "event"; event: ServerEvent }
  | { kind: "audio-requested" }
  | { kind: "reset" };

const MAX_TRANSCRIPT_LINES = 500;

function source(event: ServerEvent): TranscriptSource {
  const value = str(event.data, "source", "MANUAL");
  return value === "MIC" || value === "LOOPBACK" ? value : "MANUAL";
}

function classificationOf(event: ServerEvent): Classification | null {
  const raw = event.data["classification"];
  return typeof raw === "object" && raw !== null ? (raw as Classification) : null;
}

function newTurn(event: ServerEvent, turnId: number): TurnView {
  return {
    turnId,
    question: str(event.data, "question"),
    classification: classificationOf(event),
    phase: "detected",
    streamingSummary: "",
    answer: null,
    contextFound: false,
    hits: [],
    latencyMs: null,
    cancelReason: null,
    errorMessage: null,
  };
}

/**
 * Is this event still relevant?
 *
 * The server already drops events from superseded turns, but the client keeps
 * its own guard: a frame can be in flight when the turn changes, and a late
 * delta overwriting a newer question's answer is exactly the confusing failure
 * this design set out to avoid.
 */
function isStale(state: SessionState, event: ServerEvent): boolean {
  if (event.turn_id === null || state.current === null) return false;
  return event.turn_id < state.current.turnId;
}

function withCurrent(state: SessionState, update: Partial<TurnView>): SessionState {
  if (state.current === null) return state;
  return { ...state, current: { ...state.current, ...update } };
}

/** Move the current turn into history in a terminal phase. */
function retire(state: SessionState, update: Partial<TurnView>): SessionState {
  if (state.current === null) return state;
  const finished: TurnView = { ...state.current, ...update };
  return { ...state, current: finished, history: [...state.history, finished] };
}

function appendTranscript(state: SessionState, event: ServerEvent): TranscriptLine[] {
  const line: TranscriptLine = {
    id: `${event.seq}`,
    source: source(event),
    text: str(event.data, "text"),
    ts: event.ts,
  };
  const next = [...state.transcript, line];
  // Bound the list: an hour-long session would otherwise grow without limit.
  return next.length > MAX_TRANSCRIPT_LINES
    ? next.slice(next.length - MAX_TRANSCRIPT_LINES)
    : next;
}

function applyEvent(state: SessionState, event: ServerEvent): SessionState {
  // Replay after reconnect re-sends frames already applied. seq is monotonic
  // per session, so anything at or below the watermark is a duplicate.
  if (event.seq !== 0 && event.seq <= state.lastSeq) return state;

  const base: SessionState = {
    ...state,
    lastSeq: event.seq > state.lastSeq ? event.seq : state.lastSeq,
  };

  if (isStale(base, event)) return base;

  switch (event.type) {
    case "session.started":
      return {
        ...base,
        sessionId: str(event.data, "session_id", base.sessionId ?? ""),
        connection: "open",
        ended: false,
      };

    case "session.status": {
      const audio = str(event.data, "audio");
      const channels = Array.isArray(event.data["channels"])
        ? (event.data["channels"] as string[])
        : base.channels;
      const nextAudio: AudioState =
        audio === "ok" || audio === "stopped" || audio === "error"
          ? audio
          : base.audio;
      return {
        ...base,
        audio: nextAudio,
        audioMessage: audio === "error" ? str(event.data, "message") : null,
        channels: audio === "ok" ? channels : base.channels,
      };
    }

    case "session.ended":
      return { ...base, ended: true, audio: "off" };

    case "transcript.partial":
      return {
        ...base,
        partials: { ...base.partials, [source(event)]: str(event.data, "text") },
      };

    case "transcript.final": {
      const partials = { ...base.partials };
      delete partials[source(event)];
      return { ...base, partials, transcript: appendTranscript(base, event) };
    }

    case "question.rejected":
      return {
        ...base,
        lastRejected: {
          text: str(event.data, "text"),
          reason: (str(event.data, "reason") || null) as RejectionReason | null,
        },
      };

    case "question.detected": {
      if (event.turn_id === null) return base;
      // A new question supersedes the previous one. If the old turn never
      // reached a terminal event, retire it as cancelled so it doesn't sit in
      // the UI looking like it is still working.
      const history =
        base.current !== null && !isTerminal(base.current.phase)
          ? [...base.history, { ...base.current, phase: "cancelled" as TurnPhase,
              cancelReason: "superseded" as CancelReason }]
          : base.history;
      return {
        ...base,
        history,
        current: newTurn(event, event.turn_id),
        lastRejected: null,
      };
    }

    case "answer.started":
      // Tolerate a missing question.detected (older backend, or a replay that
      // starts mid-turn) by synthesising the turn here.
      if (base.current === null && event.turn_id !== null) {
        return { ...base, current: { ...newTurn(event, event.turn_id), phase: "streaming" } };
      }
      return withCurrent(base, { phase: "streaming" });

    case "answer.retrieving":
      return withCurrent(base, { phase: "retrieving" });

    case "answer.delta":
      return withCurrent(base, {
        phase: "streaming",
        streamingSummary: str(event.data, "summary"),
      });

    case "answer.completed": {
      const answer = event.data["answer"] as Answer | undefined;
      const hits = Array.isArray(event.data["retrieval_hits"])
        ? (event.data["retrieval_hits"] as RetrievalHitView[])
        : [];
      return retire(base, {
        phase: "answered",
        answer: answer ?? null,
        streamingSummary: answer?.summary ?? base.current?.streamingSummary ?? "",
        contextFound: bool(event.data, "context_found"),
        hits,
        latencyMs: num(event.data, "latency_ms"),
      });
    }

    case "answer.cancelled":
      return retire(base, {
        phase: "cancelled",
        cancelReason: (str(event.data, "reason") || null) as CancelReason | null,
      });

    case "answer.error":
      return retire(base, {
        phase: "failed",
        errorMessage: str(event.data, "message", "The answer failed."),
      });

    case "error":
      return { ...base, error: str(event.data, "message", "Unknown error") };

    case "pong":
      return base;

    default:
      return base;
  }
}

export function isTerminal(phase: TurnPhase): boolean {
  return phase === "answered" || phase === "cancelled" || phase === "failed";
}

export function sessionReducer(state: SessionState, action: Action): SessionState {
  switch (action.kind) {
    case "connection":
      return { ...state, connection: action.state };
    case "audio-requested":
      return { ...state, audio: "starting", audioMessage: null };
    case "event":
      return applyEvent(state, action.event);
    case "reset":
      return { ...initialSessionState };
    default:
      return state;
  }
}

// ------------------------------------------------------------------ selectors

/** Turns worth showing, newest first: the live one plus finished ones, without
 * duplicating the live turn once it has been retired. */
export function visibleTurns(state: SessionState): TurnView[] {
  const seen = new Set<number>();
  const out: TurnView[] = [];
  if (state.current !== null) {
    out.push(state.current);
    seen.add(state.current.turnId);
  }
  for (let i = state.history.length - 1; i >= 0; i -= 1) {
    const turn = state.history[i];
    if (turn !== undefined && !seen.has(turn.turnId)) {
      out.push(turn);
      seen.add(turn.turnId);
    }
  }
  return out;
}

export function isAnswering(state: SessionState): boolean {
  return state.current !== null && !isTerminal(state.current.phase);
}

/** What the answer panel should show right now: the finished summary, or the
 * partial text still streaming in. */
export function displaySummary(turn: TurnView): string {
  return turn.answer?.summary ?? turn.streamingSummary;
}
