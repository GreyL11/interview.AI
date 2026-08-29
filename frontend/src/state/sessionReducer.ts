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
  /** Server time from question.detected to the first answer.delta -- how long
   * the candidate actually stared at nothing. This, not `latencyMs`, is the
   * number that matters live. */
  firstTokenMs: number | null;
  /** Server time from question.detected to answer.completed, i.e. the whole
   * generation including text the user was already reading. Always >=
   * `firstTokenMs`; on its own it overstates the perceived wait. */
  latencyMs: number | null;
  /** question.detected `ts`, kept only to derive `firstTokenMs`. */
  detectedTs: string | null;
  cancelReason: CancelReason | null;
  /** Superseded *after* useful text had already streamed, so
   * `streamingSummary` holds a partial answer worth showing in history.
   * A plain cancellation (nothing streamed yet) leaves this false. */
  interrupted: boolean;
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

/** Milliseconds between two server event timestamps, or null if either is
 * missing or unparseable. Server-stamped on purpose: the reducer stays pure
 * and the tests stay deterministic. */
function elapsedMs(from: string | null | undefined, to: string): number | null {
  if (!from) return null;
  const start = Date.parse(from);
  const end = Date.parse(to);
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return Math.max(0, end - start);
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
    firstTokenMs: null,
    latencyMs: null,
    detectedTs: event.ts,
    cancelReason: null,
    interrupted: false,
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
              cancelReason: "superseded" as CancelReason,
              // No answer.cancelled arrived, so infer it: text already on
              // screen means this was an interruption, not a bare cancel.
              interrupted: base.current.streamingSummary.trim() !== "" }]
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
        // First delta only -- later ones are just more of the same answer.
        firstTokenMs:
          base.current?.firstTokenMs ?? elapsedMs(base.current?.detectedTs, event.ts),
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
        // Optional backend field; fall back to whether anything streamed so
        // an older backend still renders history correctly.
        interrupted:
          bool(event.data, "interrupted") ||
          (base.current?.streamingSummary ?? "").trim() !== "",
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

/**
 * One word for "what is the app doing right now", derived from state the
 * reducer already holds. A selector rather than a stored field: a second copy
 * of this could disagree with the events it was derived from, and this is
 * exactly the thing the user must be able to trust at a glance.
 */
export type LiveStatus =
  | "disconnected"
  | "connecting"
  | "error"
  | "thinking"
  | "answering"
  | "question_detected"
  | "listening"
  | "idle";

export function liveStatus(state: SessionState): LiveStatus {
  if (state.connection === "closed" || state.connection === "idle") {
    return state.sessionId === null ? "idle" : "disconnected";
  }
  if (state.connection === "connecting" || state.connection === "reconnecting") {
    return "connecting";
  }
  const current = state.current;
  if (current !== null) {
    if (current.phase === "failed") return "error";
    // Retrieving and streaming-with-no-text-yet both read as "working on it";
    // splitting them would flicker the indicator for no informational gain.
    if (current.phase === "retrieving") return "thinking";
    if (current.phase === "streaming") {
      return displaySummary(current) === "" ? "thinking" : "answering";
    }
    if (current.phase === "detected") return "question_detected";
  }
  return state.audio === "ok" ? "listening" : "idle";
}

/** How a finished turn should read in the history list. Separate from
 * `TurnPhase` because "cancelled" covers two outcomes the user experiences
 * very differently: text they were mid-read of, versus nothing at all. */
export type HistoryStatus = "answered" | "interrupted" | "cancelled" | "failed" | "active";

export function historyStatus(turn: TurnView): HistoryStatus {
  if (!isTerminal(turn.phase)) return "active";
  if (turn.phase === "answered") return "answered";
  if (turn.phase === "failed") return "failed";
  return turn.interrupted && turn.streamingSummary.trim() !== ""
    ? "interrupted"
    : "cancelled";
}

/** Whether a history row has anything worth expanding.
 *
 * A turn cancelled before any useful text produced no answer, so offering an
 * expander there would promise content that does not exist. Reads the same
 * TurnView the active panel reads -- there is deliberately no second answer
 * store to keep in sync. */
export function hasExpandableAnswer(turn: TurnView): boolean {
  const status = historyStatus(turn);
  if (status === "answered") return turn.answer !== null;
  if (status === "interrupted") return turn.streamingSummary.trim() !== "";
  return false;
}
