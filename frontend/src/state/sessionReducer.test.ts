/**
 * Runs on the Node built-in test runner with native TypeScript type stripping:
 *   node --test src/state/sessionReducer.test.ts
 * No install, no bundler, no DOM.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import type { ServerEvent, ServerEventType } from "../api/contracts.ts";
import {
  displaySummary,
  initialSessionState,
  isAnswering,
  sessionReducer,
  visibleTurns,
  type SessionState,
} from "./sessionReducer.ts";

let seq = 0;

function ev(
  type: ServerEventType,
  data: Record<string, unknown> = {},
  turnId: number | null = null,
  overrideSeq?: number,
): ServerEvent {
  seq += 1;
  return {
    type,
    seq: overrideSeq ?? seq,
    ts: "2026-01-01T00:00:00Z",
    turn_id: turnId,
    data,
  };
}

function apply(state: SessionState, ...events: ServerEvent[]): SessionState {
  return events.reduce((acc, event) => sessionReducer(acc, { kind: "event", event }), state);
}

function fresh(): SessionState {
  seq = 0;
  return { ...initialSessionState };
}

const CLASSIFICATION = {
  is_question: true,
  category: "SCENARIO",
  domain: "DATA_ENGINEERING",
  requires_personal_context: false,
  requires_rag: false,
  requires_reasoning: true,
  requires_code: false,
  confidence: 0.9,
};

function answer(summary: string) {
  return {
    summary,
    key_points: ["a", "b"],
    detailed_answer: "detail",
    approach: null,
    code: null,
    complexity: null,
    edge_cases: null,
    warnings: [],
  };
}

// --------------------------------------------------------------- connection

test("connection state transitions", () => {
  const state = sessionReducer(fresh(), { kind: "connection", state: "connecting" });
  assert.equal(state.connection, "connecting");
});

test("session.started opens the connection and records the id", () => {
  const state = apply(fresh(), ev("session.started", { session_id: "s-1" }));
  assert.equal(state.sessionId, "s-1");
  assert.equal(state.connection, "open");
  assert.equal(state.ended, false);
});

test("session.ended marks the session finished and audio off", () => {
  const state = apply(fresh(), ev("session.started", { session_id: "s-1" }), ev("session.ended"));
  assert.equal(state.ended, true);
  assert.equal(state.audio, "off");
});

test("reset returns to the initial state", () => {
  const dirty = apply(fresh(), ev("session.started", { session_id: "s-1" }));
  assert.deepEqual(sessionReducer(dirty, { kind: "reset" }), initialSessionState);
});

// ------------------------------------------------------------------- replay

test("seq watermark advances", () => {
  const state = apply(fresh(), ev("session.started", { session_id: "s-1" }));
  assert.equal(state.lastSeq, 1);
});

test("replayed frames at or below the watermark are ignored", () => {
  let state = apply(fresh(), ev("transcript.final", { text: "first", source: "LOOPBACK" }));
  assert.equal(state.transcript.length, 1);

  // Reconnect replays the same frame; it must not be appended twice.
  state = apply(state, ev("transcript.final", { text: "first", source: "LOOPBACK" }, null, 1));
  assert.equal(state.transcript.length, 1);
});

test("a higher seq after replay is still applied", () => {
  let state = apply(fresh(), ev("transcript.final", { text: "one", source: "LOOPBACK" }));
  state = apply(state, ev("transcript.final", { text: "one", source: "LOOPBACK" }, null, 1));
  state = apply(state, ev("transcript.final", { text: "two", source: "LOOPBACK" }, null, 9));
  assert.equal(state.transcript.length, 2);
  assert.equal(state.lastSeq, 9);
});

// --------------------------------------------------------------- transcript

test("partial transcript is held per source, not appended", () => {
  const state = apply(fresh(), ev("transcript.partial", { text: "how would you", source: "LOOPBACK" }));
  assert.equal(state.partials.LOOPBACK, "how would you");
  assert.equal(state.transcript.length, 0);
});

test("a later partial replaces the earlier one", () => {
  const state = apply(
    fresh(),
    ev("transcript.partial", { text: "how", source: "LOOPBACK" }),
    ev("transcript.partial", { text: "how would you", source: "LOOPBACK" }),
  );
  assert.equal(state.partials.LOOPBACK, "how would you");
});

test("final transcript clears that source's partial and appends a line", () => {
  const state = apply(
    fresh(),
    ev("transcript.partial", { text: "how would", source: "LOOPBACK" }),
    ev("transcript.final", { text: "how would you scale it?", source: "LOOPBACK" }),
  );
  assert.equal(state.partials.LOOPBACK, undefined);
  assert.equal(state.transcript.length, 1);
  assert.equal(state.transcript[0]?.text, "how would you scale it?");
});

test("mic and loopback partials are tracked independently", () => {
  const state = apply(
    fresh(),
    ev("transcript.partial", { text: "candidate", source: "MIC" }),
    ev("transcript.partial", { text: "interviewer", source: "LOOPBACK" }),
    ev("transcript.final", { text: "interviewer done", source: "LOOPBACK" }),
  );
  assert.equal(state.partials.MIC, "candidate");
  assert.equal(state.partials.LOOPBACK, undefined);
});

test("transcript is bounded", () => {
  let state = fresh();
  for (let i = 0; i < 620; i += 1) {
    state = apply(state, ev("transcript.final", { text: `line ${i}`, source: "MIC" }));
  }
  assert.equal(state.transcript.length, 500);
  assert.equal(state.transcript[499]?.text, "line 619");
});

// ---------------------------------------------------------------- questions

test("question.rejected is surfaced with its reason", () => {
  const state = apply(fresh(), ev("question.rejected", { text: "yeah", reason: "too_short" }));
  assert.equal(state.lastRejected?.reason, "too_short");
});

test("question.detected creates the current turn", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "How would you dedupe?", classification: CLASSIFICATION }, 1),
  );
  assert.equal(state.current?.turnId, 1);
  assert.equal(state.current?.question, "How would you dedupe?");
  assert.equal(state.current?.classification?.category, "SCENARIO");
  assert.equal(state.current?.phase, "detected");
});

test("a new question clears the previous rejection notice", () => {
  const state = apply(
    fresh(),
    ev("question.rejected", { text: "yeah", reason: "too_short" }),
    ev("question.detected", { question: "Real question?", classification: CLASSIFICATION }, 1),
  );
  assert.equal(state.lastRejected, null);
});

// ------------------------------------------------------------ answer stream

test("full happy path reaches answered", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?", classification: CLASSIFICATION }, 1),
    ev("answer.started", { question: "Q?" }, 1),
    ev("answer.delta", { summary: "Make the" }, 1),
    ev("answer.delta", { summary: "Make the pipeline idempotent." }, 1),
    ev("answer.completed", {
      answer: answer("Make the pipeline idempotent."),
      context_found: true,
      latency_ms: 1234,
      retrieval_hits: [{ chunk_id: "c1", document_id: "d1", score: 0.8, title: "CV" }],
    }, 1),
  );

  assert.equal(state.current?.phase, "answered");
  assert.equal(state.current?.answer?.summary, "Make the pipeline idempotent.");
  assert.equal(state.current?.contextFound, true);
  assert.equal(state.current?.latencyMs, 1234);
  assert.equal(state.current?.hits.length, 1);
  assert.equal(state.history.length, 1);
});

test("deltas build the streaming summary before completion", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?" }, 1),
    ev("answer.delta", { summary: "Make the pipe" }, 1),
  );
  assert.equal(state.current?.streamingSummary, "Make the pipe");
  assert.equal(state.current?.answer, null);
  assert.equal(displaySummary(state.current!), "Make the pipe");
});

test("answer.retrieving marks the retrieval phase", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?" }, 1),
    ev("answer.retrieving", { knowledge_types: ["RESUME"] }, 1),
  );
  assert.equal(state.current?.phase, "retrieving");
});

test("answer.started without a preceding question.detected still creates a turn", () => {
  const state = apply(fresh(), ev("answer.started", { question: "Q?" }, 7));
  assert.equal(state.current?.turnId, 7);
  assert.equal(state.current?.phase, "streaming");
});

test("answer.error retires the turn as failed", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?" }, 1),
    ev("answer.error", { code: "LLMError", message: "GEMINI_API_KEY is not configured" }, 1),
  );
  assert.equal(state.current?.phase, "failed");
  assert.match(state.current?.errorMessage ?? "", /GEMINI_API_KEY/);
  assert.equal(state.history.length, 1);
});

test("answer.cancelled retires the turn with its reason", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?" }, 1),
    ev("answer.cancelled", { reason: "user_stop" }, 1),
  );
  assert.equal(state.current?.phase, "cancelled");
  assert.equal(state.current?.cancelReason, "user_stop");
});

// ----------------------------------------------------------- stale dropping

test("a late delta from a superseded turn is ignored", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("question.detected", { question: "Second?" }, 2),
    ev("answer.delta", { summary: "answer to the FIRST question" }, 1),
  );
  // The stale delta must not overwrite the newer question's answer.
  assert.equal(state.current?.turnId, 2);
  assert.equal(state.current?.streamingSummary, "");
});

test("a late completion from a superseded turn is ignored", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("question.detected", { question: "Second?" }, 2),
    ev("answer.completed", { answer: answer("stale answer"), context_found: false }, 1),
  );
  assert.equal(state.current?.phase, "detected");
  assert.equal(state.current?.answer, null);
});

test("superseding an unfinished turn retires it as cancelled", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.delta", { summary: "partial" }, 1),
    ev("question.detected", { question: "Second?" }, 2),
  );
  assert.equal(state.history.length, 1);
  assert.equal(state.history[0]?.turnId, 1);
  assert.equal(state.history[0]?.phase, "cancelled");
  assert.equal(state.history[0]?.cancelReason, "superseded");
});

test("superseding a finished turn does not duplicate it in history", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.completed", { answer: answer("done"), context_found: false }, 1),
    ev("question.detected", { question: "Second?" }, 2),
  );
  assert.equal(state.history.length, 1);
  assert.equal(state.history[0]?.phase, "answered");
});

test("events for a newer turn are never treated as stale", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.delta", { summary: "ok" }, 5),
  );
  // turn 5 > turn 1: not stale, so the delta applies to the current turn.
  assert.equal(state.current?.streamingSummary, "ok");
});

test("rapid-fire questions leave exactly one live turn", () => {
  let state = fresh();
  for (let i = 1; i <= 5; i += 1) {
    state = apply(state, ev("question.detected", { question: `Q${i}?` }, i));
  }
  state = apply(state, ev("answer.completed", { answer: answer("final"), context_found: false }, 5));

  assert.equal(state.current?.turnId, 5);
  assert.equal(state.current?.phase, "answered");
  assert.equal(state.history.filter((t) => t.phase === "answered").length, 1);
});

// -------------------------------------------------------------------- audio

test("audio.start request shows the starting state", () => {
  const state = sessionReducer(fresh(), { kind: "audio-requested" });
  assert.equal(state.audio, "starting");
});

test("session.status ok records channels", () => {
  const state = apply(fresh(), ev("session.status", { audio: "ok", channels: ["LOOPBACK", "MIC"] }));
  assert.equal(state.audio, "ok");
  assert.deepEqual(state.channels, ["LOOPBACK", "MIC"]);
});

test("session.status error keeps the message", () => {
  const state = apply(fresh(), ev("session.status", { audio: "error", message: "no device" }));
  assert.equal(state.audio, "error");
  assert.equal(state.audioMessage, "no device");
});

test("session.status stopped clears the error message", () => {
  const state = apply(
    fresh(),
    ev("session.status", { audio: "error", message: "no device" }),
    ev("session.status", { audio: "stopped" }),
  );
  assert.equal(state.audio, "stopped");
  assert.equal(state.audioMessage, null);
});

// ---------------------------------------------------------------- selectors

test("visibleTurns lists the live turn first, then history newest-first", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q1?" }, 1),
    ev("answer.completed", { answer: answer("a1"), context_found: false }, 1),
    ev("question.detected", { question: "Q2?" }, 2),
    ev("answer.completed", { answer: answer("a2"), context_found: false }, 2),
    ev("question.detected", { question: "Q3?" }, 3),
  );
  assert.deepEqual(visibleTurns(state).map((t) => t.turnId), [3, 2, 1]);
});

test("visibleTurns does not duplicate the retired live turn", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q1?" }, 1),
    ev("answer.completed", { answer: answer("a1"), context_found: false }, 1),
  );
  assert.equal(visibleTurns(state).length, 1);
});

test("isAnswering is true only while a turn is unfinished", () => {
  let state = apply(fresh(), ev("question.detected", { question: "Q?" }, 1));
  assert.equal(isAnswering(state), true);

  state = apply(state, ev("answer.completed", { answer: answer("x"), context_found: false }, 1));
  assert.equal(isAnswering(state), false);
});

test("isAnswering is false on a fresh state", () => {
  assert.equal(isAnswering(fresh()), false);
});

test("displaySummary prefers the completed answer over the stream", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?" }, 1),
    ev("answer.delta", { summary: "partial text" }, 1),
    ev("answer.completed", { answer: answer("final text"), context_found: false }, 1),
  );
  assert.equal(displaySummary(state.current!), "final text");
});

// ------------------------------------------------------------------ misc

test("error event is recorded without disturbing the turn", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "Q?" }, 1),
    ev("error", { code: "BadMessage", message: "not json" }),
  );
  assert.equal(state.error, "not json");
  assert.equal(state.current?.turnId, 1);
});

test("pong does not change anything except the watermark", () => {
  const before = apply(fresh(), ev("session.started", { session_id: "s" }));
  const after = apply(before, ev("pong"));
  assert.equal(after.sessionId, before.sessionId);
  assert.ok(after.lastSeq > before.lastSeq);
});

test("malformed payloads fall back rather than throwing", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: 42, classification: "nonsense" }, 1),
    ev("answer.completed", { answer: null, context_found: "yes", latency_ms: "slow" }, 1),
  );
  assert.equal(state.current?.question, "");
  assert.equal(state.current?.classification, null);
  assert.equal(state.current?.contextFound, false);
  assert.equal(state.current?.latencyMs, 0);
});
