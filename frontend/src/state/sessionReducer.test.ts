/**
 * Runs on the Node built-in test runner with native TypeScript type stripping:
 *   node --test src/state/sessionReducer.test.ts
 * No install, no bundler, no DOM.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { detectLanguage } from "../components/language.ts";
import type { ServerEvent, ServerEventType } from "../api/contracts.ts";
import {
  displaySummary,
  hasExpandableAnswer,
  historyStatus,
  liveStatus,
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
    ev(
      "answer.error",
      { code: "LLMError", message: "No Groq API key is configured." },
      1,
    ),
  );
  assert.equal(state.current?.phase, "failed");
  assert.match(state.current?.errorMessage ?? "", /Groq API key/);
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

test("an interrupted turn keeps its partial answer in history", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.delta", { summary: "A cache stores" }, 1),
    ev("answer.cancelled", { reason: "superseded", interrupted: true,
                             partial_summary: "A cache stores" }, 1),
  );
  assert.equal(state.history[0]?.interrupted, true);
  assert.equal(state.history[0]?.streamingSummary, "A cache stores");
});

test("a cancellation with no streamed content is not marked interrupted", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.cancelled", { reason: "superseded", interrupted: false }, 1),
  );
  assert.equal(state.history[0]?.interrupted, false);
  assert.equal(state.history[0]?.streamingSummary, "");
});

test("interruption is inferred when the backend omits the field", () => {
  // Backward compatibility: an older backend sends no `interrupted` flag, so
  // the reducer falls back to whether anything had actually streamed.
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.delta", { summary: "partial text" }, 1),
    ev("answer.cancelled", { reason: "superseded" }, 1),
  );
  assert.equal(state.history[0]?.interrupted, true);
});

test("supersession without an answer.cancelled still flags the partial", () => {
  const state = apply(
    fresh(),
    ev("question.detected", { question: "First?" }, 1),
    ev("answer.delta", { summary: "partial" }, 1),
    ev("question.detected", { question: "Second?" }, 2),
  );
  assert.equal(state.history[0]?.interrupted, true);
  assert.equal(state.history[0]?.streamingSummary, "partial");
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

test("firstTokenMs measures question.detected -> first delta, not completion", () => {
  const at = (type: ServerEventType, ts: string, data: Record<string, unknown> = {}) => ({
    ...ev(type, data, 1),
    ts,
  });
  const state = apply(
    fresh(),
    at("question.detected", "2026-01-01T00:00:00.000Z", {
      question: "What is caching?",
      classification: CLASSIFICATION,
    }),
    at("answer.delta", "2026-01-01T00:00:00.800Z", { summary: "Caching " }),
    at("answer.delta", "2026-01-01T00:00:04.000Z", { summary: "Caching stores data." }),
    at("answer.completed", "2026-01-01T00:00:05.000Z", {
      answer: { summary: "Caching stores data.", key_points: [], detailed_answer: "" },
      latency_ms: 5000,
    }),
  );
  const turn = state.history.at(-1) ?? state.current;
  // The perceived wait is the first delta, not the last one and not completion.
  assert.equal(turn?.firstTokenMs, 800);
  assert.equal(turn?.latencyMs, 5000);
});

// ------------------------------------------------- history (earlier this session)

const ANSWER = {
  summary: "Use a hash map.",
  key_points: ["one"],
  detailed_answer: "detail",
} as const;

function detected(turnId: number, question: string) {
  return ev("question.detected", { question, classification: CLASSIFICATION }, turnId);
}

test("a completed answer is preserved in history and is expandable", () => {
  const state = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    ev("answer.delta", { summary: "Use a key-value store." }, 1),
    ev("answer.completed", { answer: ANSWER, latency_ms: 900 }, 1),
    detected(2, "What is caching?"),
  );

  const older = visibleTurns(state).find((t) => t.turnId === 1);

  if (!older) {
    throw new Error("Expected turn 1 to exist in history");
  }

  assert.equal(historyStatus(older), "answered");
  assert.equal(hasExpandableAnswer(older), true);
  assert.equal(older.answer?.summary, "Use a hash map.");
});

test("an interrupted partial answer is kept and labelled interrupted", () => {
  const state = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    ev("answer.delta", { summary: "I would shard the key space" }, 1),
    ev(
      "answer.cancelled",
      {
        reason: "superseded",
        interrupted: true,
        partial_summary: "I would shard the key space",
      },
      1,
    ),
    detected(2, "What is caching?"),
  );

  const older = visibleTurns(state).find((t) => t.turnId === 1);

  if (!older) {
    throw new Error("Expected turn 1 to exist in history");
  }

  assert.equal(historyStatus(older), "interrupted");
  assert.equal(hasExpandableAnswer(older), true);
  assert.equal(older.answer, null);
  assert.equal(displaySummary(older), "I would shard the key space");
});

test("a turn cancelled before any content offers no answer to expand", () => {
  const state = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    ev("answer.cancelled", { reason: "superseded", interrupted: false }, 1),
    detected(2, "What is caching?"),
  );

  const older = visibleTurns(state).find((t) => t.turnId === 1);

  if (!older) {
    throw new Error("Expected turn 1 to exist in history");
  }

  assert.equal(historyStatus(older), "cancelled");

  // The row must not promise content that was never generated.
  assert.equal(hasExpandableAnswer(older), false);
  assert.equal(displaySummary(older), "");
});

test("a superseded turn with no answer.cancelled event is still classified correctly", () => {
  // The backend normally sends answer.cancelled, but question.detected alone
  // must retire the previous turn sanely too (reconnect, dropped frame).
  const withText = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    ev("answer.delta", { summary: "I would shard" }, 1),
    detected(2, "What is caching?"),
  );

  const bare = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    detected(2, "What is caching?"),
  );

  assert.equal(historyStatus(visibleTurns(withText)[1]!), "interrupted");
  assert.equal(historyStatus(visibleTurns(bare)[1]!), "cancelled");
});

test("the active turn is never reported as history-complete", () => {
  const state = apply(
    fresh(),
    detected(1, "What is caching?"),
    ev("answer.delta", { summary: "Caching stores" }, 1),
  );

  assert.equal(historyStatus(state.current!), "active");
  assert.equal(hasExpandableAnswer(state.current!), false);
});

test("a delta for an old turn cannot overwrite the active answer", () => {
  const state = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    ev("answer.delta", { summary: "old partial" }, 1),
    detected(2, "What is caching?"),
    ev("answer.delta", { summary: "new answer" }, 2),

    // Late frame from the superseded turn, arriving after the switch.
    ev("answer.delta", { summary: "old partial continued" }, 1),
  );

  assert.equal(state.current?.turnId, 2);
  assert.equal(displaySummary(state.current!), "new answer");
  assert.equal(displaySummary(visibleTurns(state)[1]!), "old partial");
});

test("rapid question changes keep history ordered newest-first without duplicates", () => {
  const state = apply(
    fresh(),
    detected(1, "First?"),
    ev("answer.completed", { answer: ANSWER, latency_ms: 1 }, 1),
    detected(2, "Second?"),
    ev("answer.completed", { answer: ANSWER, latency_ms: 2 }, 2),
    detected(3, "Third?"),
  );

  const turns = visibleTurns(state);

  assert.deepEqual(
    turns.map((t) => t.turnId),
    [3, 2, 1],
  );

  assert.equal(new Set(turns.map((t) => t.turnId)).size, 3);
});

test("expanding history data does not depend on which turn is active", () => {
  // hasExpandableAnswer/historyStatus read only the turn handed to them, so a
  // history row renders the same whether or not a new answer is streaming.
  const completed = apply(
    fresh(),
    detected(1, "Design a URL shortener?"),
    ev("answer.completed", { answer: ANSWER, latency_ms: 900 }, 1),
  );

  const settled = visibleTurns(completed)[0]!;

  const streaming = apply(
    completed,
    detected(2, "What is caching?"),
    ev("answer.delta", { summary: "Caching stores" }, 2),
  );

  const whileStreaming = visibleTurns(streaming).find(
    (t) => t.turnId === 1,
  )!;

  assert.equal(historyStatus(settled), historyStatus(whileStreaming));

  assert.equal(
    hasExpandableAnswer(settled),
    hasExpandableAnswer(whileStreaming),
  );

  assert.equal(
    settled.answer?.summary,
    whileStreaming.answer?.summary,
  );
});

test("live status reflects the real phase of the session", () => {
  const base = fresh();
  assert.equal(liveStatus(base), "idle");

  const connecting = sessionReducer(base, { kind: "connection", state: "connecting" });
  assert.equal(liveStatus(connecting), "connecting");

  const open = sessionReducer(connecting, { kind: "connection", state: "open" });
  const started = apply(open, ev("session.started", { session_id: "s1" }));
  // Connected but not capturing: nothing is being listened to yet.
  assert.equal(liveStatus(started), "idle");

  const listening = apply(started, ev("session.status", { audio: "ok", channels: ["LOOPBACK"] }));
  assert.equal(liveStatus(listening), "listening");

  const detected = apply(listening, ev("question.detected",
    { question: "What is caching?", classification: CLASSIFICATION }, 1));
  assert.equal(liveStatus(detected), "question_detected");

  // Streaming with no visible text yet is still "thinking" to the user.
  const empty = apply(detected, ev("answer.started", {}, 1));
  assert.equal(liveStatus(empty), "thinking");

  const answering = apply(empty, ev("answer.delta", { summary: "Caching stores" }, 1));
  assert.equal(liveStatus(answering), "answering");

  const failed = apply(answering, ev("answer.error", { message: "boom" }, 1));
  assert.equal(liveStatus(failed), "error");
});

test("a dropped socket reads as disconnected, not idle, once a session exists", () => {
  const open = sessionReducer(
    sessionReducer(fresh(), { kind: "connection", state: "open" }),
    { kind: "connection", state: "open" },
  );
  const started = apply(open, ev("session.started", { session_id: "s1" }));
  const dropped = sessionReducer(started, { kind: "connection", state: "reconnecting" });
  assert.equal(liveStatus(dropped), "connecting");

  const closed = sessionReducer(started, { kind: "connection", state: "closed" });
  assert.equal(liveStatus(closed), "disconnected");
});

// -------------------------------------------------------- code language hints

test("code blocks label the languages an interview actually uses", () => {
  assert.equal(detectLanguage("SELECT DISTINCT salary FROM employees;"), "SQL");
  assert.equal(detectLanguage("def two_sum(nums, target):\n    return None"), "Python");
  assert.equal(detectLanguage("const f = (a) => { return a; }"), "JavaScript");
  assert.equal(detectLanguage("public static void main(String[] a) {}"), "Java");
  assert.equal(detectLanguage("#include <vector>\nstd::vector<int> v;"), "C++");
  // Pseudocode is left unlabelled rather than guessed at.
  assert.equal(detectLanguage("for each item in list:\n  do something"), null);
});
