/**
 * Interviewer-provided context: outbound payload and reducer state.
 *
 *   node --test src/api/attachments.test.ts
 *
 * Asserts the *actual* frame that would go on the socket, not an
 * intermediate, because the invariant that matters is that the bytes the
 * interviewer pasted are the bytes the backend receives.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import type { ClientMessage, ServerEvent, ServerEventType } from "./contracts.ts";
import { ATTACHMENT_MAX_CHARS } from "./contracts.ts";
import { detectKind, prepareImagePaste, prepareTextPaste } from "./attachments.ts";
import {
  initialSessionState,
  sessionReducer,
  type SessionState,
} from "../state/sessionReducer.ts";

const SCHEMA = `customers
---------
customer_id
name

orders
---------
order_id
customer_id
order_date`;

const SQL = `SELECT c.name
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id
      AND o.order_date >= CURRENT_DATE - INTERVAL '90 days'
WHERE o.order_id IS NULL;`;

const PYTHON = `def total(rows):
    return sum(
        r["amount"] for r in rows
    )`;

let seq = 0;

function ev(
  type: ServerEventType,
  data: Record<string, unknown> = {},
  overrideSeq?: number,
): ServerEvent {
  seq += 1;
  return {
    type,
    seq: overrideSeq ?? seq,
    ts: "2026-01-01T00:00:00Z",
    turn_id: null,
    data,
  };
}

function apply(state: SessionState, ...events: ServerEvent[]): SessionState {
  return events.reduce((acc, e) => sessionReducer(acc, { kind: "event", event: e }), state);
}

function sent(raw: string): ClientMessage {
  const result = prepareTextPaste(raw);
  assert.equal(result.ok, true, "expected the paste to be accepted");
  if (!result.ok) throw new Error("unreachable");
  return result.message;
}

// -------------------------------------------------------- outbound payload

test("a plain text paste produces exactly one context.attach frame", () => {
  const message = sent("some interviewer notes");
  assert.equal(message.type, "context.attach");
  // One paste, one frame -- prepare returns a single message, never a batch.
  assert.equal(Array.isArray(message), false);
});

test("the outbound frame matches the backend contract exactly", () => {
  const message = sent(SQL);
  assert.deepEqual(Object.keys(message).sort(), ["data", "type"]);
  if (message.type !== "context.attach") throw new Error("wrong type");
  // Exactly the keys app/api/ws.py reads: kind, content, name (image_base64
  // only for images).
  assert.deepEqual(Object.keys(message.data).sort(), ["content", "kind", "name"]);
});

test("multi-line content is preserved exactly, including blank lines", () => {
  const message = sent(SCHEMA);
  if (message.type !== "context.attach") throw new Error("wrong type");
  assert.equal(message.data.content, SCHEMA);
  assert.equal(message.data.content.split("\n").length, SCHEMA.split("\n").length);
  assert.ok(message.data.content.includes("\n\n"), "blank line collapsed");
});

test("SQL keeps its exact whitespace and indentation", () => {
  const message = sent(SQL);
  if (message.type !== "context.attach") throw new Error("wrong type");
  assert.equal(message.data.content, SQL);
  assert.ok(message.data.content.includes("       ON o.customer_id"), "indent lost");
});

test("a table/schema paste keeps its exact layout", () => {
  const message = sent(SCHEMA);
  if (message.type !== "context.attach") throw new Error("wrong type");
  assert.equal(message.data.content, SCHEMA);
  assert.ok(message.data.content.includes("---------"), "divider rule lost");
});

test("python indentation survives byte-for-byte", () => {
  const message = sent(PYTHON);
  if (message.type !== "context.attach") throw new Error("wrong type");
  assert.equal(message.data.content, PYTHON);
  assert.ok(message.data.content.includes('        r["amount"]'), "indent lost");
});

test("leading and trailing whitespace is not stripped client-side", () => {
  const padded = "\n\n   indented start\n   still indented\n\n";
  const message = sent(padded);
  if (message.type !== "context.attach") throw new Error("wrong type");
  assert.equal(message.data.content, padded, "frontend must not normalise");
});

// -------------------------------------------------------------------- kinds

test("kind detection is conservative and never fabricates a code kind", () => {
  assert.equal(detectKind(SQL), "sql");
  assert.equal(detectKind(PYTHON), "code");
  assert.equal(detectKind(SCHEMA), "table");
  assert.equal(detectKind("a | b | c\n1 | 2 | 3"), "table");
  // Ordinary prose that merely mentions SQL words stays generic.
  assert.equal(detectKind("I would select the right index for this."), "text");
  assert.equal(detectKind("We need to update the customer record."), "text");
  assert.equal(detectKind("just some notes about the system"), "text");
});

test("every detected kind is one the backend accepts", () => {
  const allowed = ["text", "code", "sql", "table", "image"];
  for (const sample of [SQL, PYTHON, SCHEMA, "plain", "a|b\nc|d"]) {
    assert.ok(allowed.includes(detectKind(sample)), `invented kind for ${sample}`);
  }
});

// ------------------------------------------------------------- refusals

test("an empty paste sends nothing", () => {
  for (const blank of ["", "   ", "\n\n", "\t"]) {
    const result = prepareTextPaste(blank);
    assert.equal(result.ok, false);
    if (result.ok) throw new Error("unreachable");
    assert.equal(result.reason, "empty");
  }
});

test("oversized content is refused, never truncated", () => {
  const huge = "x".repeat(ATTACHMENT_MAX_CHARS + 1);
  const result = prepareTextPaste(huge);
  assert.equal(result.ok, false);
  if (result.ok) throw new Error("unreachable");
  assert.equal(result.reason, "too_large");
  assert.ok(result.message.includes("not shortened"), "must explain no truncation");
});

test("content exactly at the limit is still sent whole", () => {
  const exact = "y".repeat(ATTACHMENT_MAX_CHARS);
  const message = sent(exact);
  if (message.type !== "context.attach") throw new Error("wrong type");
  assert.equal(message.data.content.length, ATTACHMENT_MAX_CHARS);
});

// ---------------------------------------------------------------- images

test("an image paste carries base64 on the contract's own field", () => {
  const bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const result = prepareImagePaste(bytes, "screenshot.png");
  assert.equal(result.ok, true);
  if (!result.ok) throw new Error("unreachable");
  if (result.message.type !== "context.attach") throw new Error("wrong type");

  assert.equal(result.message.data.kind, "image");
  assert.equal(result.message.data.name, "screenshot.png");
  assert.equal(result.message.data.image_base64, "iVBORw0KGgo=");
  // No fabricated transcript: the frontend does not guess at the image's text,
  // because OCR belongs to the backend.
  assert.equal(result.message.data.content, "");
});

test("an empty image is refused before encoding", () => {
  const result = prepareImagePaste(new Uint8Array([]));
  assert.equal(result.ok, false);
  if (result.ok) throw new Error("unreachable");
  assert.equal(result.reason, "empty");
});

// ------------------------------------------------------- reducer: accepted

test("context.attached records the acknowledgement as metadata only", () => {
  const state = apply(
    initialSessionState,
    ev("context.attached", { kind: "sql", name: "q.sql", chars: 120, from_image: false }),
  );

  assert.equal(state.attachments.length, 1);
  const attachment = state.attachments[0];
  assert.ok(attachment !== undefined);
  assert.equal(attachment.kind, "sql");
  assert.equal(attachment.name, "q.sql");
  assert.equal(attachment.chars, 120);
  assert.equal(attachment.fromImage, false);
  // No content field exists to leak.
  assert.equal("content" in attachment, false);
  assert.equal(JSON.stringify(state.attachments).includes("SELECT"), false);
});

test("multiple attachments accumulate in arrival order", () => {
  const state = apply(
    initialSessionState,
    ev("context.attached", { kind: "table", chars: 10 }),
    ev("context.attached", { kind: "sql", chars: 20 }),
    ev("context.attached", { kind: "text", chars: 30 }),
  );

  assert.deepEqual(state.attachments.map((a) => a.kind), ["table", "sql", "text"]);
  assert.deepEqual(state.attachments.map((a) => a.chars), [10, 20, 30]);
});

test("a replayed acknowledgement is idempotent, not a duplicate chip", () => {
  const first = ev("context.attached", { kind: "sql", chars: 20 }, 41);
  const state = apply(initialSessionState, first);
  // Reconnect replays the same seq. The reducer's own seq watermark already
  // guards this, so feed it directly to prove identity is stable too.
  const replayed = sessionReducer(
    { ...state, lastSeq: 0 },
    { kind: "event", event: first },
  );
  assert.equal(replayed.attachments.length, 1);
});

test("an OCR-derived attachment is marked as such", () => {
  const state = apply(
    initialSessionState,
    ev("context.attached", { kind: "image", chars: 44, from_image: true }),
  );
  assert.equal(state.attachments[0]?.fromImage, true);
});

// ------------------------------------------------------- reducer: rejected

test("context.rejected is surfaced as guidance, not an application error", () => {
  const state = apply(
    initialSessionState,
    ev("context.rejected", {
      kind: "text",
      reason: "too_large",
      message: "That attachment is 30,000 characters, over the 20,000 limit.",
    }),
  );

  assert.equal(state.lastAttachmentRejection?.reason, "too_large");
  assert.ok(state.lastAttachmentRejection?.message.includes("20,000"));
  // Must not light up the generic failure banner.
  assert.equal(state.error, null);
  assert.equal(state.attachments.length, 0);
});

test("every backend refusal reason renders without crashing", () => {
  for (const reason of ["empty", "too_large", "unreadable_image"]) {
    const state = apply(
      initialSessionState,
      ev("context.rejected", { kind: "image", reason, message: `nope: ${reason}` }),
    );
    assert.equal(state.lastAttachmentRejection?.reason, reason);
    assert.equal(state.error, null);
  }
});

test("a later acceptance clears the previous refusal", () => {
  const state = apply(
    initialSessionState,
    ev("context.rejected", { kind: "text", reason: "empty", message: "nothing there" }),
    ev("context.attached", { kind: "table", chars: 12 }),
  );
  assert.equal(state.lastAttachmentRejection, null);
  assert.equal(state.attachments.length, 1);
});

// ------------------------------------------------- separation of concerns

test("attachment acknowledgements never enter transcript state", () => {
  const state = apply(
    initialSessionState,
    ev("context.attached", { kind: "table", chars: 99 }),
    ev("context.rejected", { kind: "text", reason: "empty", message: "no" }),
  );

  assert.equal(state.transcript.length, 0, "attachment leaked into transcript");
  assert.deepEqual(state.partials, {});
  assert.equal(state.current, null, "attachment created a question turn");
  assert.equal(state.history.length, 0);
  assert.equal(state.lastRejected, null, "attachment refusal read as a rejected question");
});

test("a paste does not produce a question.manual frame", () => {
  const message = sent(SCHEMA);
  assert.equal(message.type, "context.attach");
  assert.notEqual(message.type as string, "question.manual");
});

test("an attachment alone opens no turn, so nothing can be answered", () => {
  const state = apply(
    initialSessionState,
    ev("context.attached", { kind: "sql", chars: 50 }),
  );
  // No turn means the coaching panel has nothing to render and no answer can
  // be in flight -- the backend's "paste alone asks nothing" rule, observed
  // from the client side.
  assert.equal(state.current, null);
  assert.equal(state.history.length, 0);
});

test("a question turn and an attachment coexist without interfering", () => {
  const state = apply(
    initialSessionState,
    ev("context.attached", { kind: "table", chars: 42 }),
    {
      ...ev("question.detected", { question: "Who has no orders?" }),
      turn_id: 7,
    },
  );

  assert.equal(state.attachments.length, 1);
  assert.equal(state.current?.turnId, 7);
  assert.equal(state.current?.question, "Who has no orders?");
  // The question text is the spoken words only -- no pasted material spliced in.
  assert.equal(state.current?.question.includes("42"), false);
});

test("attachments survive an answer completing, for the follow-up to use", () => {
  let state = apply(
    initialSessionState,
    ev("context.attached", { kind: "table", chars: 42 }),
    { ...ev("question.detected", { question: "Who has no orders?" }), turn_id: 7 },
    { ...ev("answer.started", {}), turn_id: 7 },
  );
  state = apply(state, {
    ...ev("answer.completed", {
      answer: { summary: "Use a LEFT JOIN.", key_points: [], detailed_answer: "" },
      latency_ms: 10,
    }),
    turn_id: 7,
  });

  // The backend carries material forward to a follow-up, so the UI must not
  // clear its indicator the moment the first answer lands.
  assert.equal(state.attachments.length, 1);
});
