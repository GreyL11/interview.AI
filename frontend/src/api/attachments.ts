/**
 * Classifying pasted interviewer material, without ever changing it.
 *
 * The only job here is picking a `kind` for the wire so the backend can fence
 * the block appropriately. Two hard rules:
 *
 * 1. **The content is returned byte-for-byte.** No trimming of interior
 *    whitespace, no re-indenting, no collapsing blank lines. A schema's
 *    alignment and a snippet's indentation are content. Only a wholly-empty
 *    paste is refused, and that is a refusal, not an edit.
 * 2. **Guess conservatively.** Misclassifying prose as `sql` to win syntax
 *    highlighting would tell the model something false about what it is
 *    reading. Anything not clearly identifiable goes as `text`, which is the
 *    backend's generic kind.
 */

import {
  ATTACHMENT_MAX_CHARS,
  type AttachmentKind,
  type ClientMessage,
} from "./contracts.ts";

/** SQL only when a statement *opens* the paste -- "select" appears in plenty
 * of ordinary sentences, so a bare keyword anywhere is not evidence. */
const SQL_OPENER =
  /^\s*(?:with|select|insert\s+into|update|delete\s+from|create\s+(?:table|view|index)|alter\s+table|drop\s+table|explain|merge)\b/i;

/** Structural code markers. Again anchored to a line start, so prose that
 * mentions "def" or "import" in passing does not qualify. */
const CODE_LINE =
  /^\s*(?:def |class |import |from \w+ import |function |const |let |var |public |private |package |#include|=>|async def )/m;

/** A fenced block the interviewer pasted from a chat client. */
const FENCED = /^\s*```/;

/**
 * An aligned column layout: several lines sharing a separator. Covers
 * `a | b | c` pipe tables, and `name ---` schema dumps, which is how table
 * definitions usually arrive from a chat window.
 */
function looksTabular(text: string): boolean {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  if (lines.length < 2) return false;

  const piped = lines.filter((line) => line.includes("|")).length;
  if (piped >= 2 && piped >= lines.length / 2) return true;

  const tabbed = lines.filter((line) => line.includes("\t")).length;
  if (tabbed >= 2 && tabbed >= lines.length / 2) return true;

  // "customers\n---------\ncustomer_id" -- a divider rule under a heading.
  return lines.some((line) => /^\s*[-=]{3,}\s*$/.test(line));
}

/**
 * Best-effort kind for a paste. Order matters: an explicit SQL statement wins
 * over a table-ish shape, because a formatted query is often column-aligned
 * too.
 */
export function detectKind(text: string): AttachmentKind {
  if (SQL_OPENER.test(text)) return "sql";
  if (CODE_LINE.test(text) || FENCED.test(text)) return "code";
  if (looksTabular(text)) return "table";
  return "text";
}

export type PasteRefusal = { ok: false; reason: "empty" | "too_large"; message: string };
export type PasteAccepted = { ok: true; message: ClientMessage };
export type PasteResult = PasteAccepted | PasteRefusal;

/**
 * Turn a raw paste into the outbound frame, or refuse it.
 *
 * Refusing rather than truncating is deliberate and matches the backend: a
 * half table or half query looks complete to the model and is worse than
 * nothing. The size check here is a courtesy so the user hears about it
 * before a large string crosses the socket -- the backend checks again and
 * remains authoritative.
 */
export function prepareTextPaste(raw: string, name = ""): PasteResult {
  if (raw.trim() === "") {
    return { ok: false, reason: "empty", message: "Nothing to attach." };
  }
  if (raw.length > ATTACHMENT_MAX_CHARS) {
    return {
      ok: false,
      reason: "too_large",
      message:
        `That paste is ${raw.length.toLocaleString()} characters, over the ` +
        `${ATTACHMENT_MAX_CHARS.toLocaleString()} limit. Paste the relevant ` +
        `part — it is not shortened automatically, because a partial table ` +
        `or query is worse than none.`,
    };
  }
  return {
    ok: true,
    message: {
      type: "context.attach",
      // `raw` unmodified: this is the whole point of the module.
      data: { kind: detectKind(raw), content: raw, name },
    },
  };
}

/**
 * Turn pasted image bytes into the outbound frame.
 *
 * The backend owns OCR; this only encodes. Split out so the paste handler can
 * stay synchronous while the (async) clipboard read happens at the call site.
 */
export function prepareImagePaste(bytes: Uint8Array, name = ""): PasteResult {
  if (bytes.byteLength === 0) {
    return { ok: false, reason: "empty", message: "That image was empty." };
  }
  // base64 inflates by ~4/3, and the backend measures the *decoded* OCR text
  // rather than the payload, so the char cap does not apply here. The item
  // cap and the backend's own decode failure path are the guards.
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return {
    ok: true,
    message: {
      type: "context.attach",
      data: { kind: "image", content: "", name, image_base64: btoa(binary) },
    },
  };
}
