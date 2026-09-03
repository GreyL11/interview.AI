/**
 * Paste interviewer-provided material into the current interview context.
 *
 * Deliberately its own box, next to but separate from the "ask a question"
 * input. Pasting a schema is not asking a question, and the two must not be
 * confusable: material attaches and waits, a question is answered. Keeping
 * them visually distinct is what stops a paste from reading as a prompt.
 *
 * The content never lands in component state. A paste is read from the event,
 * handed to the socket, and dropped -- so pasted material is not held in the
 * React tree, not re-rendered, and not persisted anywhere.
 */

import { useState } from "react";

import type { AttachmentView } from "../state/sessionReducer.ts";
import { useSession } from "../state/store.tsx";

const KIND_LABEL: Record<string, string> = {
  sql: "SQL",
  code: "Code",
  table: "Table",
  text: "Text",
  image: "Image",
};

function describe(attachment: AttachmentView): string {
  const label = KIND_LABEL[attachment.kind] ?? "Material";
  const source = attachment.fromImage ? " (from image)" : "";
  return `${label}${source} · ${attachment.chars.toLocaleString()} chars`;
}

export function AttachedMaterial({ attachments }: { attachments: AttachmentView[] }) {
  if (attachments.length === 0) return null;
  return (
    <ul className="attachments" aria-label="Attached interview material">
      {attachments.map((attachment) => (
        <li key={attachment.id} className="attachment">
          {/* Metadata only. The full paste is deliberately not re-displayed:
              it can be thousands of characters and the interviewer already
              has it in their own window. */}
          📎 {attachment.name !== "" ? `${attachment.name} · ` : ""}
          {describe(attachment)}
        </li>
      ))}
    </ul>
  );
}

export function ContextAttach() {
  const { state, attachPaste, attachImage } = useSession();
  const [refusal, setRefusal] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(0);

  /** Read a pasted image out of the clipboard event.
   *
   * Uses the paste event's own DataTransfer rather than reading the clipboard
   * programmatically: the event already carries the bytes, so this needs no
   * clipboard permission and no Tauri clipboard plugin. Returns false when
   * the paste held no image, so the caller can fall back to text. */
  const takeImage = (clipboard: DataTransfer | null): boolean => {
    const file = Array.from(clipboard?.items ?? [])
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .find((candidate): candidate is File => candidate !== null);
    if (file === undefined) return false;

    void file.arrayBuffer().then((buffer) => {
      const problem = attachImage(new Uint8Array(buffer), file.name);
      setRefusal(problem);
      if (problem === null) setAccepted((count) => count + 1);
    });
    return true;
  };

  const onPaste = (pasteEvent: React.ClipboardEvent<HTMLTextAreaElement>) => {
    // Always intercept: letting the default run would leave the content in
    // the textarea, where it would look like something still to be sent.
    pasteEvent.preventDefault();
    if (takeImage(pasteEvent.clipboardData)) return;

    // getData returns the paste exactly as it was copied. Nothing is trimmed
    // or re-indented here -- interior whitespace is content.
    const text = pasteEvent.clipboardData.getData("text");
    const problem = attachPaste(text);
    setRefusal(problem);
    if (problem === null) setAccepted((count) => count + 1);
  };

  const rejection = state.lastAttachmentRejection;

  return (
    <div className="context-attach">
      <textarea
        className="context-paste"
        rows={2}
        value=""
        onPaste={onPaste}
        onChange={() => {
          /* Paste-only by design. Typing here would suggest this box sends
             text on submit, which it does not -- the ask box above does. */
        }}
        placeholder="Paste a schema, query, log or screenshot to attach to this question…"
        aria-label="Paste interviewer material"
      />
      <AttachedMaterial attachments={state.attachments} />
      {/* A refusal is normal and actionable, so it reads as guidance rather
          than as an application error. Server refusals win over local ones,
          since the backend is authoritative. */}
      {rejection !== null && <p className="warning">{rejection.message}</p>}
      {rejection === null && refusal !== null && <p className="warning">{refusal}</p>}
      {rejection === null && refusal === null && accepted > 0 && (
        <p className="muted">Attached to the current interview context.</p>
      )}
    </div>
  );
}
