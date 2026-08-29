import { useState } from "react";

import { detectLanguage } from "./language.ts";

/**
 * Code and SQL presentation.
 *
 * No syntax-highlighting dependency: the smallest credible one is ~50KB gzipped
 * and would run a tokenizer on every streamed frame, on the machine that is
 * already running Whisper. A monospace block with a language label and a copy
 * button covers what a candidate actually needs mid-interview.
 */
export function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  const language = label ?? detectLanguage(code);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard can be denied; failing silently is better than an error
      // dialog over the answer the candidate is reading.
    }
  };

  return (
    <figure className="code-block">
      <figcaption className="code-head">
        <span className="code-lang">{language ?? "Code"}</span>
        <button className="code-copy" onClick={() => void copy()} aria-live="polite">
          {copied ? "Copied" : "Copy"}
        </button>
      </figcaption>
      <pre className="code">
        <code>{code}</code>
      </pre>
    </figure>
  );
}
