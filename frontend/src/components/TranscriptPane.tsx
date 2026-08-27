import { useEffect, useRef } from "react";

import type { TranscriptSource } from "../api/contracts.ts";
import type { SessionState } from "../state/sessionReducer.ts";
import { Empty } from "./Common.tsx";

const SOURCE_LABEL: Record<TranscriptSource, string> = {
  LOOPBACK: "Interviewer",
  MIC: "You",
  MANUAL: "Typed",
};

export function TranscriptPane({ state }: { state: SessionState }) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const partials = Object.entries(state.partials) as [TranscriptSource, string][];

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [state.transcript.length, partials.length]);

  if (state.transcript.length === 0 && partials.length === 0) {
    return (
      <Empty
        title="No speech yet"
        hint="Start audio, or type a question below to practise without a microphone."
      />
    );
  }

  return (
    <div className="transcript">
      {state.transcript.map((line) => (
        <p key={line.id} className={`line line-${line.source.toLowerCase()}`}>
          <span className="line-source">{SOURCE_LABEL[line.source]}</span>
          <span className="line-text">{line.text}</span>
        </p>
      ))}
      {partials.map(([source, text]) => (
        // Partials render greyed: they are display-only and never reach the LLM.
        <p key={`partial-${source}`} className="line line-partial">
          <span className="line-source">{SOURCE_LABEL[source]}</span>
          <span className="line-text">{text}…</span>
        </p>
      ))}
      <div ref={endRef} />
    </div>
  );
}
