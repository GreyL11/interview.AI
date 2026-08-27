import { useState } from "react";

import { AnswerPanel, DetectedQuestion, RejectedNote } from "../components/AnswerPanel.tsx";
import { ErrorNote, Panel } from "../components/Common.tsx";
import { StatusPills } from "../components/StatusPills.tsx";
import { TranscriptPane } from "../components/TranscriptPane.tsx";
import { isAnswering, visibleTurns } from "../state/sessionReducer.ts";
import { useSession } from "../state/store.tsx";

export function PracticeScreen() {
  const { state, starting, startError, start, stop, ask, cancel, toggleAudio } = useSession();
  const [typed, setTyped] = useState("");

  const live = state.connection === "open" || state.connection === "reconnecting";
  const turns = visibleTurns(state);
  const current = turns[0] ?? null;

  const submit = (formEvent: React.FormEvent) => {
    formEvent.preventDefault();
    const text = typed.trim();
    if (text === "") return;
    ask(text);
    setTyped("");
  };

  return (
    <div className="practice">
      <header className="practice-head">
        <div className="controls">
          {!live ? (
            <button className="primary" onClick={() => void start()} disabled={starting}>
              {starting ? "Starting…" : "Start session"}
            </button>
          ) : (
            <>
              <button className="danger" onClick={stop}>
                Stop session
              </button>
              <button onClick={toggleAudio}>
                {state.audio === "ok" ? "Stop audio" : "Start audio"}
              </button>
              {isAnswering(state) && (
                <button onClick={cancel} title="Discard the answer in progress">
                  Cancel answer
                </button>
              )}
            </>
          )}
        </div>
        <StatusPills state={state} />
      </header>

      <ErrorNote message={startError} />
      <ErrorNote message={state.error} />
      {state.audio === "error" && (
        <p className="warning">
          {state.audioMessage} — you can still practise by typing questions below.
        </p>
      )}

      <div className="practice-grid">
        <Panel title="Live transcript">
          <TranscriptPane state={state} />
          {live && (
            <form className="ask" onSubmit={submit}>
              <input
                value={typed}
                onChange={(changeEvent) => setTyped(changeEvent.target.value)}
                placeholder="Type a question to practise without audio…"
                aria-label="Type a question"
              />
              <button type="submit" disabled={typed.trim() === ""}>
                Ask
              </button>
            </form>
          )}
        </Panel>

        <Panel title="Coaching">
          {state.lastRejected !== null && current === null && (
            <RejectedNote
              text={state.lastRejected.text}
              reason={state.lastRejected.reason}
            />
          )}
          {current !== null && <DetectedQuestion turn={current} />}
          <AnswerPanel turn={current} />
        </Panel>
      </div>

      {turns.length > 1 && (
        <Panel title={`Earlier this session (${turns.length - 1})`}>
          <ul className="turn-list">
            {turns.slice(1).map((turn) => (
              <li key={turn.turnId}>
                <span className="turn-q">{turn.question}</span>
                <span className={`turn-phase phase-${turn.phase}`}>{turn.phase}</span>
              </li>
            ))}
          </ul>
        </Panel>
      )}
    </div>
  );
}
