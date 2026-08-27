import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client.ts";
import type { SessionDetail, SessionListItem } from "../api/contracts.ts";
import { Empty, ErrorNote, Panel, Pill } from "../components/Common.tsx";

export function HistoryScreen() {
  const [sessions, setSessions] = useState<SessionListItem[]>([]);
  const [selected, setSelected] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const open = useCallback(async (id: string) => {
    try {
      setSelected(await api.getSession(id));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  const remove = useCallback(
    async (id: string) => {
      await api.deleteSession(id);
      if (selected?.session.session_id === id) setSelected(null);
      await refresh();
    },
    [refresh, selected],
  );

  return (
    <div className="history">
      <Panel title={`Sessions (${sessions.length})`}>
        {sessions.length === 0 ? (
          <Empty title="No sessions yet" hint="Finished practice sessions are listed here." />
        ) : (
          <ul className="session-list">
            {sessions.map((session) => (
              <li
                key={session.session_id}
                className={selected?.session.session_id === session.session_id ? "active" : ""}
              >
                <button className="link" onClick={() => void open(session.session_id)}>
                  <span>{session.title || "Untitled session"}</span>
                  <span className="muted">
                    {new Date(session.started_at).toLocaleString()} · {session.turn_count}{" "}
                    {session.turn_count === 1 ? "question" : "questions"}
                  </span>
                </button>
                <button className="danger" onClick={() => void remove(session.session_id)}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
        <ErrorNote message={error} />
      </Panel>

      {selected !== null && (
        <Panel title={selected.session.title || "Session"}>
          {selected.summary !== null && selected.summary.summary !== "" && (
            <div className="summary">
              <h4>Summary</h4>
              <p>{selected.summary.summary}</p>
              {selected.summary.topics.length > 0 && (
                <div className="chips">
                  {selected.summary.topics.map((topic) => (
                    <Pill key={topic} label={topic} tone="idle" />
                  ))}
                </div>
              )}
            </div>
          )}

          <h4>Questions</h4>
          {selected.turns.length === 0 ? (
            <p className="muted">No questions were asked.</p>
          ) : (
            <ol className="turns">
              {selected.turns.map((turn) => (
                <li key={turn.turn_id ?? turn.seq}>
                  <p className="turn-q">{turn.question}</p>
                  <div className="chips">
                    <Pill label={turn.category || "UNKNOWN"} tone="idle" />
                    <Pill
                      label={turn.status}
                      tone={
                        turn.status === "ANSWERED"
                          ? "ok"
                          : turn.status === "FAILED"
                            ? "bad"
                            : "warn"
                      }
                    />
                    <Pill
                      label={turn.context_found ? "grounded" : "general"}
                      tone={turn.context_found ? "ok" : "idle"}
                      title={
                        turn.context_found
                          ? "Backed by your documents"
                          : "No personal context was retrieved for this answer"
                      }
                    />
                    {turn.latency_ms !== null && (
                      <span className="muted">{(turn.latency_ms / 1000).toFixed(1)}s</span>
                    )}
                  </div>
                  {turn.answer !== null && <p className="turn-a">{turn.answer.summary}</p>}
                </li>
              ))}
            </ol>
          )}

          <details className="detail">
            <summary>Full transcript ({selected.transcript.length} lines)</summary>
            <div className="transcript">
              {selected.transcript.map((line) => (
                <p key={line.id ?? `${line.source}-${line.text}`} className="line">
                  <span className="line-source">
                    {line.source === "LOOPBACK" ? "Interviewer" : line.source === "MIC" ? "You" : "Typed"}
                  </span>
                  <span className="line-text">{line.text}</span>
                </p>
              ))}
            </div>
          </details>
        </Panel>
      )}
    </div>
  );
}
