import {
  displaySummary,
  hasExpandableAnswer,
  historyStatus,
  type TurnView,
} from "../state/sessionReducer.ts";
import { CodeBlock } from "./CodeBlock.tsx";
import { Empty, Pill, Spinner } from "./Common.tsx";

const REJECTION_TEXT: Record<string, string> = {
  not_a_question: "not a question",
  too_short: "too short",
  low_confidence: "unclear",
};

export function DetectedQuestion({ turn }: { turn: TurnView }) {
  return (
    <div className="detected">
      <div className="detected-head">
        <span className="detected-label">Detected question</span>
        {turn.classification !== null && (
          <span className="chips">
            <Pill label={turn.classification.category} tone="ok" />
            {turn.classification.domain !== "GENERAL" && (
              <Pill label={turn.classification.domain} tone="idle" />
            )}
            <Pill
              label={`${Math.round(turn.classification.confidence * 100)}%`}
              tone={turn.classification.confidence >= 0.8 ? "ok" : "warn"}
              title="Classifier confidence"
            />
          </span>
        )}
      </div>
      <p className="detected-question">{turn.question}</p>
    </div>
  );
}

export function RejectedNote({ text, reason }: { text: string; reason: string | null }) {
  // Shown rather than swallowed, so the debounce behaviour is explainable
  // instead of feeling like the app randomly ignored someone.
  return (
    <p className="rejected">
      Ignored “{text}” — {REJECTION_TEXT[reason ?? ""] ?? "not a question"}.
    </p>
  );
}

export function AnswerPanel({ turn }: { turn: TurnView | null }) {
  if (turn === null) {
    return <Empty title="No question yet" hint="Answers appear here as they are generated." />;
  }

  const summary = displaySummary(turn);
  const answer = turn.answer;

  return (
    <div className="answer">
      {turn.phase === "retrieving" && <Spinner label="Searching your documents…" />}
      {turn.phase === "streaming" && summary === "" && <Spinner label="Thinking…" />}

      {summary !== "" && (
        <p className={`answer-summary${turn.answer === null ? " streaming" : ""}`}>{summary}</p>
      )}

      {turn.phase === "failed" && <p className="error-note">{turn.errorMessage}</p>}
      {turn.phase === "cancelled" && (
        <p className="muted">
          {turn.interrupted
            ? "Interrupted — partial answer, incomplete."
            : turn.cancelReason === "superseded"
              ? "Superseded by a newer question."
              : "Cancelled."}
        </p>
      )}

      {answer !== null && (
        <>
          {answer.warnings.length > 0 && (
            <div className="warnings">
              {answer.warnings.map((warning) => (
                <p key={warning} className="warning">
                  {warning}
                </p>
              ))}
            </div>
          )}

          {answer.key_points.length > 0 && (
            <ul className="key-points">
              {answer.key_points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          )}

          {answer.approach !== null && answer.approach.length > 0 && (
            <ol className="approach">
              {answer.approach.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          )}

          {answer.sections !== null && answer.sections.length > 0 && (
            <div className="sections">
              {answer.sections.map((section) => (
                <div key={section.heading} className="section">
                  <h4>{section.heading}</h4>
                  <p>{section.content}</p>
                </div>
              ))}
            </div>
          )}

          {answer.code !== null && answer.code !== "" && (
            <CodeBlock code={answer.code} />
          )}

          {answer.complexity !== null && (
            <p className="complexity">
              Time {answer.complexity.time} · Space {answer.complexity.space}
            </p>
          )}

          {answer.edge_cases !== null && answer.edge_cases.length > 0 && (
            <div className="edge-cases">
              <h4>Edge cases</h4>
              <ul>
                {answer.edge_cases.map((edge) => (
                  <li key={edge}>{edge}</li>
                ))}
              </ul>
            </div>
          )}

          {answer.detailed_answer !== "" && (
            <details className="detail">
              <summary>Full answer</summary>
              <p>{answer.detailed_answer}</p>
            </details>
          )}

          <footer className="answer-foot">
            <Pill
              label={turn.contextFound ? "grounded in your documents" : "general answer"}
              tone={turn.contextFound ? "ok" : "idle"}
              title={
                turn.contextFound
                  ? "Backed by retrieved personal context"
                  : "No personal context was found; treat first-person claims as illustrative"
              }
            />
            {/* Lead with time-to-first-answer: a bare total reads as "you
                waited this long", which is wrong once text streams. */}
            {turn.firstTokenMs !== null && (
              <span className="muted" title="Time until the answer started appearing">
                {(turn.firstTokenMs / 1000).toFixed(1)}s to first words
              </span>
            )}
            {turn.latencyMs !== null && (
              <span className="muted" title="Time until the full answer finished generating">
                {(turn.latencyMs / 1000).toFixed(1)}s total
              </span>
            )}
            {turn.hits.map((hit) => (
              <span key={hit.chunk_id} className="hit" title={`score ${hit.score.toFixed(3)}`}>
                {hit.title || hit.document_id.slice(0, 8)}
              </span>
            ))}
          </footer>
        </>
      )}
    </div>
  );
}

const HISTORY_TONE = {
  answered: "ok",
  interrupted: "warn",
  cancelled: "idle",
  failed: "warn",
  active: "ok",
} as const;

/**
 * One row in "Earlier this session".
 *
 * Uses a native <details> rather than lifting open/closed into the reducer:
 * expanding old answers is per-row view state that must never touch the live
 * turn, and the element keeps its own state through re-renders while a new
 * answer streams. Rows with nothing to show render as plain text, so a turn
 * cancelled before any content never offers an expander that opens onto
 * nothing.
 */
export function HistoryTurn({ turn }: { turn: TurnView }) {
  const status = historyStatus(turn);
  const label = (
    <>
      <span className="turn-q">{turn.question}</span>
      <Pill label={status} tone={HISTORY_TONE[status]} />
    </>
  );

  if (!hasExpandableAnswer(turn)) {
    return <div className="turn-row">{label}</div>;
  }

  return (
    <details className="turn-row">
      <summary>{label}</summary>
      {/* AnswerPanel already prints the "Interrupted — partial answer" note
          for a cancelled turn, so this row does not repeat it. */}
      <AnswerPanel turn={turn} />
    </details>
  );
}