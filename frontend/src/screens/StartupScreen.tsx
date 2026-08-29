import { openLogsFolder, retryBackend, type StartupState } from "../api/desktop.ts";

/**
 * What the user sees while the shell brings the local engine up, and what they
 * see if it never comes up.
 *
 * Deliberately no progress bar: the shell reports named stages because it
 * genuinely cannot know how long model loading will take, and a fabricated
 * percentage that stalls at 80% is worse than an honest sentence.
 */
export function StartupScreen({ state }: { state: StartupState }) {
  const failed = state.status === "failed";

  return (
    <div className="startup" role="status" aria-live="polite">
      <div className="startup-inner">
        <p className="startup-brand">Interview Coach</p>

        {failed ? (
          <>
            <h1 className="startup-title">
              Something prevented Interview Coach from starting.
            </h1>
            {state.detail !== undefined && state.detail !== "" && (
              <pre className="startup-detail">{state.detail}</pre>
            )}
            <div className="startup-actions">
              <button className="primary" onClick={() => void retryBackend()}>
                Retry
              </button>
              <button onClick={() => void openLogsFolder()}>Open Logs Folder</button>
            </div>
          </>
        ) : (
          <>
            {/* The dot is decorative; the sentence carries the meaning, so
                status is never communicated by colour or motion alone. */}
            <span className="startup-pulse" aria-hidden="true" />
            <h1 className="startup-title">{state.label}</h1>
            <p className="startup-hint">
              Everything except AI reasoning runs on this machine, so the first
              launch can take a moment.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
