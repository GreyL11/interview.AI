import { liveStatus, type LiveStatus, type SessionState } from "../state/sessionReducer.ts";

/**
 * The single "what is happening right now" indicator, plus per-channel audio.
 *
 * Every state carries a word as well as a colour and a dot shape, so nothing
 * here is communicated by colour alone.
 */
const STATUS_TEXT: Record<LiveStatus, string> = {
  idle: "Ready",
  listening: "Listening",
  question_detected: "Question detected",
  thinking: "Thinking",
  answering: "Answering",
  connecting: "Reconnecting to local engine…",
  disconnected: "Disconnected",
  error: "Answer failed",
};

const STATUS_TONE: Record<LiveStatus, string> = {
  idle: "idle",
  listening: "live",
  question_detected: "ok",
  thinking: "busy",
  answering: "busy",
  connecting: "warn",
  disconnected: "warn",
  error: "bad",
};

export function StatusPills({ state }: { state: SessionState }) {
  const status = liveStatus(state);
  return (
    <div className="pills" role="status" aria-live="polite">
      <span className={`status status-${STATUS_TONE[status]}`}>
        <span className="status-dot" aria-hidden="true" />
        {STATUS_TEXT[status]}
      </span>
      <AudioChannels state={state} />
    </div>
  );
}

/**
 * Interviewer and microphone reported separately.
 *
 * The backend deliberately keeps running on loopback alone when the mic cannot
 * be opened, so an unavailable microphone must read as one channel being off,
 * never as the app being broken. Losing loopback is the serious case, because
 * that is the channel question detection runs on.
 */
function AudioChannels({ state }: { state: SessionState }) {
  if (state.audio === "off" || state.audio === "stopped") return null;

  const interviewer = state.channels.includes("LOOPBACK");
  const microphone = state.channels.includes("MIC");

  if (state.audio === "starting") {
    return <span className="channel channel-pending">Starting audio…</span>;
  }

  return (
    <>
      <span
        className={`channel ${interviewer ? "channel-on" : "channel-off"}`}
        title={
          interviewer
            ? "System audio — questions are detected on this channel"
            : "Without system audio, questions cannot be detected automatically"
        }
      >
        Interviewer {interviewer ? "· listening" : "· unavailable"}
      </span>
      <span
        className={`channel ${microphone ? "channel-on" : "channel-muted"}`}
        title={
          microphone
            ? "Microphone — recorded for review, never answered"
            : "No microphone. Question detection is unaffected."
        }
      >
        Microphone {microphone ? "· on" : "· unavailable"}
      </span>
    </>
  );
}
