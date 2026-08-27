import type { SessionState } from "../state/sessionReducer.ts";
import { Pill } from "./Common.tsx";

/** Connection, audio, and channel indicators. Deliberately explicit: during a
 * live session the user needs to know at a glance whether the app is actually
 * listening, and to which device. */
export function StatusPills({ state }: { state: SessionState }) {
  return (
    <div className="pills">
      <Pill
        label={connectionLabel(state)}
        tone={
          state.connection === "open" ? "ok" : state.connection === "closed" ? "idle" : "warn"
        }
        title="WebSocket connection to the local backend"
      />
      <Pill
        label={audioLabel(state)}
        tone={audioTone(state)}
        title={state.audioMessage ?? "Local audio capture"}
      />
      {state.channels.map((channel) => (
        <Pill
          key={channel}
          label={channel === "LOOPBACK" ? "interviewer" : "you"}
          tone="ok"
          title={
            channel === "LOOPBACK"
              ? "System audio — questions are detected on this channel"
              : "Microphone — recorded for review, never answered"
          }
        />
      ))}
    </div>
  );
}

function connectionLabel(state: SessionState): string {
  switch (state.connection) {
    case "open":
      return "connected";
    case "connecting":
      return "connecting…";
    case "reconnecting":
      return "reconnecting…";
    case "closed":
      return "disconnected";
    default:
      return "idle";
  }
}

function audioLabel(state: SessionState): string {
  switch (state.audio) {
    case "ok":
      return "listening";
    case "starting":
      return "starting audio…";
    case "error":
      return "audio unavailable";
    case "stopped":
      return "audio stopped";
    default:
      return "audio off";
  }
}

function audioTone(state: SessionState): "ok" | "warn" | "bad" | "idle" {
  if (state.audio === "ok") return "ok";
  if (state.audio === "error") return "bad";
  if (state.audio === "starting") return "warn";
  return "idle";
}
