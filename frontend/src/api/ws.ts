import type { ClientMessage, ServerEvent } from "./contracts.ts";
import { parseServerEvent } from "./contracts.ts";
import { backendRuntime } from "./runtime.ts";

export type ConnectionListener = (
  state: "connecting" | "open" | "reconnecting" | "closed",
) => void;

const BASE_RETRY_MS = 500;
const MAX_RETRY_MS = 10_000;
const PING_INTERVAL_MS = 20_000;

/**
 * WebSocket client for one practice session.
 *
 * Reconnects with `?since_seq=` so the backend replays only what was missed.
 * The session lives in the backend, so a dropped socket pauses the view rather
 * than ending the run — reconnecting is a catch-up, not a restart.
 */
export class SessionSocket {
  private socket: WebSocket | null = null;
  private retryMs = BASE_RETRY_MS;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private closedByUs = false;
  private lastSeq = 0;
  // Local-only latency diagnostics: the backend can't see receive-to-render
  // time, so this pairs question.detected with the first answer.delta for
  // the same turn and logs the gap. Never sent anywhere.
  private pendingQuestionTurnId: number | null = null;
  private pendingQuestionAt: number | null = null;

  constructor(
    private readonly sessionId: string,
    private readonly onEvent: (event: ServerEvent) => void,
    private readonly onConnection: ConnectionListener,
  ) {}

  connect(): void {
    this.closedByUs = false;
    this.onConnection(this.lastSeq === 0 ? "connecting" : "reconnecting");

    const runtime = backendRuntime();
    const url =
      `${runtime.wsBase}/ws/session/${this.sessionId}` +
      `?token=${encodeURIComponent(runtime.token)}&since_seq=${this.lastSeq}`;

    const socket = new WebSocket(url);
    this.socket = socket;

    socket.onopen = () => {
      this.retryMs = BASE_RETRY_MS;
      this.onConnection("open");
      this.startPing();
    };

    socket.onmessage = (message) => {
      let parsed: unknown;
      try {
        parsed = JSON.parse(String(message.data));
      } catch {
        return; // a malformed frame must not take down the UI
      }
      const event = parseServerEvent(parsed);
      if (event === null) return;
      if (event.seq > this.lastSeq) this.lastSeq = event.seq;
      this.traceFirstVisibleToken(event);
      this.onEvent(event);
    };

    socket.onerror = () => {
      // onclose always follows; reconnect is handled there.
    };

    socket.onclose = () => {
      this.stopPing();
      this.socket = null;
      if (this.closedByUs) {
        this.onConnection("closed");
        return;
      }
      this.onConnection("reconnecting");
      this.scheduleRetry();
    };
  }

  /** Logs receive-to-first-visible-token time for one turn, once. This is a
   * message-to-message gap (question.detected -> first answer.delta), not a
   * paint measurement -- React dispatches and renders synchronously in the
   * same tick with no debounce in this app, so it is a close proxy without
   * needing a render-observer hook. */
  private traceFirstVisibleToken(event: ServerEvent): void {
    if (event.type === "question.detected") {
      this.pendingQuestionTurnId = event.turn_id;
      this.pendingQuestionAt = performance.now();
      return;
    }
    if (
      event.type === "answer.delta" &&
      event.turn_id === this.pendingQuestionTurnId &&
      this.pendingQuestionAt !== null
    ) {
      const ms = performance.now() - this.pendingQuestionAt;
      console.debug(`[latency] question_detected -> first answer.delta: ${ms.toFixed(1)}ms`);
      this.pendingQuestionAt = null;
    }
  }

  private scheduleRetry(): void {
    if (this.retryTimer !== null) return;
    const delay = this.retryMs;
    // Exponential backoff so a backend that is down for a while does not get
    // hammered, capped so recovery still feels immediate.
    this.retryMs = Math.min(this.retryMs * 2, MAX_RETRY_MS);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.connect();
    }, delay);
  }

  private startPing(): void {
    this.stopPing();
    this.pingTimer = setInterval(() => this.send({ type: "ping" }), PING_INTERVAL_MS);
  }

  private stopPing(): void {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  send(message: ClientMessage): boolean {
    if (this.socket === null || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }

  askManual(text: string): boolean {
    return this.send({ type: "question.manual", data: { text } });
  }

  cancelAnswer(): boolean {
    return this.send({ type: "answer.cancel" });
  }

  startAudio(): boolean {
    return this.send({ type: "audio.start" });
  }

  stopAudio(): boolean {
    return this.send({ type: "audio.stop" });
  }

  /** End the session on the backend, then close. */
  stopSession(): void {
    this.send({ type: "session.stop" });
    this.close();
  }

  close(): void {
    this.closedByUs = true;
    if (this.retryTimer !== null) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.stopPing();
    this.socket?.close();
    this.socket = null;
    this.onConnection("closed");
  }
}
