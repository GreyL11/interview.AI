import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { api } from "../api/client.ts";
import { prepareImagePaste, prepareTextPaste } from "../api/attachments.ts";
import { SessionSocket } from "../api/ws.ts";
import {
  initialSessionState,
  sessionReducer,
  type SessionState,
} from "./sessionReducer.ts";

interface SessionContextValue {
  state: SessionState;
  starting: boolean;
  startError: string | null;
  start: () => Promise<void>;
  stop: () => void;
  ask: (text: string) => void;
  cancel: () => void;
  toggleAudio: () => void;
  /** Attach pasted interviewer material to the current interview context.
   * Returns a local refusal message, or null when the frame went out. The
   * server's own acceptance/refusal arrives as an event. */
  attachPaste: (raw: string, name?: string) => string | null;
  attachImage: (bytes: Uint8Array, name?: string) => string | null;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(sessionReducer, initialSessionState);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const socketRef = useRef<SessionSocket | null>(null);

  // A live socket must not outlive the component that owns it.
  useEffect(() => () => socketRef.current?.close(), []);

  const start = useCallback(async () => {
    if (socketRef.current !== null) return;
    setStarting(true);
    setStartError(null);
    try {
      const { session_id } = await api.createSession(
        `Practice ${new Date().toLocaleString()}`,
      );
      dispatch({ kind: "reset" });
      const socket = new SessionSocket(
        session_id,
        (event) => dispatch({ kind: "event", event }),
        (connection) => dispatch({ kind: "connection", state: connection }),
      );
      socketRef.current = socket;
      socket.connect();
    } catch (error) {
      setStartError(error instanceof Error ? error.message : String(error));
    } finally {
      setStarting(false);
    }
  }, []);

  const stop = useCallback(() => {
    socketRef.current?.stopSession();
    socketRef.current = null;
  }, []);

  const ask = useCallback((text: string) => {
    socketRef.current?.askManual(text);
  }, []);

  const cancel = useCallback(() => {
    socketRef.current?.cancelAnswer();
  }, []);

  const attachPaste = useCallback((raw: string, name = "") => {
    const socket = socketRef.current;
    if (socket === null) return "Start a session before attaching material.";
    const prepared = prepareTextPaste(raw, name);
    if (!prepared.ok) return prepared.message;
    // One paste, one frame. No local queue and no retry: replaying this after
    // a reconnect would attach the same material twice, and there is no
    // message id for the backend to de-duplicate against.
    return socket.attachContext(prepared.message)
      ? null
      : "Not connected — that material was not attached.";
  }, []);

  const attachImage = useCallback((bytes: Uint8Array, name = "") => {
    const socket = socketRef.current;
    if (socket === null) return "Start a session before attaching material.";
    const prepared = prepareImagePaste(bytes, name);
    if (!prepared.ok) return prepared.message;
    return socket.attachContext(prepared.message)
      ? null
      : "Not connected — that image was not attached.";
  }, []);

  const toggleAudio = useCallback(() => {
    const socket = socketRef.current;
    if (socket === null) return;
    if (state.audio === "ok") {
      socket.stopAudio();
    } else {
      dispatch({ kind: "audio-requested" });
      socket.startAudio();
    }
  }, [state.audio]);

  const value = useMemo(
    () => ({
      state, starting, startError, start, stop, ask, cancel, toggleAudio,
      attachPaste, attachImage,
    }),
    [state, starting, startError, start, stop, ask, cancel, toggleAudio,
     attachPaste, attachImage],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (value === null) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
