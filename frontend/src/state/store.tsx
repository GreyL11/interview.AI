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
    () => ({ state, starting, startError, start, stop, ask, cancel, toggleAudio }),
    [state, starting, startError, start, stop, ask, cancel, toggleAudio],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const value = useContext(SessionContext);
  if (value === null) throw new Error("useSession must be used inside SessionProvider");
  return value;
}
