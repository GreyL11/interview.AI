import { useEffect, useState } from "react";

import { api } from "./api/client.ts";
import {
  getStartupState,
  isDesktop,
  onStartupChange,
  type StartupState,
} from "./api/desktop.ts";
import { HistoryScreen } from "./screens/HistoryScreen.tsx";
import { KnowledgeScreen } from "./screens/KnowledgeScreen.tsx";
import { PracticeScreen } from "./screens/PracticeScreen.tsx";
import { SetupScreen } from "./screens/SetupScreen.tsx";
import { StartupScreen } from "./screens/StartupScreen.tsx";
import { SessionProvider } from "./state/store.tsx";

// Four screens with no URLs to share and no deep links: a state switch is the
// whole router this app needs.
const SCREENS = ["practice", "knowledge", "history", "setup"] as const;
type Screen = (typeof SCREENS)[number];

const LABELS: Record<Screen, string> = {
  practice: "Practice",
  knowledge: "Knowledge",
  history: "History",
  setup: "Setup",
};

/** Health poll interval once the app is up. Slow on purpose: this is a
 * liveness check for a local process, not a data source. */
const HEALTH_INTERVAL_MS = 10_000;

export function App() {
  const [screen, setScreen] = useState<Screen>("practice");
  const [backendUp, setBackendUp] = useState<boolean | null>(null);
  const [startup, setStartup] = useState<StartupState | null>(null);

  // The shell owns the backend, so it is the source of truth for startup.
  useEffect(() => {
    let cancelled = false;
    let stop: (() => void) | undefined;
    void (async () => {
      const initial = await getStartupState();
      if (!cancelled) setStartup(initial);
      stop = await onStartupChange((next) => {
        if (!cancelled) setStartup(next);
      });
    })();
    return () => {
      cancelled = true;
      stop?.();
    };
  }, []);

  const ready = startup?.status === "ready";

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    const check = async () => {
      try {
        await api.health();
        if (!cancelled) setBackendUp(true);
      } catch {
        if (!cancelled) setBackendUp(false);
      }
    };
    void check();
    const timer = setInterval(() => void check(), HEALTH_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [ready]);

  // Settings is one keystroke away, the way a native app would do it. Escape is
  // left alone: the transcript input needs it.
  useEffect(() => {
    const onKey = (keyEvent: KeyboardEvent) => {
      if ((keyEvent.ctrlKey || keyEvent.metaKey) && keyEvent.key === ",") {
        keyEvent.preventDefault();
        setScreen("setup");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (startup === null || startup.status !== "ready") {
    return (
      <StartupScreen
        state={startup ?? { status: "starting", stage: "launching", label: "Opening Interview Coach" }}
      />
    );
  }

  return (
    <SessionProvider>
      <div className="app">
        <nav className="nav">
          <span className="brand">Interview Coach</span>
          {SCREENS.map((candidate) => (
            <button
              key={candidate}
              className={screen === candidate ? "nav-item active" : "nav-item"}
              aria-current={screen === candidate ? "page" : undefined}
              onClick={() => setScreen(candidate)}
            >
              {LABELS[candidate]}
            </button>
          ))}
          <span className="nav-spacer" />
          <span
            className={`backend-dot ${backendUp === true ? "up" : backendUp === false ? "down" : ""}`}
          >
            {backendUp === true ? "Engine ready" : backendUp === false ? "Engine down" : "…"}
          </span>
        </nav>

        <main className="main">
          {backendUp === false && (
            <p className="error-note" role="alert">
              {isDesktop()
                ? "Lost contact with the local engine. It will reconnect automatically if it comes back."
                : "The local backend is not responding. Start it with uvicorn app.main:app --port 8000."}
            </p>
          )}
          {screen === "practice" && <PracticeScreen />}
          {screen === "knowledge" && <KnowledgeScreen />}
          {screen === "history" && <HistoryScreen />}
          {screen === "setup" && <SetupScreen />}
        </main>

        <footer className="foot">
          For mock interviews and practice where AI assistance is permitted. Audio and
          documents stay on this machine.
        </footer>
      </div>
    </SessionProvider>
  );
}
