import { useEffect, useState } from "react";

import { api } from "./api/client.ts";
import { HistoryScreen } from "./screens/HistoryScreen.tsx";
import { KnowledgeScreen } from "./screens/KnowledgeScreen.tsx";
import { PracticeScreen } from "./screens/PracticeScreen.tsx";
import { SetupScreen } from "./screens/SetupScreen.tsx";
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

export function App() {
  const [screen, setScreen] = useState<Screen>("practice");
  const [backendUp, setBackendUp] = useState<boolean | null>(null);

  useEffect(() => {
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
    const timer = setInterval(() => void check(), 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <SessionProvider>
      <div className="app">
        <nav className="nav">
          <span className="brand">Interview Coach</span>
          {SCREENS.map((candidate) => (
            <button
              key={candidate}
              className={screen === candidate ? "nav-item active" : "nav-item"}
              onClick={() => setScreen(candidate)}
            >
              {LABELS[candidate]}
            </button>
          ))}
          <span className="nav-spacer" />
          <span className={`backend-dot ${backendUp === true ? "up" : backendUp === false ? "down" : ""}`}>
            {backendUp === true ? "backend ready" : backendUp === false ? "backend down" : "…"}
          </span>
        </nav>

        <main className="main">
          {backendUp === false && (
            <p className="error-note">
              The local backend is not responding. In development, start it with{" "}
              <code>uvicorn app.main:app --port 8000</code>.
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
