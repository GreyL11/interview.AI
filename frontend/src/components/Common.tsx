import type { ReactNode } from "react";

export function Pill({
  label,
  tone,
  title,
}: {
  label: string;
  tone: "ok" | "warn" | "bad" | "idle";
  title?: string;
}) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {label}
    </span>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="empty">
      <p className="empty-title">{title}</p>
      {hint !== undefined && <p className="empty-hint">{hint}</p>}
    </div>
  );
}

export function Panel({
  title,
  actions,
  children,
}: {
  title: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{title}</h2>
        {actions}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

export function ErrorNote({ message }: { message: string | null }) {
  if (message === null) return null;
  return <p className="error-note">{message}</p>;
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="spinner" role="status" aria-live="polite">
      {label}
    </span>
  );
}
