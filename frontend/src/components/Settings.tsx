import type { ReactNode } from "react";

/**
 * Layout primitives for the Settings screen.
 *
 * Three small pieces rather than one clever component: a settings page is a
 * list of labelled facts, and the variation between rows is entirely in the
 * value, not the structure.
 */

export function SettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <section className="settings-section">
      <header className="settings-section-head">
        <h2>{title}</h2>
        {description !== undefined && <p className="settings-section-desc">{description}</p>}
      </header>
      <div className="settings-rows">{children}</div>
    </section>
  );
}

export function SettingsRow({
  label,
  hint,
  control,
  children,
}: {
  label: string;
  hint?: ReactNode;
  /** Right-hand side: a badge, a value, or an actual control. */
  control?: ReactNode;
  /** Full-width content below the label, for anything a row cannot hold. */
  children?: ReactNode;
}) {
  return (
    <div className="settings-row">
      <div className="settings-row-main">
        <div className="settings-row-label">
          <span className="settings-label">{label}</span>
          {hint !== undefined && <span className="settings-hint">{hint}</span>}
        </div>
        {control !== undefined && <div className="settings-row-control">{control}</div>}
      </div>
      {children}
    </div>
  );
}

export type BadgeTone = "ok" | "warn" | "bad" | "idle";

/**
 * Status as a word plus a shape, never colour alone: a red and a green dot are
 * the same dot to a colour-blind user, and this screen exists to answer
 * "is it working?".
 */
export function StatusBadge({ label, tone }: { label: string; tone: BadgeTone }) {
  return (
    <span className={`status-badge badge-${tone}`}>
      <span className="badge-dot" aria-hidden="true" />
      {label}
    </span>
  );
}
