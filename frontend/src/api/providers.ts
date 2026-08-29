import type { ProviderStatus } from "./contracts.ts";

/**
 * Turning provider state into something a candidate can act on.
 *
 * Its own module rather than living in the screen so it stays testable:
 * `node --test` strips types from `.ts` but cannot parse JSX. The ordering of
 * the checks is the whole logic — "not configured" has to win over "cooling
 * down", or a provider with no key would be reported as merely rate limited.
 */

export type BadgeTone = "ok" | "warn" | "bad" | "idle";

export interface ProviderDescription {
  sentence: string;
  label: string;
  tone: BadgeTone;
}

const LABELS: Record<string, string> = { groq: "Groq", gemini: "Gemini" };

export function providerLabel(name: string): string {
  return LABELS[name] ?? name;
}

export function describeProvider(provider: ProviderStatus): ProviderDescription {
  const name = providerLabel(provider.name);

  if (!provider.configured) {
    return {
      sentence: `${name} has no API key, so it cannot be used.`,
      label: "Not configured",
      tone: "idle",
    };
  }
  if (!provider.enabled) {
    return {
      sentence: `${name} is configured but turned off.`,
      label: "Disabled",
      tone: "idle",
    };
  }
  if (provider.cooling_down) {
    const seconds = provider.cooldown_remaining_seconds;
    return {
      sentence:
        `${name} is temporarily unavailable` +
        (seconds !== null ? ` for about ${Math.ceil(seconds)}s` : "") +
        ". The backup provider is being used.",
      label: "Cooling down",
      tone: "warn",
    };
  }
  if (!provider.active) {
    // Configured after the engine started, so the router was built without it.
    // The only honest thing to say is that a restart is needed.
    return {
      sentence: `${name} is configured but was not loaded when the engine started.`,
      label: "Restart required",
      tone: "warn",
    };
  }
  if (!provider.available) {
    return {
      sentence: `${name} is configured but not currently answering.`,
      label: "Unavailable",
      tone: "warn",
    };
  }
  return {
    sentence:
      provider.role === "primary"
        ? `${name} is configured and answering questions.`
        : `${name} is configured and ready as backup.`,
    label: provider.role === "primary" ? "Active" : "Backup ready",
    tone: "ok",
  };
}

/** "groq,gemini" -> "Groq → Gemini". Empty input yields an empty string. */
export function formatPriority(priority: string): string {
  return priority
    .split(",")
    .map((name) => name.trim())
    .filter((name) => name !== "")
    .map(providerLabel)
    .join(" → ");
}
