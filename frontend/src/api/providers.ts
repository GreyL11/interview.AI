import type { ModelState, ModelStatus, ProviderStatus } from "./contracts.ts";

/**
 * Turning engine state into something a candidate can act on.
 *
 * Its own module rather than living in the screen so it stays testable:
 * `node --test` strips types from `.ts` but cannot parse JSX. The ordering of
 * the checks is the whole logic — "not configured" has to win over "the last
 * request failed", or a provider with no key would be reported as merely
 * having had a bad request.
 */

export type BadgeTone = "ok" | "warn" | "bad" | "idle";

export interface ProviderDescription {
  sentence: string;
  label: string;
  tone: BadgeTone;
}

const LABELS: Record<string, string> = { groq: "Groq" };

export function providerLabel(name: string): string {
  if (name in LABELS) return LABELS[name] as string;
  // A provider this build has no label for is still shown by name rather than
  // hidden: keys can be stored for providers the UI does not know about.
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/**
 * What each failure classification means for the person reading it.
 *
 * One sentence per kind, each naming the thing to do. This is the UI half of
 * the backend's error taxonomy; the vague "AI service request failed" that
 * these replaced told the user nothing and sent them to the logs.
 */
const FAILURE_SENTENCES: Record<string, { sentence: string; label: string; tone: BadgeTone }> = {
  auth: {
    sentence: "The API key was rejected. Enter it again below.",
    label: "Key rejected",
    tone: "bad",
  },
  model_unavailable: {
    sentence: "The configured model is not available to this account.",
    label: "Model unavailable",
    tone: "bad",
  },
  rate_limit: {
    sentence: "The rate limit was reached. Answers resume shortly.",
    label: "Rate limited",
    tone: "warn",
  },
  timeout: {
    sentence: "The last request timed out. Ask again.",
    label: "Timed out",
    tone: "warn",
  },
  network: {
    sentence: "This machine could not reach the provider. Check the internet connection.",
    label: "Offline",
    tone: "warn",
  },
  server: {
    sentence: "The provider reported a server error. Try again shortly.",
    label: "Provider error",
    tone: "warn",
  },
  malformed: {
    sentence: "The last answer could not be read. Asking again usually works.",
    label: "Bad response",
    tone: "warn",
  },
};

export function describeProvider(provider: ProviderStatus): ProviderDescription {
  const name = providerLabel(provider.name);

  if (!provider.configured) {
    return {
      sentence: `${name} has no API key, so questions cannot be answered yet.`,
      label: "Not configured",
      tone: "idle",
    };
  }
  if (!provider.active) {
    // The engine could not be consulted at all, so nothing here can be trusted.
    return {
      sentence: `${name} is configured, but the local engine did not load it.`,
      label: "Restart required",
      tone: "warn",
    };
  }

  const failure = provider.last_error_kind;
  if (failure !== null && failure !== "not_configured") {
    const known = FAILURE_SENTENCES[failure];
    if (known !== undefined) {
      return { sentence: `${name}: ${known.sentence}`, label: known.label, tone: known.tone };
    }
    return {
      sentence: `${name} is configured, but the last request did not succeed.`,
      label: "Last request failed",
      tone: "warn",
    };
  }

  return {
    sentence: `${name} is configured and answering questions.`,
    label: "Active",
    tone: "ok",
  };
}

/**
 * How a local model's download is going.
 *
 * "Not downloaded" and "failed halfway" need different things from the user, so
 * the UI is given a state rather than a boolean — and `failed` carries the
 * backend's own sentence, which names the directory to delete or the connection
 * to check.
 */
export interface ModelDescription {
  sentence: string;
  label: string;
  tone: BadgeTone;
  /** True while work is in progress, so the screen knows to keep polling. */
  busy: boolean;
}

const MODEL_NOUNS: Record<string, { thing: string; when: string }> = {
  stt: { thing: "Speech recognition", when: "the first time you start live audio" },
  embedding: { thing: "Document search", when: "the first time you add a document" },
};

/**
 * " — running on your graphics card" / " — running on the processor", or
 * nothing at all.
 *
 * Phrased for someone who does not know what CUDA is, and only shown once the
 * model has actually loaded: `device` is null until then, and claiming an
 * accelerator before anything has run on it would be a guess.
 */
function accelerator(model: ModelStatus): string {
  if (model.device === "cuda") return " — running on your graphics card";
  if (model.device === "cpu") return " — running on the processor";
  return "";
}

export function describeModel(model: ModelStatus | undefined, kind: string): ModelDescription {
  const noun = MODEL_NOUNS[kind] ?? { thing: "This model", when: "on first use" };

  if (model === undefined) {
    return {
      sentence: "The local engine did not report this model's status.",
      label: "Unknown",
      tone: "idle",
      busy: false,
    };
  }

  const state: ModelState = model.state;
  switch (state) {
    case "downloading":
      return {
        sentence: `Downloading now. This happens once and can take a few minutes.`,
        label: "Downloading",
        tone: "warn",
        busy: true,
      };
    case "loading":
      return {
        sentence: "Downloaded. Loading it into memory now.",
        label: "Loading",
        tone: "warn",
        busy: true,
      };
    case "ready":
      return {
        sentence: `Ready to use${accelerator(model)}.`,
        label: "Ready",
        tone: "ok",
        busy: false,
      };
    case "downloaded":
      // On disk but not yet loaded this run. Usable, and loading is fast.
      return {
        sentence: "Downloaded and ready to load.",
        label: "Ready",
        tone: "ok",
        busy: false,
      };
    case "failed":
      return {
        // The backend's sentence names the actual fix, so it is shown verbatim
        // rather than replaced with something generic.
        sentence: model.detail ?? `${noun.thing} could not be prepared.`,
        label: "Failed",
        tone: "bad",
        busy: false,
      };
    case "not_downloaded":
    default:
      return {
        sentence: `Downloads ${noun.when}.`,
        label: "Not downloaded",
        tone: "idle",
        busy: false,
      };
  }
}

/** True when any model is mid-download, so the screen should keep refreshing. */
export function anyModelBusy(models: ModelStatus[]): boolean {
  return models.some((model) => model.state === "downloading" || model.state === "loading");
}
