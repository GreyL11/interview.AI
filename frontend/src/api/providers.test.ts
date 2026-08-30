import assert from "node:assert/strict";
import { test } from "node:test";

import type { ModelStatus, ProviderStatus } from "./contracts.ts";
import {
  anyModelBusy,
  describeModel,
  describeProvider,
  providerLabel,
} from "./providers.ts";

function provider(overrides: Partial<ProviderStatus> = {}): ProviderStatus {
  return {
    name: "groq",
    model: "openai/gpt-oss-120b",
    configured: true,
    active: true,
    last_error_kind: null,
    ...overrides,
  };
}

function model(overrides: Partial<ModelStatus> = {}): ModelStatus {
  return {
    name: "distil-small.en",
    kind: "stt",
    state: "not_downloaded",
    downloaded: false,
    path: "C:\\models",
    detail: null,
    device: null,
    ...overrides,
  };
}

// ------------------------------------------------------------------ providers

test("a working provider reads as active", () => {
  const described = describeProvider(provider());
  assert.equal(described.tone, "ok");
  assert.match(described.sentence, /Groq/);
  assert.match(described.sentence, /answering/);
});

test("a provider with no key is reported as unconfigured, not as failing", () => {
  // Ordering matters: "no key" has to win over every failure state, or the
  // user is told to check their connection when they never entered a key.
  const described = describeProvider(
    provider({ configured: false, last_error_kind: "not_configured" }),
  );
  assert.equal(described.label, "Not configured");
  assert.equal(described.tone, "idle");
});

test("a rejected key says to re-enter it rather than blaming the network", () => {
  const described = describeProvider(provider({ last_error_kind: "auth" }));
  assert.equal(described.label, "Key rejected");
  assert.equal(described.tone, "bad");
  assert.match(described.sentence, /Enter it again/);
});

test("an unavailable model names the model as the problem", () => {
  const described = describeProvider(provider({ last_error_kind: "model_unavailable" }));
  assert.equal(described.label, "Model unavailable");
  assert.match(described.sentence, /not available/);
});

test("a rate limit reads as temporary, not broken", () => {
  const described = describeProvider(provider({ last_error_kind: "rate_limit" }));
  assert.equal(described.tone, "warn");
  assert.match(described.sentence, /resume/);
});

test("a network failure points at the connection", () => {
  const described = describeProvider(provider({ last_error_kind: "network" }));
  assert.match(described.sentence, /internet connection/);
});

test("every failure kind produces its own distinct sentence", () => {
  // The whole point of the taxonomy: two different causes must not read the
  // same, or classifying them was pointless.
  const kinds = [
    "auth",
    "model_unavailable",
    "rate_limit",
    "timeout",
    "network",
    "server",
    "malformed",
  ] as const;
  const sentences = kinds.map(
    (kind) => describeProvider(provider({ last_error_kind: kind })).sentence,
  );
  assert.equal(new Set(sentences).size, kinds.length);
});

test("an unrecognised failure kind still produces something useful", () => {
  // A backend newer than this frontend can send a kind that is not in the map.
  const described = describeProvider(
    provider({ last_error_kind: "something_new" as never }),
  );
  assert.equal(described.tone, "warn");
  assert.ok(described.sentence.length > 0);
});

test("a provider the engine never loaded asks for a restart", () => {
  const described = describeProvider(provider({ active: false }));
  assert.equal(described.label, "Restart required");
});

test("no user-facing sentence leaks configuration or key vocabulary", () => {
  const states: ProviderStatus[] = [
    provider(),
    provider({ configured: false }),
    provider({ active: false }),
    provider({ last_error_kind: "auth" }),
    provider({ last_error_kind: "network" }),
  ];
  for (const state of states) {
    const { sentence } = describeProvider(state);
    for (const forbidden of ["API_KEY", "GROQ_", "GEMINI_", "env", "Bearer"]) {
      assert.ok(!sentence.includes(forbidden), `${forbidden} leaked into "${sentence}"`);
    }
  }
});

test("Gemini is gone from the provider vocabulary", () => {
  // Regression guard for the removal: a stale label would make a dead provider
  // look supported the moment any code passed its name through.
  const rendered = describeProvider(provider({ name: "groq" }));
  assert.ok(!/gemini/i.test(rendered.sentence));
  assert.ok(!/gemini/i.test(JSON.stringify(rendered)));
});

test("an unknown provider name is still shown rather than hidden", () => {
  // Keys can be stored for providers this build has no code for.
  assert.equal(providerLabel("groq"), "Groq");
  assert.equal(providerLabel("someprovider"), "Someprovider");
});

// --------------------------------------------------------------------- models

test("a model that has not been downloaded says when it will be", () => {
  const described = describeModel(model(), "stt");
  assert.equal(described.label, "Not downloaded");
  assert.equal(described.busy, false);
  assert.match(described.sentence, /live audio/);
});

test("a downloading model reads as busy so the screen keeps polling", () => {
  const described = describeModel(model({ state: "downloading" }), "stt");
  assert.equal(described.label, "Downloading");
  assert.equal(described.busy, true);
});

test("a loading model is busy too", () => {
  assert.equal(describeModel(model({ state: "loading" }), "stt").busy, true);
});

test("a model is only ready once it has actually loaded", () => {
  const described = describeModel(
    model({ state: "ready", downloaded: true }),
    "stt",
  );
  assert.equal(described.label, "Ready");
  assert.equal(described.tone, "ok");
});

test("a ready model names the hardware it actually runs on", () => {
  // Phrased for someone who does not know what CUDA is.
  const gpu = describeModel(model({ state: "ready", device: "cuda" }), "stt");
  assert.match(gpu.sentence, /graphics card/);

  const cpu = describeModel(model({ state: "ready", device: "cpu" }), "stt");
  assert.match(cpu.sentence, /processor/);
});

test("the accelerator is not claimed before the model has loaded", () => {
  // `device` is null until something has actually run on it, so saying "GPU"
  // here would be a guess rather than a fact.
  const described = describeModel(model({ state: "ready", device: null }), "stt");
  assert.doesNotMatch(described.sentence, /graphics card|processor/);
  assert.match(described.sentence, /Ready/);
});

test("a failed model shows the engine's own explanation", () => {
  // The backend's sentence names the directory to delete or the connection to
  // check; replacing it with something generic would throw that away.
  const detail = "Could not download the speech model. The first run needs internet access.";
  const described = describeModel(model({ state: "failed", detail }), "stt");
  assert.equal(described.label, "Failed");
  assert.equal(described.tone, "bad");
  assert.equal(described.sentence, detail);
});

test("a failed model with no detail still says something", () => {
  const described = describeModel(model({ state: "failed", detail: null }), "embedding");
  assert.equal(described.tone, "bad");
  assert.ok(described.sentence.length > 0);
});

test("the document search model describes its own trigger", () => {
  assert.match(describeModel(model({ kind: "embedding" }), "embedding").sentence, /document/);
});

test("a missing model entry does not crash the screen", () => {
  const described = describeModel(undefined, "stt");
  assert.equal(described.label, "Unknown");
  assert.equal(described.busy, false);
});

test("polling runs only while something is actually happening", () => {
  assert.equal(anyModelBusy([]), false);
  assert.equal(anyModelBusy([model({ state: "ready" })]), false);
  assert.equal(anyModelBusy([model({ state: "not_downloaded" })]), false);
  assert.equal(anyModelBusy([model({ state: "failed" })]), false);
  assert.equal(anyModelBusy([model({ state: "downloading" })]), true);
  assert.equal(
    anyModelBusy([model({ state: "ready" }), model({ state: "loading" })]),
    true,
  );
});
