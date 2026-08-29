/**
 * Runs on the Node built-in test runner with native TypeScript type stripping:
 *   node --test src/api/providers.test.ts
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import type { ProviderStatus } from "./contracts.ts";
import { describeProvider, formatPriority, providerLabel } from "./providers.ts";

function provider(overrides: Partial<ProviderStatus> = {}): ProviderStatus {
  return {
    name: "groq",
    model: "openai/gpt-oss-120b",
    configured: true,
    enabled: true,
    active: true,
    available: true,
    cooling_down: false,
    cooldown_remaining_seconds: null,
    role: "primary",
    ...overrides,
  };
}

test("an available primary provider reads as active", () => {
  const described = describeProvider(provider());
  assert.equal(described.label, "Active");
  assert.equal(described.tone, "ok");
  assert.match(described.sentence, /answering questions/);
});

test("an available secondary provider reads as backup, not as a problem", () => {
  const described = describeProvider(provider({ name: "gemini", role: "backup" }));
  assert.equal(described.label, "Backup ready");
  assert.equal(described.tone, "ok");
  assert.match(described.sentence, /Gemini/);
});

test("a provider with no key is reported unconfigured, never as rate limited", () => {
  // Ordering matters: the router leaves a keyless provider looking unavailable
  // for several reasons at once, and "no key" is the only actionable one.
  const described = describeProvider(
    provider({ configured: false, available: false, cooling_down: true }),
  );
  assert.equal(described.label, "Not configured");
  assert.match(described.sentence, /no API key/);
});

test("a rate-limited provider explains itself and names the remaining wait", () => {
  const described = describeProvider(
    provider({ cooling_down: true, available: false, cooldown_remaining_seconds: 12.3 }),
  );
  assert.equal(described.label, "Cooling down");
  assert.equal(described.tone, "warn");
  assert.match(described.sentence, /about 13s/);
  assert.match(described.sentence, /backup provider/);
});

test("a rate-limited provider with no known cooldown omits the duration", () => {
  const described = describeProvider(
    provider({ cooling_down: true, available: false, cooldown_remaining_seconds: null }),
  );
  assert.equal(described.label, "Cooling down");
  assert.doesNotMatch(described.sentence, /about/);
});

test("a key added after the engine started reports restart required", () => {
  // The router is built once at startup, so a key applied afterwards genuinely
  // cannot take effect until it is rebuilt. Saying anything else would lie.
  const described = describeProvider(provider({ active: false, available: false }));
  assert.equal(described.label, "Restart required");
  assert.equal(described.tone, "warn");
});

test("a disabled provider is not confused with a broken one", () => {
  const described = describeProvider(provider({ enabled: false, available: false }));
  assert.equal(described.label, "Disabled");
  assert.equal(described.tone, "idle");
});

test("no description leaks key material or raw setting names", () => {
  const cases = [
    provider(),
    provider({ configured: false }),
    provider({ cooling_down: true, cooldown_remaining_seconds: 5 }),
    provider({ active: false }),
    provider({ enabled: false }),
  ];
  for (const candidate of cases) {
    const { sentence, label } = describeProvider(candidate);
    for (const forbidden of ["API_KEY", "GROQ_", "GEMINI_", "env"]) {
      assert.ok(!sentence.includes(forbidden), `${forbidden} in "${sentence}"`);
      assert.ok(!label.includes(forbidden), `${forbidden} in "${label}"`);
    }
  }
});

test("priority renders as a readable chain", () => {
  assert.equal(formatPriority("groq,gemini"), "Groq → Gemini");
  assert.equal(formatPriority(" gemini , groq "), "Gemini → Groq");
  assert.equal(formatPriority(""), "");
});

test("an unknown provider name is shown as-is rather than dropped", () => {
  assert.equal(providerLabel("anthropic"), "anthropic");
});
