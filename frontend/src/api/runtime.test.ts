import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  backendRuntime,
  DEV_RUNTIME,
  isPackaged,
  parseBackendConfig,
  setBackendConfig,
} from "./runtime.ts";

/**
 * The rule under test, in one sentence: inside the desktop app, guessing is
 * never allowed.
 *
 * The packaged build shipped with a fallback that quietly used development port
 * 8000 whenever the shell's injected config was missing. The visible symptom
 * was "Cannot reach the backend at http://127.0.0.1:8000" while the real
 * backend sat healthy on its actual random port — an error naming a port
 * nothing was ever going to be listening on.
 */

// The modules under test read `window`, and node:test has no DOM. Pointing
// `window` at `globalThis` is enough for two boolean-ish properties and avoids
// pulling in jsdom to test what is essentially a pure function.
type TestWindow = {
  __BACKEND__?: unknown;
  __TAURI_INTERNALS__?: unknown;
  window?: unknown;
};
const win = globalThis as unknown as TestWindow;
win.window = globalThis;

function asDesktop() {
  win.__TAURI_INTERNALS__ = {};
}

function asBrowser() {
  delete win.__TAURI_INTERNALS__;
}

afterEach(() => {
  delete win.__BACKEND__;
  delete win.__TAURI_INTERNALS__;
});

// -------------------------------------------------------------- validation

test("a well-formed injected config is accepted", () => {
  assert.deepEqual(parseBackendConfig({ port: 51234, token: "abc" }), {
    port: 51234,
    token: "abc",
  });
});

test("a malformed injected config is rejected rather than half-used", () => {
  // Each of these would otherwise become a NaN port or an empty Bearer token,
  // surfacing much later as an unexplainable "failed to fetch".
  const rejected: unknown[] = [
    undefined,
    null,
    "not an object",
    42,
    {},
    { port: 8000 },
    { token: "abc" },
    { port: "8000", token: "abc" },
    { port: 0, token: "abc" },
    { port: -1, token: "abc" },
    { port: 70000, token: "abc" },
    { port: 1.5, token: "abc" },
    { port: Number.NaN, token: "abc" },
    { port: 8000, token: "" },
    { port: 8000, token: 123 },
  ];
  for (const value of rejected) {
    assert.equal(parseBackendConfig(value), null, `should reject ${JSON.stringify(value)}`);
  }
});

test("setBackendConfig refuses to store something malformed", () => {
  assert.equal(setBackendConfig({ port: "nope" }), false);
  assert.equal(win.__BACKEND__, undefined);

  assert.equal(setBackendConfig({ port: 51234, token: "t" }), true);
  assert.deepEqual(win.__BACKEND__, { port: 51234, token: "t" });
});

// ------------------------------------------------------------ development

test("a browser with no shell falls back to the documented dev port", () => {
  asBrowser();
  const runtime = backendRuntime();
  assert.equal(runtime.kind, "ready");
  if (runtime.kind !== "ready") return;
  assert.equal(runtime.baseUrl, `http://127.0.0.1:${DEV_RUNTIME.port}`);
  assert.equal(runtime.wsBase, `ws://127.0.0.1:${DEV_RUNTIME.port}`);
  assert.equal(runtime.token, DEV_RUNTIME.token);
  assert.equal(runtime.packaged, false);
});

test("isPackaged tracks the shell, not the injected config", () => {
  asBrowser();
  assert.equal(isPackaged(), false);
  asDesktop();
  assert.equal(isPackaged(), true);
});

// --------------------------------------------------------------- packaged

test("a packaged app uses the port and token the shell chose", () => {
  asDesktop();
  setBackendConfig({ port: 51234, token: "shell-token" });

  const runtime = backendRuntime();
  assert.equal(runtime.kind, "ready");
  if (runtime.kind !== "ready") return;
  assert.equal(runtime.baseUrl, "http://127.0.0.1:51234");
  assert.equal(runtime.wsBase, "ws://127.0.0.1:51234");
  assert.equal(runtime.token, "shell-token");
  assert.equal(runtime.packaged, true);
});

test("a packaged app with no config reports unavailable instead of guessing", () => {
  // THE regression. Before this, the assertion below was `baseUrl ===
  // "http://127.0.0.1:8000"` in effect, and the app confidently reported a
  // failure against a port the backend was never on.
  asDesktop();

  const runtime = backendRuntime();
  assert.equal(runtime.kind, "unavailable");
  if (runtime.kind !== "unavailable") return;
  assert.ok(runtime.reason.length > 0);
  assert.ok(!runtime.reason.includes("8000"));
});

test("a packaged app with a malformed config reports unavailable too", () => {
  asDesktop();
  win.__BACKEND__ = { port: "not-a-number", token: "" };

  const runtime = backendRuntime();
  assert.equal(runtime.kind, "unavailable");
  if (runtime.kind !== "unavailable") return;
  // Distinguishable from "not injected yet", because they need different fixes.
  assert.match(runtime.reason, /could not read/i);
});

test("the dev port is never reachable from inside the packaged app", () => {
  asDesktop();
  for (const injected of [undefined, {}, { port: 0, token: "" }]) {
    if (injected === undefined) delete win.__BACKEND__;
    else win.__BACKEND__ = injected;

    const runtime = backendRuntime();
    assert.notEqual(
      runtime.kind,
      "ready",
      `packaged app must not resolve a runtime from ${JSON.stringify(injected)}`,
    );
  }
});

test("the dev fallback survives a config the shell has not written yet", () => {
  // Outside the shell the fallback is correct and must stay: `npm run dev` in a
  // browser has no injector at all.
  asBrowser();
  win.__BACKEND__ = undefined;
  assert.equal(backendRuntime().kind, "ready");
});
