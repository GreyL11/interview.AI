import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client.ts";
import type {
  AudioDevice,
  ModelStatus,
  SettingsView,
} from "../api/contracts.ts";
import { appVersion, isDesktop, openDataFolder, openLogsFolder } from "../api/desktop.ts";
import { describeProvider, formatPriority, providerLabel } from "../api/providers.ts";
import { SettingsRow, SettingsSection, StatusBadge } from "../components/Settings.tsx";

export function SetupScreen() {
  const [settings, setSettings] = useState<SettingsView | null>(null);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [version, setVersion] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      // Audio devices and model status are secondary: a failure there must not
      // blank the whole screen, so only `settings` gates rendering.
      const current = await api.getSettings();
      setSettings(current);
      setError(null);
      const [deviceList, modelList] = await Promise.all([
        api.audioDevices().catch(() => []),
        api.modelStatus().catch(() => []),
      ]);
      setDevices(deviceList);
      setModels(modelList);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    void appVersion().then(setVersion);
  }, [refresh]);

  const save = useCallback(
    async (update: Parameters<typeof api.updateSettings>[0], note: string) => {
      try {
        setSettings(await api.updateSettings(update));
        setStatus(note);
        setError(null);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    },
    [],
  );

  if (loading && settings === null) {
    return (
      <div className="settings">
        <p className="muted" role="status">
          Loading settings…
        </p>
      </div>
    );
  }

  // Backend down: say which parts are unavailable, not that the app is broken.
  if (settings === null) {
    return (
      <div className="settings">
        <SettingsSection
          title="Settings unavailable"
          description="These settings are read from the local AI engine, which is not responding right now."
        >
          <SettingsRow
            label="Local engine"
            hint={error ?? "Not responding."}
            control={<StatusBadge label="Unavailable" tone="bad" />}
          />
          <SettingsRow
            label="Retry"
            hint="Settings will load once the engine is back."
            control={
              <button onClick={() => void refresh()}>Try again</button>
            }
          />
          {isDesktop() && (
            <SettingsRow
              label="Diagnostics"
              hint="The engine's log may explain why it stopped."
              control={<button onClick={() => void openLogsFolder()}>Open Logs Folder</button>}
            />
          )}
        </SettingsSection>
      </div>
    );
  }

  // Read defensively. An older engine (or a partial response) has no
  // `providers`, and Settings failing to render is a far worse outcome than
  // one section being empty.
  const providers = Array.isArray(settings.providers) ? settings.providers : [];
  const priority = settings.provider_priority ?? "";
  // Prefer the default device. Windows reports several inputs per channel and
  // the first one is often not the one that will actually be captured.
  const pick = (channel: "LOOPBACK" | "MIC"): AudioDevice | undefined => {
    const matching = devices.filter((device) => device.channel === channel);
    return matching.find((device) => device.is_default) ?? matching[0];
  };
  const interviewerDevice = pick("LOOPBACK");
  const micDevice = pick("MIC");
  const sttModel = models.find((model) => model.kind === "stt");
  const embeddingModel = models.find((model) => model.kind === "embedding");

  return (
    <div className="settings">
      {settings.settings_persist === false && (
        <p className="settings-notice" role="note">
          API keys are saved on this machine. Other settings here apply immediately but
          are not saved — they return to their defaults when you restart Interview Coach.
        </p>
      )}

      <SettingsSection
        title="AI providers"
        description="Answers are generated in the cloud. Everything else — audio, speech recognition, documents and history — stays on this machine."
      >
        {providers.map((provider) => {
          const described = describeProvider(provider);
          return (
            <SettingsRow
              key={provider.name}
              label={providerLabel(provider.name)}
              hint={
                <>
                  {described.sentence}{" "}
                  <span className="settings-mono">{provider.model}</span>
                </>
              }
              control={<StatusBadge label={described.label} tone={described.tone} />}
            />
          );
        })}
        {providers.length === 0 && (
          <SettingsRow
            label="Providers"
            hint="The local engine did not report any provider information."
            control={<StatusBadge label="Unknown" tone="idle" />}
          />
        )}
        <SettingsRow
          label="Provider order"
          hint="The first available provider answers; the next takes over if it is rate limited."
          control={
            <span className="settings-value">{formatPriority(priority)}</span>
          }
        />
        {["groq", "gemini"].map((name) => (
          <ProviderKeyRow
            key={name}
            provider={name}
            label={providerLabel(name)}
            configured={providers.some((p) => p.name === name && p.configured)}
            secureStorage={settings.secure_storage_available}
            onChanged={refresh}
          />
        ))}
      </SettingsSection>

      <SettingsSection
        title="Interview audio"
        description="Questions are detected from the interviewer's audio. Your microphone is transcribed for review only and is never answered."
      >
        {!settings.audio_available && (
          <p className="settings-notice" role="note">
            Audio capture is unavailable in this build. Typed practice still works.
          </p>
        )}
        <SettingsRow
          label="Interviewer audio"
          hint={
            interviewerDevice !== undefined
              ? interviewerDevice.name
              : "No system-audio device detected. Questions cannot be detected automatically."
          }
          control={
            <StatusBadge
              label={interviewerDevice !== undefined ? "Detected" : "Unavailable"}
              tone={interviewerDevice !== undefined ? "ok" : "bad"}
            />
          }
        />
        <SettingsRow
          label="Your microphone"
          hint={
            micDevice !== undefined
              ? micDevice.name
              : "No microphone detected. Interviewer audio and question detection are unaffected."
          }
          control={
            <StatusBadge
              label={micDevice !== undefined ? "Detected" : "Unavailable"}
              tone={micDevice !== undefined ? "ok" : "idle"}
            />
          }
        />
        <SettingsRow
          label="Capture interviewer audio"
          hint="Turning this off disables automatic question detection."
          control={
            <input
              type="checkbox"
              aria-label="Capture interviewer audio"
              checked={settings.audio_capture_loopback}
              onChange={(changeEvent) =>
                void save(
                  { audio_capture_loopback: changeEvent.target.checked },
                  "Audio channels updated.",
                )
              }
            />
          }
        />
        <SettingsRow
          label="Capture your microphone"
          hint="Recorded alongside the interview for review."
          control={
            <input
              type="checkbox"
              aria-label="Capture your microphone"
              checked={settings.audio_capture_mic}
              onChange={(changeEvent) =>
                void save(
                  { audio_capture_mic: changeEvent.target.checked },
                  "Audio channels updated.",
                )
              }
            />
          }
        />
      </SettingsSection>

      <SettingsSection
        title="Speech recognition"
        description="Runs entirely on this machine. Models download once, on first use."
      >
        <SettingsRow
          label="Speech model"
          hint={
            sttModel?.downloaded === true
              ? "Ready to use."
              : "Downloads the first time you start live audio."
          }
          control={
            <StatusBadge
              label={sttModel?.downloaded === true ? "Ready" : "Not downloaded"}
              tone={sttModel?.downloaded === true ? "ok" : "warn"}
            />
          }
        >
          <label className="settings-field">
            <span className="settings-field-label">Accuracy and speed</span>
            <select
              value={settings.stt_model}
              onChange={(changeEvent) =>
                void save({ stt_model: changeEvent.target.value }, "Speech model updated.")
              }
            >
              <option value="tiny.en">Fastest — least accurate</option>
              <option value="base.en">Fast</option>
              <option value="distil-small.en">Balanced — recommended</option>
              <option value="small.en">Most accurate — slowest</option>
            </select>
          </label>
        </SettingsRow>
        <SettingsRow
          label="Document search model"
          hint={
            embeddingModel?.downloaded === true
              ? "Ready to use."
              : "Downloads the first time you add a document."
          }
          control={
            <StatusBadge
              label={embeddingModel?.downloaded === true ? "Ready" : "Not downloaded"}
              tone={embeddingModel?.downloaded === true ? "ok" : "warn"}
            />
          }
        />
      </SettingsSection>

      <SettingsSection
        title="Data and privacy"
        description="Documents, session history, the search index and downloaded models are stored on this machine only."
      >
        <SettingsRow
          label="API keys"
          hint={
            settings.secure_storage_available
              ? "Kept in this machine's credential manager. They are never shown again after saving, never included in settings responses, and never written to the log."
              : "This machine has no credential manager, so keys are held in memory for the current session only and are never written to a file."
          }
          control={
            <StatusBadge
              label={settings.secure_storage_available ? "Credential manager" : "Session only"}
              tone={settings.secure_storage_available ? "ok" : "warn"}
            />
          }
        />
        <SettingsRow
          label="Audio recordings"
          hint="Audio is transcribed as it is captured and is never written to disk."
          control={<StatusBadge label="Not stored" tone="ok" />}
        />
        <SettingsRow label="Storage location" hint="Everything Interview Coach keeps lives here.">
          <span className="settings-mono settings-path">{settings.data_dir}</span>
        </SettingsRow>
        {isDesktop() && (
          <SettingsRow
            label="Local files"
            hint="Open the folder holding your documents, history and models."
            control={
              <div className="settings-actions">
                <button onClick={() => void openDataFolder()}>Open Data Folder</button>
                <button onClick={() => void openLogsFolder()}>Open Logs Folder</button>
              </div>
            }
          />
        )}
      </SettingsSection>

      <SettingsSection title="About">
        <SettingsRow
          label="Interview Coach"
          hint={isDesktop() ? "Desktop application" : "Running in a browser (development)"}
          control={
            <span className="settings-value">{version !== null ? `Version ${version}` : "—"}</span>
          }
        />
        <SettingsRow
          label="Local engine"
          hint="Speech recognition, documents and history run here."
          control={<StatusBadge label="Running" tone="ok" />}
        />
      </SettingsSection>

      {status !== null && (
        <p className="muted" role="status">
          {status}
        </p>
      )}
      {error !== null && (
        <p className="error-note" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

/**
 * Save / replace / remove one provider key.
 *
 * The field is always empty on mount and cleared on submit: there is nothing
 * to preload, because the API never returns a key once set. `persisted` comes
 * from the save response rather than being assumed, so a machine without a
 * credential store is told the truth instead of being reassured.
 */
function ProviderKeyRow({
  provider,
  label,
  configured,
  secureStorage,
  onChanged,
}: {
  provider: string;
  label: string;
  configured: boolean;
  secureStorage: boolean;
  onChanged: () => Promise<void>;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState<"saving" | "removing" | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const inputId = `key-${provider}`;

  const run = async (action: "saving" | "removing", work: () => Promise<string>) => {
    setBusy(action);
    setFailure(null);
    setResult(null);
    try {
      const message = await work();
      setValue("");
      setResult(message);
      await onChanged();
    } catch (cause) {
      setFailure(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };

  const save = () =>
    run("saving", async () => (await api.setProviderKey(provider, value.trim())).detail);

  const remove = () =>
    run("removing", async () => (await api.removeProviderKey(provider)).detail);

  return (
    <SettingsRow
      label={`${label} API key`}
      hint={
        configured
          ? "Saved on this machine. It is never shown again."
          : secureStorage
            ? "Stored in this machine's credential manager, not in a file."
            : "This machine has no credential store, so a key will only last until you restart."
      }
      control={
        <StatusBadge
          label={configured ? "Configured" : "Not configured"}
          tone={configured ? "ok" : "idle"}
        />
      }
    >
      <div className="settings-key">
        <label className="visually-hidden" htmlFor={inputId}>
          {configured ? `Replace ${label} API key` : `${label} API key`}
        </label>
        <input
          id={inputId}
          type="password"
          autoComplete="off"
          spellCheck={false}
          disabled={busy !== null}
          placeholder={configured ? "Paste a new key to replace it" : "Paste your key"}
          value={value}
          onChange={(changeEvent) => setValue(changeEvent.target.value)}
          onKeyDown={(keyEvent) => {
            if (keyEvent.key === "Enter" && value.trim() !== "") void save();
          }}
        />
        <button
          disabled={value.trim() === "" || busy !== null}
          title={value.trim() === "" ? "Paste a key first" : undefined}
          onClick={() => void save()}
        >
          {busy === "saving" ? "Saving…" : configured ? "Replace" : "Save"}
        </button>
        {configured && (
          <button
            className="danger"
            disabled={busy !== null}
            onClick={() => void remove()}
          >
            {busy === "removing" ? "Removing…" : "Remove"}
          </button>
        )}
      </div>
      {result !== null && (
        <p className="settings-feedback" role="status">
          {result}
        </p>
      )}
      {failure !== null && (
        <p className="error-note" role="alert">
          {failure}
        </p>
      )}
    </SettingsRow>
  );
}
