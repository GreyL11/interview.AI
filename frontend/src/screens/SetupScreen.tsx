import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client.ts";
import type { AudioDevice, ModelStatus, SettingsView } from "../api/contracts.ts";
import { ErrorNote, Panel, Pill } from "../components/Common.tsx";

export function SetupScreen() {
  const [settings, setSettings] = useState<SettingsView | null>(null);
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [apiKey, setApiKey] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [current, deviceList, modelList] = await Promise.all([
        api.getSettings(),
        api.audioDevices(),
        api.modelStatus(),
      ]);
      setSettings(current);
      setDevices(deviceList);
      setModels(modelList);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
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

  if (settings === null) {
    return (
      <div className="setup">
        <Panel title="Setup">
          <p className="muted">Loading…</p>
          <ErrorNote message={error} />
        </Panel>
      </div>
    );
  }

  return (
    <div className="setup">
      <Panel title="Gemini">
        <p className="muted">
          The key is stored locally by the backend and never returned by the API — this
          screen can only tell you whether one is set.
        </p>
        <div className="field-row">
          <Pill
            label={settings.gemini_key_configured ? "key configured" : "no key"}
            tone={settings.gemini_key_configured ? "ok" : "bad"}
          />
          <span className="muted">{settings.gemini_model}</span>
        </div>
        <label className="field">
          <span>API key</span>
          <input
            type="password"
            value={apiKey}
            placeholder={settings.gemini_key_configured ? "•••••• (set)" : "paste your key"}
            onChange={(changeEvent) => setApiKey(changeEvent.target.value)}
            autoComplete="off"
          />
        </label>
        <button
          className="primary"
          disabled={apiKey.trim() === ""}
          onClick={() => {
            void save({ gemini_api_key: apiKey.trim() }, "API key saved.");
            setApiKey("");
          }}
        >
          Save key
        </button>
      </Panel>

      <Panel title="Models">
        <ul className="models">
          {models.map((model) => (
            <li key={model.kind}>
              <Pill
                label={model.downloaded ? "downloaded" : "not downloaded"}
                tone={model.downloaded ? "ok" : "warn"}
              />
              <span className="model-name">{model.name}</span>
              <span className="muted">{model.kind}</span>
            </li>
          ))}
        </ul>
        <p className="hint">
          Models download once, on first use, into the local data directory. The speech
          model is only needed for live audio.
        </p>
        <label className="field">
          <span>Speech-to-text model</span>
          <select
            value={settings.stt_model}
            onChange={(changeEvent) =>
              void save({ stt_model: changeEvent.target.value }, "Speech model updated.")
            }
          >
            <option value="tiny.en">tiny.en — fastest, least accurate</option>
            <option value="base.en">base.en — fast</option>
            <option value="distil-small.en">distil-small.en — recommended</option>
            <option value="small.en">small.en — most accurate, slowest</option>
          </select>
        </label>
        <label className="field">
          <span>Compute device</span>
          <select
            value={settings.stt_device}
            onChange={(changeEvent) =>
              void save({ stt_device: changeEvent.target.value }, "Device updated.")
            }
          >
            <option value="auto">auto — GPU when usable, else CPU</option>
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
          </select>
        </label>
      </Panel>

      <Panel title="Audio">
        {!settings.audio_available && (
          <p className="warning">
            Audio support is not installed. Install <code>faster-whisper</code> and{" "}
            <code>sounddevice</code> to enable live capture; typed practice works without
            them.
          </p>
        )}
        <label className="check">
          <input
            type="checkbox"
            checked={settings.audio_capture_loopback}
            onChange={(changeEvent) =>
              void save(
                { audio_capture_loopback: changeEvent.target.checked },
                "Audio channels updated.",
              )
            }
          />
          <span>
            Capture system audio (the interviewer) — <strong>questions are detected on
            this channel</strong>
          </span>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.audio_capture_mic}
            onChange={(changeEvent) =>
              void save(
                { audio_capture_mic: changeEvent.target.checked },
                "Audio channels updated.",
              )
            }
          />
          <span>Capture microphone (you) — recorded for review, never answered</span>
        </label>

        <h4>Detected devices</h4>
        {devices.length === 0 ? (
          <p className="muted">No capture devices detected.</p>
        ) : (
          <ul className="devices">
            {devices.map((device) => (
              <li key={`${device.channel}-${device.index}`}>
                <Pill
                  label={device.channel === "LOOPBACK" ? "interviewer" : "you"}
                  tone={device.channel === "LOOPBACK" ? "ok" : "idle"}
                />
                <span>{device.name}</span>
                {device.is_default && <span className="muted">default</span>}
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title="Retrieval">
        <label className="field">
          <span>Results per question ({settings.rag_top_k})</span>
          <input
            type="range"
            min={1}
            max={10}
            value={settings.rag_top_k}
            onChange={(changeEvent) =>
              void save({ rag_top_k: Number(changeEvent.target.value) }, "Retrieval updated.")
            }
          />
        </label>
        <label className="field">
          <span>Minimum similarity ({settings.rag_min_similarity.toFixed(2)})</span>
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.05}
            value={settings.rag_min_similarity}
            onChange={(changeEvent) =>
              void save(
                { rag_min_similarity: Number(changeEvent.target.value) },
                "Retrieval updated.",
              )
            }
          />
        </label>
        <p className="hint">
          Raise the threshold if answers cite loosely-related documents; lower it if
          relevant experience is being missed.
        </p>
      </Panel>

      <Panel title="Storage">
        <p className="muted">
          Documents, embeddings, session history and models all live in:
        </p>
        <code className="path">{settings.data_dir}</code>
      </Panel>

      {status !== null && <p className="muted">{status}</p>}
      <ErrorNote message={error} />
    </div>
  );
}
