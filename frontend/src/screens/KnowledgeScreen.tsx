import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api/client.ts";
import {
  KNOWLEDGE_TYPES,
  PERSONAL_KNOWLEDGE_TYPES,
  type DocumentRecord,
  type KnowledgeType,
} from "../api/contracts.ts";
import { Empty, ErrorNote, Panel, Pill } from "../components/Common.tsx";

const ACCEPT = ".pdf,.docx,.md,.markdown,.txt";

export function KnowledgeScreen() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [knowledgeType, setKnowledgeType] = useState<KnowledgeType>("RESUME");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      setError(null);
      for (const file of Array.from(files)) {
        setBusy(`Uploading ${file.name}…`);
        try {
          const uploaded = await api.uploadDocument(file, knowledgeType);
          setBusy(`Indexing ${file.name}…`);
          // Upload and ingest are separate calls so a slow parse cannot make
          // the upload itself look like it failed.
          const result = await api.ingestDocument(uploaded.document_id);
          if (result.status === "FAILED") {
            setError(`${file.name}: ${result.error ?? "ingestion failed"}`);
          }
        } catch (cause) {
          setError(cause instanceof Error ? cause.message : String(cause));
        }
      }
      setBusy(null);
      await refresh();
    },
    [knowledgeType, refresh],
  );

  const remove = useCallback(
    async (id: string) => {
      setBusy("Deleting…");
      try {
        await api.deleteDocument(id);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
      setBusy(null);
      await refresh();
    },
    [refresh],
  );

  const reingest = useCallback(
    async (id: string) => {
      setBusy("Re-indexing…");
      try {
        await api.ingestDocument(id);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
      setBusy(null);
      await refresh();
    },
    [refresh],
  );

  return (
    <div className="knowledge">
      <Panel title="Add documents">
        <p className="muted">
          Everything here stays on this machine. Only the relevant excerpt for a given
          question is ever sent to the model — never the whole library.
        </p>

        <label className="field">
          <span>Knowledge type</span>
          <select
            value={knowledgeType}
            onChange={(changeEvent) =>
              setKnowledgeType(changeEvent.target.value as KnowledgeType)
            }
          >
            {KNOWLEDGE_TYPES.map((type) => (
              <option key={type} value={type}>
                {type}
                {PERSONAL_KNOWLEDGE_TYPES.includes(type) ? " (personal experience)" : ""}
              </option>
            ))}
          </select>
        </label>
        <p className="hint">
          Personal types are the only ones used to answer “tell me about a time…”
          questions. Technical and reference material is kept separate so it can never be
          presented as something you did.
        </p>

        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onDragOver={(dragEvent) => {
            dragEvent.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(dropEvent) => {
            dropEvent.preventDefault();
            setDragging(false);
            void upload(dropEvent.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
        >
          <p>Drop PDF, DOCX, Markdown or TXT here, or click to choose</p>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPT}
            hidden
            onChange={(changeEvent) => {
              if (changeEvent.target.files !== null) void upload(changeEvent.target.files);
              changeEvent.target.value = "";
            }}
          />
        </div>

        {busy !== null && <p className="muted">{busy}</p>}
        <ErrorNote message={error} />
      </Panel>

      <Panel title={`Your documents (${documents.length})`}>
        {documents.length === 0 ? (
          <Empty
            title="Nothing indexed yet"
            hint="Add your CV first — it is what grounds personal-experience answers."
          />
        ) : (
          <table className="docs">
            <thead>
              <tr>
                <th>File</th>
                <th>Type</th>
                <th>Status</th>
                <th>Chunks</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {documents.map((document) => (
                <tr key={document.document_id}>
                  <td title={document.filename}>{document.filename}</td>
                  <td>
                    <Pill
                      label={document.knowledge_type}
                      tone={
                        PERSONAL_KNOWLEDGE_TYPES.includes(document.knowledge_type)
                          ? "ok"
                          : "idle"
                      }
                    />
                  </td>
                  <td>
                    <Pill
                      label={document.status}
                      tone={
                        document.status === "READY"
                          ? "ok"
                          : document.status === "FAILED"
                            ? "bad"
                            : "warn"
                      }
                      title={document.error ?? undefined}
                    />
                  </td>
                  <td>{document.chunk_count}</td>
                  <td className="row-actions">
                    {document.status !== "READY" && (
                      <button onClick={() => void reingest(document.document_id)}>Retry</button>
                    )}
                    <button
                      className="danger"
                      onClick={() => void remove(document.document_id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}
