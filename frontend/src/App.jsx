import { useState, useRef } from "react";

const API_BASE = "http://localhost:8000";

/**
 * This is the piece that's actually "your job" in a Frontend+AI role: the
 * backend already did retrieval + reasoning + tool calling (see orchestrator.py).
 * Everything below is about presenting that well — loading states, errors,
 * and turning structured JSON (answer/sources/confidence) into real UI.
 */
export default function App() {
  const [documents, setDocuments] = useState([]); // [{name, chunkCount}]
  const [messages, setMessages] = useState([]);   // [{role, text, sources, confidence, error}]
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setUploadError(null);
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Upload failed.");
      }

      setDocuments((prev) => [...prev, { name: data.filename, chunkCount: data.chunks_created }]);
    } catch (err) {
      // Network errors (backend not running) land here too, not just API errors.
      setUploadError(err.message || "Could not reach the server.");
    } finally {
      setIsUploading(false);
    }
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || isThinking) return;

    setMessages((prev) => [...prev, { role: "user", text: question }]);
    setInput("");
    setIsThinking(true);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Something went wrong.");
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          confidence: data.confidence,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", error: true, text: err.message || "Could not reach the server." },
      ]);
    } finally {
      setIsThinking(false);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div>Marginnote</div>
          <div className="brand-sub">Ask questions, get answers grounded in your document</div>
        </div>

        <div
          className={`dropzone${isDragging ? " dragging" : ""}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <div className="dropzone-label">
            {isUploading ? (
              "Reading and indexing…"
            ) : (
              <>
                <strong>Click to upload</strong> or drop a .txt / .pdf
              </>
            )}
          </div>
        </div>

        {uploadError && <div className="upload-error">{uploadError}</div>}

        <div className="doc-list">
          <div className="doc-list-title">Uploaded</div>
          {documents.length === 0 && (
            <div className="sidebar-empty">Nothing uploaded yet.</div>
          )}
          {documents.map((doc) => (
            <div className="doc-item" key={doc.name}>
              <span className="doc-item-name">{doc.name}</span>
              <span className="doc-item-count">{doc.chunkCount} chunks</span>
            </div>
          ))}
        </div>
      </aside>

      <main className="main">
        <div className="conversation">
          {messages.length === 0 && (
            <div className="empty-state">
              <h1>Nothing asked yet</h1>
              <p>Upload a document on the left, then ask it a question below.</p>
            </div>
          )}

          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}

          {isThinking && (
            <div className="message assistant">
              <div className="thinking">
                Reading the document
                <span className="thinking-dot" />
                <span className="thinking-dot" />
                <span className="thinking-dot" />
              </div>
            </div>
          )}
        </div>

        <div className="composer">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleSend(); }}
            placeholder="Ask something about the document…"
            disabled={isThinking}
          />
          <button onClick={handleSend} disabled={isThinking || !input.trim()}>
            Ask
          </button>
        </div>
      </main>
    </div>
  );
}

function MessageBubble({ message }) {
  const { role, text, sources, confidence, error } = message;

  return (
    <div className={`message ${role}${error ? " error" : ""}`}>
      <div className="message-bubble">{text}</div>

      {role === "assistant" && !error && sources && sources.length > 0 && (
        <div className="citations">
          {sources.map((s, i) => (
            <div className="citation" key={i}>
              <span className="citation-ref">[{i + 1}]</span>
              <span className="citation-doc">{s.doc_name}</span>
              {" — "}
              {s.text.slice(0, 140)}
              {s.text.length > 140 ? "…" : ""}
            </div>
          ))}
        </div>
      )}

      {role === "assistant" && !error && typeof confidence === "number" && (
        <div className="confidence">confidence {Math.round(confidence * 100)}%</div>
      )}
    </div>
  );
}
