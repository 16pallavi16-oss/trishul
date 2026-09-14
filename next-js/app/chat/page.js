"use client";

import { useState, useRef, useEffect } from "react";
import styles from "./chat.module.css";

const API_URL = "http://localhost:8000";

// Splits an answer like "Revenue grew 8%[1], driven by RevPAR[1]." into plain
// text and clickable citation-marker buttons, so [1] in the model's own
// output becomes something a person can actually click.
function AnswerText({ text, citations, onCiteClick }) {
  const parts = text.split(/(\[\d+\])/g);

  return (
    <p className={styles.answerText}>
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return <span key={i}>{part}</span>;

        const index = parseInt(match[1], 10);
        const citation = citations.find((c) => c.index === index);
        if (!citation) return <span key={i}>{part}</span>;

        return (
          <button
            key={i}
            className={styles.citeMark}
            onClick={() => onCiteClick(citation)}
            aria-label={`Open source ${index}`}
          >
            {index}
          </button>
        );
      })}
    </p>
  );
}

function SourcePanel({ citation, onClose }) {
  const open = citation !== null;
  const filename = citation?.file_path?.split("/").pop() ?? "Unknown source";

  return (
    <aside className={`${styles.panel} ${open ? styles.panelOpen : ""}`}>
      {citation && (
        <>
          <div className={styles.panelHeader}>
            <div>
              <div className={styles.panelFilename}>{filename}</div>
              <div className={styles.panelMeta}>Page {citation.page ?? "—"}</div>
            </div>
            <button className={styles.panelClose} onClick={onClose} aria-label="Close">
              ×
            </button>
          </div>
          <div className={styles.panelBody}>{citation.text}</div>
        </>
      )}
    </aside>
  );
}

export default function ChatPage() {
  const [threadId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, thread_id: threadId }),
      });
      if (!res.ok) throw new Error(`Request failed (${res.status})`);
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, citations: data.citations },
      ]);
    } catch (err) {
      setError("Couldn't reach the server. Check that the API is running.");
      setMessages((prev) => prev.slice(0, -1)); // remove the optimistic user message on failure
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.main}>
        <header className={styles.header}>Document Q&A</header>

        <div className={styles.thread}>
          {messages.length === 0 && (
            <div className={styles.empty}>Ask a question about the ingested documents.</div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className={styles.userMessage}>
                {m.content}
              </div>
            ) : (
              <div key={i} className={styles.assistantMessage}>
                <AnswerText
                  text={m.content}
                  citations={m.citations ?? []}
                  onCiteClick={setSelectedCitation}
                />
              </div>
            )
          )}

          {loading && <div className={styles.thinking}>Thinking…</div>}
          {error && <div className={styles.error}>{error}</div>}
          <div ref={bottomRef} />
        </div>

        <div className={styles.inputRow}>
          <textarea
            className={styles.input}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question…"
            rows={1}
            disabled={loading}
          />
          <button className={styles.sendButton} onClick={sendMessage} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </div>

      <SourcePanel citation={selectedCitation} onClose={() => setSelectedCitation(null)} />
    </div>
  );
}