"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  BookOpenText,
  FileText,
  MessageSquareText,
  SendHorizontal,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Upload,
  X,
} from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

function apiUrl() {
  if (API_URL) return API_URL.replace(/\/$/, "");
  return `${window.location.protocol}//${window.location.hostname}:8000`;
}

async function streamChat({ question, threadId, onCitations, onToken, onDone, onError }) {
  const res = await fetch(`${apiUrl()}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const raw of events) {
      if (!raw.trim()) continue;
      let eventName = "message";
      let dataLine = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine);
      if (eventName === "citations") onCitations(payload.citations);
      else if (eventName === "token") onToken(payload.token);
      else if (eventName === "done") onDone(payload);
      else if (eventName === "error") onError(payload.message);
    }
  }
}

function AnswerBody({ text, citations, onCiteClick, streaming }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="font-serif text-lg leading-[1.75] text-stone-800 sm:text-xl">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return <span key={i}>{part}</span>;
        const index = parseInt(match[1], 10);
        const citation = citations.find((c) => c.index === index);
        if (!citation) return <span key={i}>{part}</span>;
        return (
          <button
            key={i}
            onClick={() => onCiteClick(citation)}
            aria-label={`Open source ${index}`}
            className="mx-0.5 inline-flex h-[19px] min-w-[19px] items-center justify-center rounded-md border border-teal-700/20 bg-teal-50 px-1 align-super font-sans text-[11px] font-semibold leading-none text-teal-800 transition duration-200 hover:-translate-y-0.5 hover:border-teal-700 hover:bg-teal-700 hover:text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-700"
          >
            {index}
          </button>
        );
      })}
      {streaming && (
        <span className="ml-0.5 inline-block h-[1em] w-0.5 animate-pulse bg-teal-700 align-text-bottom" />
      )}
    </p>
  );
}

function FeedbackButtons({ messageId }) {
  const [sent, setSent] = useState(null);

  async function send(rating) {
    setSent(rating); // optimistic -- the click should feel instant
    try {
      await fetch(`${apiUrl()}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: messageId, rating }),
      });
    } catch {
      setSent(null); // roll back if it never reached the server
    }
  }

  if (!messageId) return null;

  return (
    <div className="mt-2 flex gap-1">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => send("up")}
        aria-label="Helpful"
        className={sent === "up" ? "text-teal-700" : "text-stone-400"}
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => send("down")}
        aria-label="Not helpful"
        className={sent === "down" ? "text-rose-700" : "text-stone-400"}
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

export default function ChatPage() {
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function updateLast(patch) {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      next[next.length - 1] = typeof patch === "function" ? patch(last) : { ...last, ...patch };
      return next;
    });
  }

  async function clearConversation() {
    try {
      await fetch(`${apiUrl()}/chat/${threadId}`, { method: "DELETE" });
    } catch {
      // clearing server-side is best-effort; the local reset matters more
    }
    setMessages([]);
    setSelected(null);
    setError(null);
    setThreadId(crypto.randomUUID()); // fresh thread, so old context can't leak in
  }

  async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "assistant", content: "", citations: [], streaming: true },
    ]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      await streamChat({
        question,
        threadId,
        onCitations: (citations) => updateLast({ citations }),
        onToken: (token) => updateLast((last) => ({ ...last, content: last.content + token })),
        onDone: (payload) =>
          updateLast({
            content: payload.answer,
            messageId: payload.message_id,
            evalCount: payload.eval_count,
            streaming: false,
          }),
        onError: (message) => {
          setError(message);
          updateLast({ streaming: false });
        },
      });
    } catch {
      setError(`Couldn't reach the API at ${apiUrl()}. Start FastAPI in the project root, then try again.`);
      setMessages((prev) => prev.slice(0, -2));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-[#f6f8f7] text-stone-900">
      <header className="z-10 w-full border-b border-stone-200/80 bg-white/90 px-4 py-3 shadow-sm backdrop-blur sm:px-8">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-800 text-white shadow-sm shadow-teal-900/15">
              <BookOpenText className="h-5 w-5" />
            </div>
            <div>
              <h1 className="font-serif text-xl font-semibold tracking-tight sm:text-2xl">Document Q&amp;A</h1>
              <p className="hidden text-xs font-medium tracking-wide text-stone-500 sm:block">YOUR KNOWLEDGE, CONNECTED</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" asChild className="rounded-xl border-stone-200 bg-white transition hover:border-teal-700/30 hover:bg-teal-50">
              <Link href="/upload">
                <Upload className="mr-1.5 h-4 w-4" />
                Upload
              </Link>
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={clearConversation}
              disabled={messages.length === 0 || loading}
              className="rounded-xl text-stone-500 transition-colors hover:bg-rose-50 hover:text-rose-700"
            >
              <Trash2 className="mr-1.5 h-4 w-4" />
              Clear
            </Button>
          </div>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-8 sm:py-8">
        <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 lg:flex-row lg:items-stretch">
          <section aria-label="Conversation" className="flex min-h-full min-w-0 flex-1 flex-col transition-[flex] duration-300 ease-out">
            <div className={`mx-auto flex w-full flex-1 flex-col gap-6 ${selected ? "max-w-6xl" : "max-w-3xl"} ${messages.length === 0 ? "justify-center" : "justify-start"}`}>
            {messages.length === 0 && (
              <div className="empty-state-enter mx-auto max-w-xl space-y-5 text-center">
                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-[1.35rem] bg-teal-100 text-teal-800 shadow-sm shadow-teal-900/5">
                  <MessageSquareText className="h-7 w-7" strokeWidth={1.8} />
                </div>
                <div className="space-y-3">
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-teal-800">Your document assistant</p>
                  <h2 className="font-serif text-4xl font-semibold tracking-tight text-stone-900 sm:text-5xl">Ask your documents</h2>
                  <p className="mx-auto max-w-lg text-base leading-relaxed text-stone-600 sm:text-lg">
                    Find answers across your files, with clear citations that take you straight to the source.
                  </p>
                </div>
                <div className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-white/80 px-3.5 py-2 text-sm text-stone-600 shadow-sm">
                  <span className="h-2 w-2 rounded-full bg-teal-600" />
                  Grounded in your uploaded documents
                </div>
              </div>
            )}

            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="message-enter flex justify-end">
                  <div className="max-w-[88%] rounded-2xl rounded-br-md bg-teal-800 px-5 py-3.5 text-base leading-relaxed text-white shadow-md shadow-teal-950/10 sm:max-w-[75%] sm:text-lg">
                    {m.content}
                  </div>
                </div>
              ) : (
                <div key={i} className="message-enter flex flex-col gap-4 lg:flex-row lg:items-start">
                  <div className="min-w-0 flex-1 rounded-2xl border border-stone-200/80 bg-white px-5 py-4 shadow-sm shadow-stone-900/[0.03] sm:px-6 sm:py-5">
                    <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.12em] text-stone-500">
                      <Sparkles className="h-4 w-4 text-teal-700" />
                      Answer
                    </div>
                    {m.content ? (
                      <AnswerBody
                        text={m.content}
                        citations={m.citations ?? []}
                        onCiteClick={(citation) => setSelected({ citation, messageIndex: i })}
                        streaming={m.streaming}
                      />
                    ) : (
                      <p className="animate-pulse text-sm text-stone-500">Searching your documents…</p>
                    )}
                    <div className="mt-3 flex items-center justify-between gap-4 border-t border-stone-100 pt-2">
                      {m.evalCount != null ? (
                        <p className="text-xs text-stone-400">{m.evalCount} generated tokens</p>
                      ) : <span />}
                      {!m.streaming && <FeedbackButtons messageId={m.messageId} />}
                    </div>
                  </div>
                  {selected?.messageIndex === i && (
                    <div className="citation-panel-space-enter w-full shrink-0 overflow-hidden lg:w-[360px]">
                      <aside
                        key={`${selected.citation.file_path}-${selected.citation.page}-${selected.citation.index}`}
                        aria-label="Citation source"
                        className="max-h-[65dvh] w-full overflow-y-auto rounded-2xl border border-stone-200 bg-white shadow-lg shadow-stone-900/[0.06] lg:max-h-[calc(100dvh-10rem)]"
                      >
                        <div className="sticky top-0 flex items-start justify-between gap-4 border-b border-stone-100 bg-white/95 px-5 py-4 backdrop-blur">
                          <div className="flex min-w-0 items-start gap-3">
                            <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-teal-50 text-teal-800">
                              <FileText className="h-4 w-4" />
                            </div>
                            <div className="min-w-0">
                              <h2 className="break-words text-sm font-semibold leading-snug text-stone-900">
                                {selected.citation.file_path?.split(/[\\/]/).pop() ?? "Source"}
                              </h2>
                              <p className="mt-1 text-xs text-stone-500">Page {selected.citation.page ?? "—"} · Citation {selected.citation.index}</p>
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setSelected(null)}
                            aria-label="Close citation source"
                            className="h-8 w-8 shrink-0 rounded-lg text-stone-500 hover:bg-stone-100"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                        <div className="p-5">
                          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-teal-800">Source excerpt</p>
                          <p className="whitespace-pre-wrap font-serif text-[15px] leading-[1.8] text-stone-700">
                            {selected.citation.text}
                          </p>
                        </div>
                      </aside>
                    </div>
                  )}
                </div>
              )
            )}

            {error && (
              <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
          </section>

        </div>
      </main>

      <footer className="z-10 w-full border-t border-stone-200/80 bg-white/90 px-4 py-3 backdrop-blur sm:px-8 sm:py-4">
          <div className="mx-auto flex max-w-3xl items-end gap-3 rounded-2xl border border-stone-300/90 bg-white p-2 shadow-lg shadow-stone-900/[0.06] transition duration-200 focus-within:border-teal-600 focus-within:ring-4 focus-within:ring-teal-600/10">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask anything about your documents…"
              rows={1}
              disabled={loading}
              className="max-h-40 min-h-[52px] flex-1 resize-none border-0 bg-transparent px-4 py-3 text-base shadow-none focus-visible:ring-0 sm:text-lg"
            />
            <Button onClick={sendMessage} disabled={loading || !input.trim()} aria-label="Send message" className="mb-0.5 h-11 w-11 rounded-xl bg-teal-800 text-white shadow-sm transition duration-200 hover:-translate-y-0.5 hover:bg-teal-900 disabled:translate-y-0 disabled:opacity-45">
              <SendHorizontal className="h-4 w-4" />
            </Button>
          </div>
          <p className="mx-auto mt-2 max-w-3xl px-2 text-[11px] text-stone-400">Enter to send <span className="px-1">·</span> Shift + Enter for a new line</p>
      </footer>
    </div>
  );
}
