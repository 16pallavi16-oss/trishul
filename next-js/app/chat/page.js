"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { ThumbsUp, ThumbsDown, Trash2, Upload, SendHorizontal, MessageSquareText } from "lucide-react";

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
            className="mx-0.5 inline-flex h-[18px] min-w-[18px] items-center justify-center rounded border border-teal-700 bg-teal-50 px-1 align-super font-sans text-[11px] font-semibold leading-none text-teal-800 transition-colors hover:bg-teal-700 hover:text-white"
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

  const last = messages[messages.length - 1];
  const searching = loading && last?.role === "assistant" && last.content === "";

  return (
    <div className="flex h-dvh flex-col bg-[#f6f8f7] text-stone-900">
      <header className="w-full border-b border-stone-200/80 bg-white/90 px-4 py-4 shadow-sm backdrop-blur sm:px-8">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          <h1 className="font-serif text-2xl font-semibold tracking-tight sm:text-3xl">Document Q&amp;A</h1>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" asChild>
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
            >
              <Trash2 className="mr-1.5 h-4 w-4" />
              Clear
            </Button>
          </div>
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col overflow-y-auto px-4 py-6 sm:px-8 sm:py-8">
          <div className={`mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 ${messages.length === 0 ? "justify-center" : "justify-start"}`}>
            {messages.length === 0 && (
              <div className="mx-auto max-w-xl space-y-3 text-center">
                <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-100 text-teal-800">
                  <MessageSquareText className="h-7 w-7" />
                </div>
                <h2 className="font-serif text-3xl font-semibold tracking-tight sm:text-4xl">Ask your documents</h2>
                <p className="text-lg leading-relaxed text-stone-600">Ask a question about the documents you’ve ingested. Answers include citations to their sources.</p>
              </div>
            )}

            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-teal-800 px-5 py-3 text-base leading-relaxed text-white shadow-sm sm:max-w-[75%] sm:text-lg">
                    {m.content}
                  </div>
                </div>
              ) : (
                <div key={i}>
                  <AnswerBody
                    text={m.content}
                    citations={m.citations ?? []}
                    onCiteClick={setSelected}
                    streaming={m.streaming}
                  />
                  {m.evalCount != null && (
                    <p className="mt-1 text-xs text-stone-400">{m.evalCount} generated tokens</p>
                  )}
                  {!m.streaming && <FeedbackButtons messageId={m.messageId} />}
                </div>
              )
            )}

            {searching && <p className="text-sm italic text-stone-500">Searching documents…</p>}
            {error && (
              <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-base text-rose-800">
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
      </main>

      <footer className="w-full border-t border-stone-200/80 bg-white/90 px-4 py-4 sm:px-8">
          <div className="mx-auto flex max-w-3xl items-end gap-3 rounded-2xl border border-stone-300 bg-white p-2 shadow-md shadow-stone-900/5 focus-within:border-teal-600 focus-within:ring-4 focus-within:ring-teal-600/10">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask a question…"
              rows={1}
              disabled={loading}
              className="max-h-40 min-h-[52px] flex-1 resize-none border-0 bg-transparent px-4 py-3 text-base shadow-none focus-visible:ring-0 sm:text-lg"
            />
            <Button onClick={sendMessage} disabled={loading || !input.trim()}>
              <SendHorizontal className="h-4 w-4" />
            </Button>
          </div>
      </footer>

      <Sheet open={selected !== null} onOpenChange={(open) => !open && setSelected(null)}>
        <SheetContent className="w-[420px] overflow-y-auto sm:max-w-[420px]">
          <SheetHeader>
            <SheetTitle className="break-words text-sm">
              {selected?.file_path?.split("/").pop() ?? "Source"}
            </SheetTitle>
            <p className="text-sm text-stone-500">Page {selected?.page ?? "—"}</p>
          </SheetHeader>
          <p className="mt-5 whitespace-pre-wrap font-serif text-[15px] leading-[1.7] text-stone-800">
            {selected?.text}
          </p>
        </SheetContent>
      </Sheet>
    </div>
  );
}
