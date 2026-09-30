"use client";

import { useState, useRef } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { UploadCloud, MessageSquare, FileWarning, CheckCircle2 } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

function apiUrl() {
  if (API_URL) return API_URL.replace(/\/$/, "");
  return `${window.location.protocol}//${window.location.hostname}:8000`;
}

export default function UploadPage() {
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef(null);

  // A dropped folder arrives as a tree of FileSystemEntry objects, not a flat
  // file list -- this walks it recursively to pull out every actual file.
  async function filesFromEntry(entry, path = "") {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      return [{ file, path: path + file.name }];
    }
    if (entry.isDirectory) {
      const reader = entry.createReader();
      const entries = await new Promise((res) => reader.readEntries(res));
      const nested = await Promise.all(
        entries.map((e) => filesFromEntry(e, `${path}${entry.name}/`))
      );
      return nested.flat();
    }
    return [];
  }

  async function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const items = Array.from(e.dataTransfer.items)
      .map((i) => i.webkitGetAsEntry?.())
      .filter(Boolean);
    const collected = (await Promise.all(items.map((entry) => filesFromEntry(entry)))).flat();
    if (collected.length) upload(collected);
  }

  function handlePick(e) {
    const collected = Array.from(e.target.files).map((file) => ({
      file,
      path: file.webkitRelativePath || file.name,
    }));
    if (collected.length) upload(collected);
  }

  async function upload(collected) {
    setError(null);
    setUploading(true);
    setJob(null);

    const form = new FormData();
    for (const { file, path } of collected) {
      form.append("files", file, path);
    }

    try {
      const res = await fetch(`${apiUrl()}/ingest/upload`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed (${res.status})`);
      const { job_id } = await res.json();
      streamProgress(job_id);
    } catch (err) {
      setError(err.message);
      setUploading(false);
    }
  }

  async function streamProgress(jobId) {
    const res = await fetch(`${apiUrl()}/ingest/jobs/${jobId}/stream`);
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
        const dataLine = raw
          .split("\n")
          .find((l) => l.startsWith("data:"))
          ?.slice(5)
          .trim();
        if (dataLine) setJob(JSON.parse(dataLine));
      }
    }
    setUploading(false);
  }

  const pct = job?.total ? Math.round((job.completed / job.total) * 100) : 0;
  const finished = job?.status === "done" || job?.status === "failed";

  return (
    <div className="min-h-screen bg-stone-50">
      <header className="flex items-center justify-between border-b border-stone-200 px-8 py-4">
        <h1 className="font-serif text-xl font-semibold text-stone-900">Ingest Documents</h1>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/chat">
            <MessageSquare className="mr-1.5 h-4 w-4" />
            Chat
          </Link>
        </Button>
      </header>

      <main className="mx-auto max-w-2xl px-8 py-10">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`rounded-lg border-2 border-dashed p-12 text-center transition-colors ${
            dragging ? "border-teal-600 bg-teal-50" : "border-stone-300 bg-white"
          }`}
        >
          <UploadCloud className="mx-auto h-9 w-9 text-stone-400" />
          <p className="mt-3 text-[15px] text-stone-700">Drop a folder here to ingest it</p>
          <p className="mt-1 text-sm text-stone-500">PDF, DOCX, PPTX, XLSX, CSV</p>
          <Button
            variant="outline"
            size="sm"
            className="mt-5"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
          >
            Or choose a folder
          </Button>
          <input
            ref={inputRef}
            type="file"
            webkitdirectory=""
            directory=""
            multiple
            hidden
            onChange={handlePick}
          />
        </div>

        {error && (
          <div className="mt-5 rounded-md border border-rose-200 bg-rose-50 px-3.5 py-2.5 text-sm text-rose-800">
            {error}
          </div>
        )}

        {job && (
          <div className="mt-8 rounded-lg border border-stone-200 bg-white p-6">
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium text-stone-900">
                {job.status === "done"
                  ? "Ingestion complete"
                  : job.status === "failed"
                  ? "Ingestion failed"
                  : "Ingesting…"}
              </span>
              <span className="font-mono text-sm text-stone-500">
                {job.completed ?? 0} / {job.total ?? 0}
              </span>
            </div>

            <Progress value={pct} className="mt-3" />

            <p className="mt-3 truncate text-sm text-stone-500">
              {job.current ? `Reading ${job.current}` : finished ? `${job.chunks ?? 0} chunks created` : "Starting…"}
            </p>

            {job.status === "done" && (
              <div className="mt-4 flex items-center gap-2 text-sm text-teal-800">
                <CheckCircle2 className="h-4 w-4" />
                {job.chunks ?? 0} chunks from {job.total ?? 0} files.
                <Link href="/chat" className="underline underline-offset-2">
                  Ask a question
                </Link>
              </div>
            )}

            {job.errors?.length > 0 && (
              <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3">
                <div className="flex items-center gap-1.5 text-sm font-medium text-amber-900">
                  <FileWarning className="h-4 w-4" />
                  {job.errors.length} file{job.errors.length > 1 ? "s" : ""} skipped
                </div>
                <ul className="mt-2 space-y-1">
                  {job.errors.map((e, i) => (
                    <li key={i} className="text-sm text-amber-800">
                      <span className="font-medium">{e.file}</span> — {e.error}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
