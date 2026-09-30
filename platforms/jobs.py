"""
Background ingestion jobs with live progress.

Ingesting a folder takes minutes (OCR especially), far too long to hold an
HTTP request open. So the upload endpoint starts a job in a background
thread and returns a job_id immediately; the client then polls or streams
progress from Redis using that id.

Progress lives in Redis rather than Postgres because it is written on every
single file and read constantly while a job runs -- high-churn, short-lived
data that nobody needs after the job finishes.
"""
import json
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import redis
from dotenv import load_dotenv

from .ingest import ingest_file, SUPPORTED_EXTS
from .db import Session, engine, init_db, persist_file_and_chunks

load_dotenv()

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
JOB_TTL_SECONDS = 60 * 60 * 24  # progress records expire after a day

_redis_client = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _job_key(job_id: str) -> str:
    return f"ingest_job:{job_id}"


def _write(job_id: str, **fields) -> None:
    r = _get_redis()
    key = _job_key(job_id)
    r.hset(key, mapping={k: json.dumps(v) for k, v in fields.items()})
    r.expire(key, JOB_TTL_SECONDS)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    raw = _get_redis().hgetall(_job_key(job_id))
    if not raw:
        return None
    return {k: json.loads(v) for k, v in raw.items()}


def _collect_files(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def _run_job(job_id: str, folder: Path) -> None:
    """Runs in a background thread. Every state change is written to Redis
    so the client sees progress without this function returning anything."""
    try:
        init_db()
        files = _collect_files(folder)
        _write(
            job_id,
            status="running",
            total=len(files),
            completed=0,
            current=None,
            chunks=0,
            errors=[],
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        if not files:
            _write(job_id, status="done", finished_at=datetime.now(timezone.utc).isoformat())
            return

        completed = 0
        total_chunks = 0
        errors: List[Dict[str, str]] = []

        for path in files:
            _write(job_id, current=path.name)
            try:
                file_chunks = ingest_file(path)
                if file_chunks:
                    with Session(engine) as session:
                        persist_file_and_chunks(path, file_chunks, session)
                total_chunks += len(file_chunks)
            except Exception as e:
                # One bad file must not kill the whole job -- record it and
                # keep going, same policy as ingest_folder's per-file guard.
                errors.append({"file": path.name, "error": str(e)})
            completed += 1
            _write(job_id, completed=completed, chunks=total_chunks, errors=errors)

        _write(
            job_id,
            status="done",
            current=None,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as e:
        _write(
            job_id,
            status="failed",
            error=str(e),
            traceback=traceback.format_exc()[-2000:],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def start_ingest_job(folder: Path) -> str:
    """Kicks off ingestion in a background thread, returns the job id at once."""
    job_id = str(uuid.uuid4())
    _write(
        job_id,
        status="queued",
        total=0,
        completed=0,
        current=None,
        chunks=0,
        errors=[],
        folder=str(folder),
    )
    thread = threading.Thread(target=_run_job, args=(job_id, folder), daemon=True)
    thread.start()
    return job_id


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def stream_job_progress(job_id: str, poll_interval: float = 0.4):
    """
    Generator of SSE events tracking a job to completion.

    Emits `progress` whenever the state actually changes (not on every poll
    tick -- no point resending identical data), then one final `done` event.
    """
    import time

    last_sent = None
    while True:
        job = get_job(job_id)
        if job is None:
            yield _sse("error", {"message": f"Unknown job {job_id}"})
            return

        snapshot = json.dumps(job, sort_keys=True)
        if snapshot != last_sent:
            yield _sse("progress", job)
            last_sent = snapshot

        if job.get("status") in ("done", "failed"):
            yield _sse("done", job)
            return

        time.sleep(poll_interval)