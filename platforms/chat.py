"""
Thread-aware RAG chat: retrieve relevant chunks, build a context-aware
prompt (including this thread's prior turns), call Ollama, return an
answer with structured citations.

Thread history lives in Redis -- a list per thread_id, each entry one
{"role", "content"} turn. This is the first real use of the Redis
container from docker-compose.yml.
"""
import os
import json
from typing import List, Dict, Any

import redis
import requests
import psycopg2

from .search import hybrid_search
from .vectorstore import DATABASE_URL
from . import threads

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1:8b-instruct-q4_K_M")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

THREAD_HISTORY_TURNS = 5  # how many prior Q&A pairs to include as context
THREAD_TTL_SECONDS = 60 * 60 * 24 * 7  # threads expire after 7 days of inactivity

_redis_client = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _thread_key(thread_id: str) -> str:
    return f"thread:{thread_id}"


def get_thread_history(thread_id: str) -> List[Dict[str, str]]:
    r = _get_redis()
    raw = r.lrange(_thread_key(thread_id), -THREAD_HISTORY_TURNS * 2, -1)
    return [json.loads(m) for m in raw]


def append_to_thread(thread_id: str, role: str, content: str) -> None:
    r = _get_redis()
    key = _thread_key(thread_id)
    r.rpush(key, json.dumps({"role": role, "content": content}))
    r.expire(key, THREAD_TTL_SECONDS)


def _file_paths_for(file_ids: List[int]) -> Dict[int, str]:
    """Looks up source_file.path for a set of file_ids, so citations can show
    an actual filename instead of a bare database id."""
    unique_ids = list({fid for fid in file_ids if fid is not None})
    if not unique_ids:
        return {}
    conn = psycopg2.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("SELECT id, path FROM source_file WHERE id = ANY(%s)", (unique_ids,))
        rows = cur.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}


def _build_context_block(results: List[Dict[str, Any]]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] (file_id={r['file_id']}, page={r['page']})\n{r['text']}")
    return "\n\n".join(lines)


NO_CONTEXT_ANSWER = "I don't know from this corpus."


def _build_messages(question: str, context_block: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a helpful assistant answering questions using ONLY the document "
        "excerpts provided below as context. Follow these rules strictly:\n"
        "1. Base your answer entirely on the provided excerpts. Do not use any "
        "outside knowledge, even if you happen to know the answer from elsewhere.\n"
        "2. Cite every claim inline using the excerpt's bracketed number, e.g. [1].\n"
        f'3. If the excerpts do not contain enough information to answer the '
        f'question, respond with exactly: "{NO_CONTEXT_ANSWER}" Do not guess, '
        f"speculate, or fill gaps with general knowledge."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)  # this thread's prior turns, so follow-ups have context
    messages.append({"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"})
    return messages


def _call_ollama_chat(messages: List[Dict[str, str]]) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": CHAT_MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer_question(question: str, thread_id: str, k: int = 5, method: str = "rrf") -> Dict[str, Any]:
    search_result = hybrid_search(question, k=k, method=method)
    retrieved = search_result["results"]

    if not retrieved:
        # Nothing came back from either search backend at all -- don't even
        # give the model a chance to fill the gap with outside knowledge.
        answer_text = NO_CONTEXT_ANSWER
    else:
        context_block = _build_context_block(retrieved)
        history = get_thread_history(thread_id)
        messages = _build_messages(question, context_block, history)
        answer_text = _call_ollama_chat(messages)

    append_to_thread(thread_id, "user", question)
    append_to_thread(thread_id, "assistant", answer_text)

    threads.save_message(thread_id, "user", question)
    threads.save_message(thread_id, "assistant", answer_text)

    file_paths = _file_paths_for([r["file_id"] for r in retrieved])

    citations = [
        {
            "index": i + 1,
            "chunk_id": r["chunk_id"],
            "file_id": r["file_id"],
            "file_path": file_paths.get(r["file_id"]),
            "page": r["page"],
            "text": r["text"],
            "fused_score": r["fused_score"],
        }
        for i, r in enumerate(retrieved)
    ]

    return {
        "thread_id": thread_id,
        "question": question,
        "answer": answer_text,
        "citations": citations,
    }