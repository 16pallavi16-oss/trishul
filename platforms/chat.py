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
OLLAMA_NUM_PREDICT = os.environ.get("OLLAMA_NUM_PREDICT")

THREAD_HISTORY_TURNS = 5  # how many prior Q&A pairs to include as context

# Token budget for conversation history. Rather than a hard turn count, older
# turns get summarised once the history exceeds this, so long conversations
# stay within the model's context window without silently losing what was
# discussed earlier.
HISTORY_TOKEN_BUDGET = 50
KEEP_RECENT_TURNS = 4  # most recent messages always kept verbatim, never summarised
THREAD_TTL_SECONDS = 60 * 60 * 24 * 7  # threads expire after 7 days of inactivity

_redis_client = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _thread_key(thread_id: str) -> str:
    return f"thread:{thread_id}"


def get_thread_history(thread_id: str, full: bool = False) -> List[Dict[str, str]]:
    """
    full=False returns only the most recent turns (the old fixed-window
    behaviour). full=True returns the entire stored thread, which is what
    build_budgeted_history needs -- otherwise older turns would be truncated
    away before the summariser ever got a chance to see them.
    """
    r = _get_redis()
    start = 0 if full else -THREAD_HISTORY_TURNS * 2
    raw = r.lrange(_thread_key(thread_id), start, -1)
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
    payload = {"model": CHAT_MODEL, "messages": messages, "stream": False}
    if OLLAMA_NUM_PREDICT:
        payload["options"] = {"num_predict": int(OLLAMA_NUM_PREDICT)}
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _build_citations(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    file_paths = _file_paths_for([r["file_id"] for r in retrieved])
    return [
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


def answer_question(question: str, thread_id: str, k: int = 5, method: str = "rrf") -> Dict[str, Any]:
    search_result = hybrid_search(question, k=k, method=method)
    retrieved = search_result["results"]

    if not retrieved:
        # Nothing came back from either search backend at all -- don't even
        # give the model a chance to fill the gap with outside knowledge.
        answer_text = NO_CONTEXT_ANSWER
    else:
        context_block = _build_context_block(retrieved)
        history = build_budgeted_history(thread_id)
        messages = _build_messages(question, context_block, history)
        answer_text = _call_ollama_chat(messages)

    append_to_thread(thread_id, "user", question)
    append_to_thread(thread_id, "assistant", answer_text)

    threads.save_message(thread_id, "user", question)
    assistant_message_id = threads.save_message(thread_id, "assistant", answer_text)

    citations = _build_citations(retrieved)

    return {
        "thread_id": thread_id,
        "message_id": assistant_message_id,
        "question": question,
        "answer": answer_text,
        "citations": citations,
    }


def _call_ollama_chat_stream(messages: List[Dict[str, str]], stats=None):
    """
    Yields answer tokens one at a time as Ollama produces them.

    Ollama's streaming mode returns newline-delimited JSON (one JSON object
    per line), NOT SSE -- so this reads lines and pulls the token out of
    each. Converting to SSE happens a layer up, in answer_question_stream.
    """
    payload = {"model": CHAT_MODEL, "messages": messages, "stream": True}
    if OLLAMA_NUM_PREDICT:
        payload["options"] = {"num_predict": int(OLLAMA_NUM_PREDICT)}

    with requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=120,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        # chunk_size=1 is deliberate: the default 512-byte buffer holds
        # tokens back until it fills, which defeats streaming entirely.
        for line in resp.iter_lines(chunk_size=1):
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                if stats is not None:
                    stats["eval_count"] = data.get("eval_count")
                    stats["prompt_eval_count"] = data.get("prompt_eval_count")
                break


def _sse(event: str, payload: Dict[str, Any]) -> str:
    """Formats one Server-Sent Event. The blank line at the end is what
    tells the client that this event is complete -- it is required."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def answer_question_stream(question: str, thread_id: str, k: int = 5, method: str = "rrf"):
    """
    Generator of SSE-formatted strings, in this event order:

        citations  -- sent once, BEFORE any token. Retrieval finishes before
                      generation starts, so the UI can render the source list
                      immediately instead of waiting for the full answer.
        token      -- sent many times, one per token from the model.
        done       -- sent once, carrying the fully assembled answer.
        error      -- sent instead of `done` if generation fails partway.

    Persistence deliberately happens only after the stream finishes, so a
    half-generated answer never gets written into the thread history.
    """
    search_result = hybrid_search(question, k=k, method=method)
    retrieved = search_result["results"]

    yield _sse("citations", {"citations": _build_citations(retrieved)})

    pieces: List[str] = []
    generation_stats = {}

    if not retrieved:
        # Same hard guard as the non-streaming path: never call the model
        # when there is no context for it to ground an answer in.
        pieces.append(NO_CONTEXT_ANSWER)
        yield _sse("token", {"token": NO_CONTEXT_ANSWER})
    else:
        context_block = _build_context_block(retrieved)
        history = build_budgeted_history(thread_id)
        messages = _build_messages(question, context_block, history)
        try:
            for token in _call_ollama_chat_stream(messages, generation_stats):
                pieces.append(token)
                yield _sse("token", {"token": token})
        except Exception as e:
            yield _sse("error", {"message": f"Generation failed: {e}"})
            return

    answer_text = "".join(pieces)

    append_to_thread(thread_id, "user", question)
    append_to_thread(thread_id, "assistant", answer_text)
    threads.save_message(thread_id, "user", question)
    assistant_message_id = threads.save_message(thread_id, "assistant", answer_text)

    yield _sse(
        "done",
        {
            "answer": answer_text,
            "message_id": assistant_message_id,
            "eval_count": generation_stats.get("eval_count", 0),
            "prompt_eval_count": generation_stats.get("prompt_eval_count", 0),
        },
    )


def _estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token for English text.

    Deliberately an approximation -- an exact count needs the model's own
    tokenizer, which would mean another dependency and another round trip
    for something that only needs to be good enough to decide "is this
    history getting too long." Erring high is safe; erring low risks
    overflowing the context window.
    """
    return len(text) // 4 + 1


def _history_tokens(history: List[Dict[str, str]]) -> int:
    return sum(_estimate_tokens(m["content"]) for m in history)


def _summarise_messages(messages: List[Dict[str, str]]) -> str:
    """Compress older turns while keeping a distinct note for every pair."""
    exchanges = []
    for i in range(0, len(messages), 2):
        question = messages[i]
        answer = messages[i + 1] if i + 1 < len(messages) else None
        exchanges.append(
            f"Exchange {i // 2 + 1}:\n"
            f"Question: {question['content']}\n"
            f"Answer: {answer['content'] if answer else '(no answer yet)'}"
        )
    transcript = "\n\n".join(exchanges)
    prompt = [
        {
            "role": "system",
            "content": (
                "Summarize the conversation as a numbered list with exactly one short "
                "bullet for EVERY exchange, in the same order. Do not omit an exchange "
                "or merge separate exchanges. Preserve each question's topic and its "
                "important answer facts, names, figures, and decisions. Keep each "
                "bullet concise. Return only the numbered list."
            ),
        },
        {"role": "user", "content": transcript},
    ]
    return _call_ollama_chat(prompt)


def _key_summary(thread_id: str) -> str:
    return f"thread:{thread_id}:summary"


def build_budgeted_history(thread_id: str) -> List[Dict[str, str]]:
    """
    Returns history that fits the token budget.

    Under budget: every turn, verbatim.
    Over budget: the most recent KEEP_RECENT_TURNS messages verbatim, with
    everything older replaced by a single summary message. The summary is
    cached in Redis so a long thread doesn't re-summarise on every single
    request -- only when older turns have actually grown since last time.
    """
    history = get_thread_history(thread_id, full=True)
    if _history_tokens(history) <= HISTORY_TOKEN_BUDGET:
        return history

    older, recent = history[:-KEEP_RECENT_TURNS], history[-KEEP_RECENT_TURNS:]
    if not older:
        return recent

    r = _get_redis()
    cache_key = _key_summary(thread_id)
    cached = r.hgetall(cache_key)

    # Re-summarise only if the number of older messages has changed since
    # the cached summary was made.
    if cached.get("version") == "2" and cached.get("covers") == str(len(older)):
        summary = cached["text"]
    else:
        summary = _summarise_messages(older)
        r.hset(
            cache_key,
            mapping={"text": summary, "covers": str(len(older)), "version": "2"},
        )
        r.expire(cache_key, THREAD_TTL_SECONDS)

    return [
        {"role": "system", "content": f"Summary of earlier conversation: {summary}"},
        *recent,
    ]
