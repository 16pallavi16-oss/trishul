"""
Sync chunks (text only -- no embeddings needed) into Meilisearch, for
keyword/lexical search alongside the semantic search in vectorstore.py.

Usage:
    python -m platforms.lexical sync
    python -m platforms.lexical query "your question here"
"""
import os
import sys

import psycopg2
import meilisearch
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create a .env file in the project root "
        "(see .env.example)."
    )

MEILI_URL = os.environ.get("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.environ.get("MEILI_MASTER_KEY")
MEILI_INDEX = "chunks"


def _get_client() -> meilisearch.Client:
    return meilisearch.Client(MEILI_URL, MEILI_MASTER_KEY)


def _get_pg_connection():
    return psycopg2.connect(DATABASE_URL)


def ensure_index(client: meilisearch.Client) -> None:
    """
    Create the index if it doesn't exist yet, and configure which fields are
    actually searched/filterable. Safe to call every run -- update_settings
    is idempotent.
    """
    try:
        client.get_index(MEILI_INDEX)
    except meilisearch.errors.MeilisearchApiError:
        task = client.create_index(MEILI_INDEX, {"primaryKey": "id"})
        client.wait_for_task(task.task_uid)

    index = client.index(MEILI_INDEX)
    task = index.update_settings(
        {
            "searchableAttributes": ["text"],
            "filterableAttributes": ["file_id", "page"],
        }
    )
    client.wait_for_task(task.task_uid)


def sync_chunks_to_meilisearch(batch_size: int = 500) -> int:
    """
    Pushes every chunk in Postgres into the Meilisearch index. Meilisearch's
    add_documents is an upsert keyed on `id` (the chunk's own primary key),
    so re-running this is safe and simply overwrites existing entries --
    unlike vectorstore.py's sync, there's no separate "already synced"
    tracking table needed here.
    """
    client = _get_client()
    ensure_index(client)
    index = client.index(MEILI_INDEX)

    conn = _get_pg_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT id, file_id, chunk_index, text, page FROM chunk")
        rows = cur.fetchall()
    conn.close()

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        docs = [
            {
                "id": chunk_id,
                "file_id": file_id,
                "chunk_index": chunk_index,
                "text": text,
                "page": page,
            }
            for chunk_id, file_id, chunk_index, text, page in batch
        ]
        task = index.add_documents(docs, primary_key="id")
        client.wait_for_task(task.task_uid)
        total += len(batch)
        print(f"[sync] {total}/{len(rows)} chunks indexed in Meilisearch")

    return total


def query(query_text: str, top_k: int = 5) -> None:
    client = _get_client()
    index = client.index(MEILI_INDEX)
    result = index.search(query_text, {"limit": top_k})

    print(f"\n=== Lexical query: {query_text!r} ===")
    for hit in result["hits"]:
        print(
            f"  file_id={hit.get('file_id')}  page={hit.get('page')}  "
            f"{hit.get('text', '')[:80]!r}"
        )
    if not result["hits"]:
        print("  (no matches)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m platforms.lexical sync\n"
            '  python -m platforms.lexical query "your question here"'
        )
        sys.exit(1)

    command = sys.argv[1]
    if command == "sync":
        n = sync_chunks_to_meilisearch()
        print(f"\nDone -- indexed {n} chunks in Meilisearch.")
    elif command == "query":
        if len(sys.argv) < 3:
            print('Usage: python -m platforms.lexical query "your question here"')
            sys.exit(1)
        query(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)