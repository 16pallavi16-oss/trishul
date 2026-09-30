"""
Sync chunks + embeddings (already computed via Ollama and stored in Postgres
by db.py) into two vector stores side by side:
  - Qdrant, running as a server (see docker-compose.yml, port 6333)
  - pgvector, a native extension on the SAME Postgres container (the
    pgvector/pgvector:pg16 image ships it pre-built -- no manual install)

Usage:
    python -m platforms.vectorstore sync
    python -m platforms.vectorstore query "your question here"
"""
import os
import sys

# This package is named `platform`, colliding with the stdlib module of the
# same name -- see _compat.py. .embed is a local import so it's listed
# first by convention, but with the patch-in-place design in _compat.py,
# the actual order no longer matters.
from .embed import embed_texts

import psycopg2
from pgvector.psycopg2 import register_vector
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Create a .env file in the project root "
        "(see .env.example)."
    )

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "chunks"
EMBEDDING_DIM = 768  # nomic-embed-text


def _get_pg_connection():
    conn = psycopg2.connect(DATABASE_URL)
    # Must run before register_vector(): on a database where the extension
    # has never been enabled, register_vector() looks up the `vector` type
    # and fails immediately if it doesn't exist yet. This has to happen on
    # every connection (not just once in ensure_pgvector_table) since
    # compare_query() also needs a registered connection and may run
    # without sync/ensure_pgvector_table having been called first.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def _get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_pgvector_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunk_vector (
                chunk_id INT PRIMARY KEY REFERENCES chunk(id),
                embedding VECTOR({EMBEDDING_DIM})
            );
            """
        )
    conn.commit()


def ensure_qdrant_collection(qc: QdrantClient) -> None:
    if not qc.collection_exists(QDRANT_COLLECTION):
        qc.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def sync_chunks_to_vector_stores(batch_size: int = 256) -> int:
    """
    Copies every chunk that has an embedding and hasn't been synced yet
    (chunk.id not already present in chunk_vector) into both Qdrant and the
    pgvector mirror table. Returns the number of chunks synced.
    """
    conn = _get_pg_connection()
    ensure_pgvector_table(conn)
    qc = _get_qdrant_client()
    ensure_qdrant_collection(qc)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.file_id, c.chunk_index, c.text, c.page, c.embedding
            FROM chunk c
            LEFT JOIN chunk_vector cv ON cv.chunk_id = c.id
            WHERE c.embedding IS NOT NULL AND cv.chunk_id IS NULL
            """
        )
        rows = cur.fetchall()

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]

        qc.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=list(embedding),
                    payload={
                        "file_id": file_id,
                        "chunk_index": chunk_index,
                        "text": text,
                        "page": page,
                    },
                )
                for chunk_id, file_id, chunk_index, text, page, embedding in batch
            ],
        )

        with conn.cursor() as cur:
            for chunk_id, file_id, chunk_index, text, page, embedding in batch:
                cur.execute(
                    "INSERT INTO chunk_vector (chunk_id, embedding) VALUES (%s, %s) "
                    "ON CONFLICT (chunk_id) DO NOTHING",
                    (chunk_id, list(embedding)),
                )
        conn.commit()
        total += len(batch)
        print(f"[sync] {total}/{len(rows)} chunks written to Qdrant + pgvector")

    conn.close()
    return total


def compare_query(query_text: str, top_k: int = 5) -> None:
    """Embed a query and print top_k results from Qdrant and pgvector side by side."""
    embeddings = embed_texts([query_text])
    query_vec = embeddings[0]
    if query_vec is None:
        print("[error] could not embed query -- is Ollama running?")
        return

    qc = _get_qdrant_client()
    qdrant_hits = qc.query_points(
        collection_name=QDRANT_COLLECTION, query=query_vec, limit=top_k
    ).points

    conn = _get_pg_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.text, c.file_id, c.page, 1 - (cv.embedding <=> %s::vector) AS score
            FROM chunk_vector cv
            JOIN chunk c ON c.id = cv.chunk_id
            ORDER BY cv.embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vec, query_vec, top_k),
        )
        pg_hits = cur.fetchall()
    conn.close()

    print(f"\n=== Query: {query_text!r} ===")
    print(f"\n--- Qdrant top {top_k} ---")
    for h in qdrant_hits:
        print(
            f"  score={h.score:.4f}  file_id={h.payload.get('file_id')}  "
            f"page={h.payload.get('page')}  {h.payload.get('text', '')[:80]!r}"
        )

    print(f"\n--- pgvector top {top_k} ---")
    for text, file_id, page, score in pg_hits:
        print(f"  score={score:.4f}  file_id={file_id}  page={page}  {text[:80]!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m platforms.vectorstore sync\n"
            '  python -m platforms.vectorstore query "your question here"'
        )
        sys.exit(1)

    command = sys.argv[1]
    if command == "sync":
        n = sync_chunks_to_vector_stores()
        print(f"\nDone -- synced {n} chunks to Qdrant and pgvector.")
    elif command == "query":
        if len(sys.argv) < 3:
            print('Usage: python -m platforms.vectorstore query "your question here"')
            sys.exit(1)
        compare_query(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)