import os
from typing import List, Optional

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_BATCH_SIZE = 128  # texts per HTTP request, keeps requests reasonably sized


def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Embed a list of texts via a locally running Ollama server.

    Returns a list the same length and order as `texts`, one embedding
    vector per input. If a batch request fails (Ollama not running, model
    not pulled, etc.), the corresponding entries are None rather than
    raising -- callers can store partial results and keep going, the same
    way ingest.py's OCR fallback degrades rather than aborting the whole run.
    """
    if not texts:
        return []

    results: List[Optional[List[float]]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise ValueError(
                    f"expected {len(batch)} embeddings, got "
                    f"{len(embeddings) if embeddings else 0}"
                )
            results.extend(embeddings)
        except Exception as e:
            print(f"[embed-failed] batch of {len(batch)} texts: {e}")
            results.extend([None] * len(batch))

    return results