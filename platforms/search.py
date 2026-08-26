from platforms.embed import embed_texts
from platforms.vectorstore import _get_qdrant_client, QDRANT_COLLECTION
from platforms.lexical import _get_client as _get_meili_client, MEILI_INDEX

RRF_K = 60  # standard constant from the original Reciprocal Rank Fusion paper


def _dense_search(query_vec, pool_size):
    if query_vec is None:
        return []
    qc = _get_qdrant_client()
    hits = qc.query_points(collection_name=QDRANT_COLLECTION, query=query_vec, limit=pool_size).points
    return [
        {"id": h.id, "score": h.score, "file_id": h.payload.get("file_id"),
         "page": h.payload.get("page"), "text": h.payload.get("text")}
        for h in hits
    ]


def _lexical_search(query_text, pool_size):
    client = _get_meili_client()
    index = client.index(MEILI_INDEX)
    result = index.search(query_text, {"limit": pool_size, "showRankingScore": True})
    return [
        {"id": hit["id"], "score": hit.get("_rankingScore", 0.0),
         "file_id": hit.get("file_id"), "page": hit.get("page"), "text": hit.get("text")}
        for hit in result["hits"]
    ]


def _collect_meta(dense_results, lexical_results):
    meta = {}
    for r in dense_results:
        meta.setdefault(r["id"], {}).update(
            {"text": r["text"], "file_id": r["file_id"], "page": r["page"], "dense_score": r["score"]}
        )
    for r in lexical_results:
        meta.setdefault(r["id"], {}).update(
            {"text": r["text"], "file_id": r["file_id"], "page": r["page"], "lexical_score": r["score"]}
        )
    return meta


def _reciprocal_rank_fusion(dense_results, lexical_results, k=RRF_K):
    scores = {}
    for rank, r in enumerate(dense_results, start=1):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank)
    for rank, r in enumerate(lexical_results, start=1):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank)

    meta = _collect_meta(dense_results, lexical_results)
    merged = [
        {"chunk_id": cid, "fused_score": s, "dense_score": meta[cid].get("dense_score"),
         "lexical_score": meta[cid].get("lexical_score"), "file_id": meta[cid].get("file_id"),
         "page": meta[cid].get("page"), "text": meta[cid].get("text")}
        for cid, s in scores.items()
    ]
    merged.sort(key=lambda x: x["fused_score"], reverse=True)
    return merged


def _min_max_normalize(results):
    if not results:
        return {}
    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    return {r["id"]: (r["score"] - lo) / span for r in results}


def _weighted_sum_fusion(dense_results, lexical_results, dense_weight, lexical_weight):
    dense_norm = _min_max_normalize(dense_results)
    lexical_norm = _min_max_normalize(lexical_results)
    meta = _collect_meta(dense_results, lexical_results)

    all_ids = set(dense_norm) | set(lexical_norm)
    merged = []
    for cid in all_ids:
        fused = dense_weight * dense_norm.get(cid, 0.0) + lexical_weight * lexical_norm.get(cid, 0.0)
        merged.append(
            {"chunk_id": cid, "fused_score": fused, "dense_score": meta[cid].get("dense_score"),
             "lexical_score": meta[cid].get("lexical_score"), "file_id": meta[cid].get("file_id"),
             "page": meta[cid].get("page"), "text": meta[cid].get("text")}
        )
    merged.sort(key=lambda x: x["fused_score"], reverse=True)
    return merged


def hybrid_search(q, k=5, method="rrf", pool=20, dense_weight=0.5, lexical_weight=0.5):
    query_vec = embed_texts([q])[0]
    dense_results = _dense_search(query_vec, pool)
    lexical_results = _lexical_search(q, pool)

    if method == "weighted":
        merged = _weighted_sum_fusion(dense_results, lexical_results, dense_weight, lexical_weight)
    else:
        merged = _reciprocal_rank_fusion(dense_results, lexical_results)

    return {"query": q, "method": method, "results": merged[:k]}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m platforms.search "your question here" [k] [method]')
        sys.exit(1)

    q = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    method = sys.argv[3] if len(sys.argv) > 3 else "rrf"

    result = hybrid_search(q, k=k, method=method)
    print(f"\n=== {result['method']} search: {result['query']!r} ===")
    for r in result["results"]:
        print(
            f"  fused={r['fused_score']:.4f}  dense={r['dense_score']}  "
            f"lexical={r['lexical_score']}  file_id={r['file_id']}  page={r['page']}"
        )
        print(f"    {r['text']!r}")