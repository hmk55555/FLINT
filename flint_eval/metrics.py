"""Retrieval metrics with pytrec_eval."""

_MEASURES = {
    "ndcg@10": "ndcg_cut_10",
    "ndcg@1000": "ndcg_cut_1000",
    "recall@5": "recall_5",
    "recall@100": "recall_100",
    "recall@1000": "recall_1000",
    "mrr@10": "recip_rank",
}


def evaluate(run, qrels, metrics):
    import pytrec_eval

    run = {str(q): {str(d): float(s) for d, s in docs.items()} for q, docs in run.items()}
    out = {}

    measures = {_MEASURES[m] for m in metrics if m != "mrr@10"}
    if measures:
        per_query = pytrec_eval.RelevanceEvaluator(qrels, measures).evaluate(run)
        for m in metrics:
            if m != "mrr@10":
                out[m] = _mean(v[_MEASURES[m]] for v in per_query.values())

    if "mrr@10" in metrics:
        # recip_rank uses the whole ranking, so cut it to 10 first.
        top10 = {q: dict(sorted(docs.items(), key=lambda x: -x[1])[:10]) for q, docs in run.items()}
        per_query = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(top10)
        out["mrr@10"] = _mean(v["recip_rank"] for v in per_query.values())

    return {m: out[m] for m in metrics}


def _mean(values):
    values = list(values)
    return round(sum(values) / max(len(values), 1), 4)


def max_pool_passages(run, parent_of):
    pooled = {}
    for qid, docs in run.items():
        best = {}
        for did, score in docs.items():
            parent = parent_of.get(did, did)
            if parent not in best or score > best[parent]:
                best[parent] = score
        pooled[qid] = best
    return pooled


def drop_self_matches(run):
    for qid, docs in run.items():
        docs.pop(qid, None)
    return run
