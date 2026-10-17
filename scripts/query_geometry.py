"""Query vectors, effective rank, and PLAID cells and candidates per query."""

import argparse
import json
import os
import random
import statistics

import common  # noqa: F401

from flint_eval import benchmarks, plaid


def effective_rank(Q):
    import torch

    s2 = torch.linalg.svdvals(Q.float()) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--doc-maxlen", type=int, default=512)
    ap.add_argument("--query-maxlen", type=int, default=512)
    ap.add_argument("--model-variant", choices=["flint", "constbert", "colbert"])
    ap.add_argument("--num-queries", type=int, default=150)
    ap.add_argument("--ncells", type=int, default=8)
    ap.add_argument("--num-vectors", type=int, help="Randomly keep this many vectors per query")
    args = ap.parse_args()

    bench = benchmarks.load(args.benchmark, cache_dir=os.path.join(args.output_dir, "cache"))
    index_dir = os.path.join(args.output_dir, "indexes", args.benchmark.replace("/", "-"))
    plaid.build_index(args.checkpoint, bench.doc_texts, index_dir, args.doc_maxlen, args.model_variant)
    searcher = plaid.load_searcher(args.checkpoint, index_dir, bench.doc_texts, args.query_maxlen,
                                   ncells=args.ncells, model_variant=args.model_variant)
    ranker, rng = searcher.ranker, random.Random(42)

    queries = bench.splits[0].query_texts
    queries = rng.sample(queries, min(args.num_queries, len(queries)))

    rows = []
    for text in queries:
        Q = searcher.encode([text])[0]
        Q = Q[Q.norm(dim=-1) > 1e-6]  # drop masked vectors
        if args.num_vectors and Q.size(0) > args.num_vectors:
            Q = Q[sorted(rng.sample(range(Q.size(0)), args.num_vectors))]
        Qs = Q.cuda().half() if ranker.use_gpu else Q
        cells, _ = ranker.get_cells(Qs, args.ncells)
        pids, _ = ranker.generate_candidates(searcher.config, Q.unsqueeze(0))
        rows.append({"vectors": Q.size(0), "effective_rank": effective_rank(Q),
                     "cells": int(cells.numel()), "candidates": int(pids.numel())})

    result = {"checkpoint": args.checkpoint, "benchmark": args.benchmark, "num_queries": len(rows),
              "ncells": args.ncells, "num_vectors": args.num_vectors}
    for key in ("vectors", "effective_rank", "cells", "candidates"):
        result[f"mean_{key}"] = round(statistics.mean(r[key] for r in rows), 2)
    print(json.dumps(result, indent=2))
    suffix = f"_{args.num_vectors}vec" if args.num_vectors else ""
    out = os.path.join(args.output_dir, args.benchmark.replace("/", "-"), f"query_geometry{suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({**result, "per_query": rows}, f, indent=2)


if __name__ == "__main__":
    main()
