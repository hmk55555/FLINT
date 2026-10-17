"""Mean response time and index size of a checkpoint."""

import argparse
import json
import os
import random
import statistics
import time

import common  # noqa: F401

from flint_eval import benchmarks, plaid


def timed(fn, sync):
    sync()
    start = time.perf_counter()
    out = fn()
    sync()
    return out, (time.perf_counter() - start) * 1000


def dir_size(path):
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--doc-maxlen", type=int, default=512)
    ap.add_argument("--query-maxlen", type=int, default=512)
    ap.add_argument("--model-variant", choices=["flint", "constbert", "colbert"])
    ap.add_argument("--num-queries", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ncells", type=int, default=8)
    args = ap.parse_args()

    import torch

    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    bench = benchmarks.load(args.benchmark, cache_dir=os.path.join(args.output_dir, "cache"))
    index_dir = os.path.join(args.output_dir, "indexes", args.benchmark.replace("/", "-"))
    plaid.build_index(args.checkpoint, bench.doc_texts, index_dir, args.doc_maxlen, args.model_variant)
    searcher = plaid.load_searcher(args.checkpoint, index_dir, bench.doc_texts, args.query_maxlen,
                                   ncells=args.ncells, model_variant=args.model_variant)

    queries = bench.splits[0].query_texts
    queries = random.Random(42).sample(queries, min(args.num_queries, len(queries)))

    for text in queries[:args.warmup]:
        searcher.dense_search(searcher.encode([text]), k=1000)

    encode_ms, search_ms = [], []
    for text in queries:
        enc, srch = [], []
        for _ in range(args.reps):
            Q, t_enc = timed(lambda: searcher.encode([text]), sync)
            _, t_srch = timed(lambda: searcher.dense_search(Q, k=1000), sync)
            enc.append(t_enc)
            srch.append(t_srch)
        encode_ms.append(statistics.median(enc))
        search_ms.append(statistics.median(srch))

    result = {
        "checkpoint": args.checkpoint, "benchmark": args.benchmark, "num_queries": len(queries),
        "ncells": args.ncells,
        "encode_ms": round(statistics.mean(encode_ms), 2),
        "search_ms": round(statistics.mean(search_ms), 2),
        "mrt_ms": round(statistics.mean(e + s for e, s in zip(encode_ms, search_ms)), 2),
        "index_size_mb": round(dir_size(index_dir) / 1e6, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    print(json.dumps(result, indent=2))
    out = os.path.join(args.output_dir, args.benchmark.replace("/", "-"), "latency.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
