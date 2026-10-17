"""Evaluate a checkpoint on retrieval benchmarks."""

import argparse
import json
import os

import common  # noqa: F401

from flint_eval import benchmarks, metrics, plaid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint directory or Hugging Face id")
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--doc-maxlen", type=int, default=512)
    ap.add_argument("--query-maxlen", type=int, default=512)
    ap.add_argument("--model-variant", choices=["flint", "constbert", "colbert"],
                    help="Override the variant stored in the checkpoint")
    args = ap.parse_args()

    for name in args.benchmarks:
        bench = benchmarks.load(name, cache_dir=os.path.join(args.output_dir, "cache"))
        print(f"#> {name}: {len(bench.doc_ids):,} documents", flush=True)

        index_dir = os.path.join(args.output_dir, "indexes", name.replace("/", "-"))
        plaid.build_index(args.checkpoint, bench.doc_texts, index_dir, args.doc_maxlen, args.model_variant)
        searcher = plaid.load_searcher(args.checkpoint, index_dir, bench.doc_texts, args.query_maxlen,
                                       model_variant=args.model_variant)

        for split in bench.splits:
            run = plaid.search(searcher, split.query_ids, split.query_texts, bench.doc_ids)
            if bench.parent_of:
                run = metrics.max_pool_passages(run, bench.parent_of)
            if name.startswith("beir/"):
                run = metrics.drop_self_matches(run)

            scores = metrics.evaluate(run, split.qrels, split.metrics)
            print(f"#> {name} [{split.name}] {scores}", flush=True)
            out = os.path.join(args.output_dir, name.replace("/", "-"), f"{split.name}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump({"checkpoint": args.checkpoint, "benchmark": name, "split": split.name,
                           "doc_maxlen": args.doc_maxlen, "query_maxlen": args.query_maxlen,
                           "metrics": scores}, f, indent=2)


if __name__ == "__main__":
    main()
