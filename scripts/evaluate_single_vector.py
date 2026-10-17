"""Evaluate a single-vector (sentence-transformers) model."""

import argparse
import json
import os

import common  # noqa: F401

from flint_eval import benchmarks, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--output-dir", default="results-single-vector")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--bsize", type=int, default=64)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_length

    def encode(texts, prefix):
        return model.encode([prefix + t for t in texts], batch_size=args.bsize, convert_to_tensor=True,
                            normalize_embeddings=True, show_progress_bar=True).half()

    for name in args.benchmarks:
        bench = benchmarks.load(name, cache_dir=os.path.join(args.output_dir, "cache"))
        cached = os.path.join(args.output_dir, "embeddings", args.model.replace("/", "-"), name.replace("/", "-") + ".pt")
        if os.path.exists(cached):
            D = torch.load(cached).to(model.device)
        else:
            D = encode(bench.doc_texts, args.doc_prefix)
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            torch.save(D.cpu(), cached)
        for split in bench.splits:
            Q = encode(split.query_texts, args.query_prefix)
            run = {}
            for start in range(0, Q.size(0), 256):
                scores, idx = (Q[start:start + 256] @ D.T).float().topk(min(1000, D.size(0)), dim=1)
                for qid, s_row, i_row in zip(split.query_ids[start:start + 256], scores.tolist(), idx.tolist()):
                    run[qid] = {bench.doc_ids[i]: s for i, s in zip(i_row, s_row)}
            if bench.parent_of:
                run = metrics.max_pool_passages(run, bench.parent_of)
            if name.startswith("beir/"):
                run = metrics.drop_self_matches(run)

            scores = metrics.evaluate(run, split.qrels, split.metrics)
            print(f"#> {name} [{split.name}] {scores}", flush=True)
            out = os.path.join(args.output_dir, name.replace("/", "-"), f"{split.name}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump({"model": args.model, "benchmark": name, "split": split.name,
                           "metrics": scores}, f, indent=2)
        del D
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
