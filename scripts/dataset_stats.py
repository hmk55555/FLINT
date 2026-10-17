"""Benchmark sizes and mean token lengths."""

import argparse
import statistics

import common  # noqa: F401

from flint_eval import benchmarks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="bert-base-uncased")
    ap.add_argument("--cache-dir", default="results/cache")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def mean_len(texts):
        lengths = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
        return round(statistics.mean(lengths), 1)

    print(f"{'benchmark':28s} {'split':10s} {'documents':>10s} {'queries':>8s} {'doc len':>8s} {'query len':>9s}")
    for name in args.benchmarks:
        bench = benchmarks.load(name, cache_dir=args.cache_dir)
        doc_len = mean_len(bench.doc_texts)
        for split in bench.splits:
            print(f"{name:28s} {split.name:10s} {len(bench.doc_ids):>10,} {len(split.query_ids):>8,} "
                  f"{doc_len:>8} {mean_len(split.query_texts):>9}")


if __name__ == "__main__":
    main()
