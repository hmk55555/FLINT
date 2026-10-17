"""Prepare the MS MARCO training data (BGE-M3 scores) for the ModernBERT models."""

import argparse
import os

DATASET = "lightonai/ms-marco-en-bge"


def clean(s):
    return s.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def convert(out_dir, limit=0, nway=32):
    import ujson
    from datasets import load_dataset

    os.makedirs(out_dir, exist_ok=True)
    train = load_dataset(DATASET, name="train", split="train")
    if limit:
        train = train.select(range(min(limit, len(train))))

    needed = {d for row in train for d in row["document_ids"][:nway]}
    query_text = {row["query_id"]: clean(row["text"]) for row in load_dataset(DATASET, name="queries", split="train")}

    docid2pid = {}
    with open(os.path.join(out_dir, "collection.tsv"), "w") as f:
        for row in load_dataset(DATASET, name="documents", split="train", streaming=True):
            did = row["document_id"]
            if did in needed and did not in docid2pid:
                docid2pid[did] = len(docid2pid)
                f.write(f"{docid2pid[did]}\t{clean(row['text'])}\n")
                if len(docid2pid) == len(needed):
                    break

    n_examples, seen_qids = 0, set()
    with open(os.path.join(out_dir, "examples.jsonl"), "w") as ef, \
            open(os.path.join(out_dir, "queries.train.tsv"), "w") as qf:
        for row in train:
            dids, scores = row["document_ids"][:nway], row["scores"][:nway]
            if len(dids) != nway or row["query_id"] not in query_text or any(d not in docid2pid for d in dids):
                continue
            qid = int(row["query_id"])
            ef.write(ujson.dumps([qid] + [[docid2pid[d], float(s)] for d, s in zip(dids, scores)]) + "\n")
            n_examples += 1
            if qid not in seen_qids:
                qf.write(f"{qid}\t{query_text[row['query_id']]}\n")
                seen_qids.add(qid)

    print(f"#> {n_examples:,} examples, {len(seen_qids):,} queries, {len(docid2pid):,} passages -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="Only use the first N training queries")
    args = ap.parse_args()
    convert(args.out_dir, limit=args.limit)


if __name__ == "__main__":
    main()
