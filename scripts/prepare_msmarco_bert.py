"""Prepare the MS MARCO training data for BERT FLINT."""

import argparse
import csv
import os
import shutil

import numpy as np
import ujson


def download_examples(out_dir):
    from huggingface_hub import hf_hub_download

    dst = os.path.join(out_dir, "examples.64.json")
    if os.path.exists(dst):
        return dst
    path = hf_hub_download(repo_id="colbert-ir/colbertv2.0_msmarco_64way", filename="examples.json")
    shutil.copyfile(path, dst)
    return dst


def write_queries(out_dir):
    import ir_datasets

    dst = os.path.join(out_dir, "queries.train.tsv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for q in ir_datasets.load("msmarco-passage/train").queries_iter():
            w.writerow([q.query_id, q.text])
    return dst


def write_collection(out_dir):
    import ir_datasets

    dst = os.path.join(out_dir, "collection.tsv")
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for d in ir_datasets.load("msmarco-passage").docs_iter():
            w.writerow([d.doc_id, d.text])
    return dst


def convert_to_binary(json_path, bin_path, nway=64):
    with open(json_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    width = nway + 1
    mmap = np.memmap(bin_path, dtype="int32", mode="w+", shape=(total, width))
    with open(json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = ujson.loads(line)
            pids = [x[0] if isinstance(x, list) else x for x in row[1:]]
            vals = ([row[0]] + pids)[:width]
            vals += [-1] * (width - len(vals))
            mmap[i, :] = np.asarray(vals, dtype="int32")
    mmap.flush()
    del mmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    examples_json = download_examples(args.out_dir)
    write_queries(args.out_dir)
    write_collection(args.out_dir)
    convert_to_binary(examples_json, os.path.join(args.out_dir, "examples.64.bin"), nway=64)


if __name__ == "__main__":
    main()
