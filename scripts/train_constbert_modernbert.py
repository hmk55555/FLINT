"""Train the ConstBERT baseline on ModernBERT-large."""

import argparse
import os

from common import count_lines, find_latest_checkpoint, require_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Directory with examples.jsonl, queries.train.tsv and collection.tsv")
    ap.add_argument("--output-root", default="experiments")
    ap.add_argument("--experiment", default="ConstBERT-modernbert-large-c32-d512")
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    ap.add_argument("--num-vectors", type=int, default=32, help="C: vectors per document")
    ap.add_argument("--nranks", type=int, default=4, help="Number of GPUs")
    ap.add_argument("--bsize", type=int, default=16, help="Global batch size (split across GPUs)")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--maxsteps", type=int, default=-1, help="-1 = one epoch over examples.jsonl")
    ap.add_argument("--warmup", type=int, default=-1, help="-1 = 5%% of maxsteps")
    ap.add_argument("--nway", type=int, default=32)
    ap.add_argument("--doc-maxlen", type=int, default=512)
    ap.add_argument("--query-maxlen", type=int, default=84)
    args = ap.parse_args()

    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert.infra.launcher import Launcher
    from colbert.training.training import train

    triples = os.path.join(args.data_dir, "examples.jsonl")
    queries = os.path.join(args.data_dir, "queries.train.tsv")
    collection = os.path.join(args.data_dir, "collection.tsv")
    require_files(triples, queries, collection)

    maxsteps = args.maxsteps
    if maxsteps <= 0:
        n_examples = count_lines(triples)
        maxsteps = max(1, n_examples // args.bsize)
    warmup = args.warmup if args.warmup >= 0 else max(1, int(0.05 * maxsteps))

    experiment_root = os.path.join(args.output_root, args.experiment)
    checkpoint = find_latest_checkpoint(experiment_root) or args.backbone
    resume = checkpoint != args.backbone

    with Run().context(RunConfig(nranks=args.nranks, experiment=args.experiment, root=args.output_root)):
        config = ColBERTConfig(
            checkpoint=checkpoint,
            model_name=args.backbone,
            model_variant="constbert",
            resume=resume,
            dim=128,
            doc_maxlen=args.doc_maxlen,
            doc_pad_to_maxlen=True,
            query_maxlen=args.query_maxlen,
            nway=args.nway,
            num_doc_latents=args.num_vectors,
            attend_to_mask_tokens=False,
            mask_punctuation=False,
            triples=triples,
            queries=queries,
            collection=collection,
            bsize=args.bsize,
            accumsteps=1,
            lr=args.lr,
            warmup=warmup,
            maxsteps=maxsteps,
            warmup_bert=None,
            use_ib_negatives=False,
            distillation_alpha=1.0,
            ignore_scores=False,
            normalize_kd_scores=True,
            amp=True,
            amp_dtype="bfloat16",
            similarity="cosine",
        )
        Launcher(train).launch(config, config.triples, config.queries, config.collection)


if __name__ == "__main__":
    main()
