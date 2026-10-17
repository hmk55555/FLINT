"""Train FLINT on BERT-base."""

import argparse
import os

from common import find_latest_checkpoint, require_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Directory with examples.64.bin, queries.train.tsv and collection.tsv")
    ap.add_argument("--output-root", default="experiments")
    ap.add_argument("--experiment", default="FLINT-bert-base")
    ap.add_argument("--backbone", default="bert-base-uncased")
    ap.add_argument("--nranks", type=int, default=2, help="Number of GPUs")
    ap.add_argument("--latents", type=int, default=32, help="Number of query and of document latents")
    ap.add_argument("--bsize", type=int, default=48, help="Global batch size (split across GPUs)")
    ap.add_argument("--accumsteps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--warmup-bert", type=int, default=10000,
                    help="Steps during which the backbone is frozen and only the FLINT head trains")
    ap.add_argument("--maxsteps", type=int, default=267000, help="Length of the LR schedule")
    ap.add_argument("--stop-at-step", type=int, default=154000,
                    help="Stop early without changing the LR schedule (0 = run all of --maxsteps)")
    ap.add_argument("--nway", type=int, default=64)
    ap.add_argument("--doc-maxlen", type=int, default=220)
    ap.add_argument("--query-maxlen", type=int, default=84)
    args = ap.parse_args()

    from colbert.infra import ColBERTConfig, Run, RunConfig
    from colbert.infra.launcher import Launcher
    from colbert.training.training import train

    triples = os.path.join(args.data_dir, "examples.64.bin")
    queries = os.path.join(args.data_dir, "queries.train.tsv")
    collection = os.path.join(args.data_dir, "collection.tsv")
    require_files(triples, queries, collection)

    experiment_root = os.path.join(args.output_root, args.experiment)
    checkpoint = find_latest_checkpoint(experiment_root) or args.backbone
    resume = checkpoint != args.backbone

    with Run().context(RunConfig(nranks=args.nranks, experiment=args.experiment, root=args.output_root)):
        config = ColBERTConfig(
            checkpoint=checkpoint,
            model_name=args.backbone,
            resume=resume,
            dim=128,
            triples=triples,
            queries=queries,
            collection=collection,
            doc_maxlen=args.doc_maxlen,
            query_maxlen=args.query_maxlen,
            nway=args.nway,
            bsize=args.bsize,
            accumsteps=args.accumsteps,
            lr=args.lr,
            warmup=args.warmup,
            warmup_bert=args.warmup_bert,
            maxsteps=args.maxsteps,
            stop_at_step=(args.stop_at_step or None),
            use_ib_negatives=True,
            distillation_alpha=1.0,
            amp=True,
            amp_dtype="float16",
            mask_punctuation=True,
            num_doc_latents=args.latents,
            num_query_latents=args.latents,
            perceiver_dropout=0.1,
        )
        Launcher(train).launch(config, config.triples, config.queries, config.collection)


if __name__ == "__main__":
    main()
