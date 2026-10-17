"""PLAID indexing and search."""

import os


def _config(**kwargs):
    from colbert.infra import ColBERTConfig

    return ColBERTConfig(**{k: v for k, v in kwargs.items() if v is not None})


def build_index(checkpoint, doc_texts, index_dir, doc_maxlen, model_variant=None, bsize=128):
    from colbert.indexer import Indexer
    from colbert.infra import Run, RunConfig

    marker = os.path.join(index_dir, "checkpoint.txt")
    if os.path.exists(os.path.join(index_dir, "metadata.json")):
        if os.path.exists(marker) and open(marker).read() != checkpoint:
            raise ValueError(f"{index_dir} was built with another checkpoint; use a different --output-dir.")
        print(f"#> Using existing index {index_dir}", flush=True)
        return
    root, name = os.path.split(os.path.abspath(index_dir))
    # Empty documents give NaN vectors.
    doc_texts = [t if t.strip() else "." for t in doc_texts]
    with Run().context(RunConfig(nranks=1, root=root, index_root=root)):
        config = _config(nbits=2, doc_maxlen=doc_maxlen, index_bsize=bsize, model_variant=model_variant)
        Indexer(checkpoint=checkpoint, config=config).index(name=name, collection=doc_texts, overwrite=True)
    with open(marker, "w") as f:
        f.write(checkpoint)


def load_searcher(checkpoint, index_dir, doc_texts, query_maxlen, ncells=4, model_variant=None):
    from colbert.infra import Run, RunConfig
    from colbert.searcher import Searcher

    root, name = os.path.split(os.path.abspath(index_dir))
    with Run().context(RunConfig(nranks=1, root=root, index_root=root)):
        searcher = Searcher(index=name, checkpoint=checkpoint, collection=doc_texts, index_root=root,
                            config=_config(query_maxlen=query_maxlen, model_variant=model_variant))
    searcher.configure(ncells=ncells, centroid_score_threshold=0.4, ndocs=4096)
    return searcher


def search(searcher, query_ids, query_texts, doc_ids, k=1000):
    from tqdm import tqdm

    Q = searcher.encode(query_texts)
    run = {}
    for i, qid in enumerate(tqdm(query_ids, desc="search")):
        pids, _, scores = searcher.dense_search(Q[i:i + 1], k=k)
        run[qid] = {doc_ids[p]: float(s) for p, s in zip(pids, scores)}
    return run
