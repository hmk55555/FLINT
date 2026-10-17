"""Corpora, queries and relevance judgments."""

import json
import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BEIR = {
    "arguana": "beir/arguana",
    "climate-fever": "beir/climate-fever",
    "dbpedia-entity": "beir/dbpedia-entity/test",
    "fever": "beir/fever/test",
    "fiqa": "beir/fiqa/test",
    "hotpotqa": "beir/hotpotqa/test",
    "nfcorpus": "beir/nfcorpus/test",
    "nq": "beir/nq",
    "quora": "beir/quora/test",
    "scidocs": "beir/scidocs",
    "scifact": "beir/scifact/test",
    "trec-covid": "beir/trec-covid",
    "webis-touche2020": "beir/webis-touche2020/v2",
}

CRUMB = {
    "code": "code_retrieval",
    "stackexchange": "stack_exchange",
    "clinical": "clinical_trial",
    "paper": "paper_retrieval",
    "tot": "tip_of_the_tongue",
    "theorem": "theorem_retrieval",
    "legal": "legal_qa",
    "setops": "set_operation_entity_retrieval",
}

ALL_BENCHMARKS = (
    [f"beir/{d}" for d in BEIR]
    + ["msmarco-passage", "trec-tot-2025", "browsecomp-plus"]
    + [f"crumb/{t}" for t in CRUMB]
)

BEIR_METRICS = ["ndcg@10", "recall@100", "recall@1000"]
MSMARCO_METRICS = ["mrr@10", "recall@1000"]
TOT_METRICS = ["ndcg@10", "ndcg@1000", "recall@1000"]
BROWSECOMP_METRICS = ["ndcg@10", "recall@5", "recall@100", "recall@1000"]
CRUMB_METRICS = ["ndcg@10", "recall@100", "recall@1000"]

MAX_DOC_CHARS = 50_000


@dataclass
class Split:
    name: str
    query_ids: list
    query_texts: list
    qrels: dict
    metrics: list


@dataclass
class Benchmark:
    name: str
    doc_ids: list
    doc_texts: list
    splits: list
    parent_of: dict = field(default=None)


def load(name, cache_dir="cache"):
    if name.startswith("beir/"):
        bench = _load_beir(name)
    elif name == "msmarco-passage":
        bench = _load_msmarco()
    elif name == "trec-tot-2025":
        bench = _load_trec_tot()
    elif name == "browsecomp-plus":
        bench = _load_browsecomp(cache_dir)
    elif name.startswith("crumb/"):
        bench = _load_crumb(name)
    else:
        raise ValueError(f"Unknown benchmark {name!r}. Known: {', '.join(ALL_BENCHMARKS)}")

    bench.doc_texts = [(t or "")[:MAX_DOC_CHARS] for t in bench.doc_texts]
    for split in bench.splits:
        _drop_empty_queries(split)
    return bench


def title_dot_text(title, text):
    title, text = (title or "").strip(), (text or "").strip()
    return f"{title}. {text}" if title and text else (title or text)


def _normalize_space(text):
    return " ".join((text or "").split())


def _drop_empty_queries(split):
    keep = [i for i, t in enumerate(split.query_texts) if t and t.strip()]
    if len(keep) < len(split.query_ids):
        print(f"#> {split.name}: dropped {len(split.query_ids) - len(keep)} empty queries", flush=True)
        split.query_ids = [split.query_ids[i] for i in keep]
        split.query_texts = [split.query_texts[i] for i in keep]


def _irds_queries_and_qrels(ir_key, normalize=False):
    import ir_datasets

    ds = ir_datasets.load(ir_key)
    qrels = {}
    for r in ds.qrels_iter():
        qrels.setdefault(str(r.query_id), {})[str(r.doc_id)] = int(r.relevance)
    ids, texts = [], []
    for q in ds.queries_iter():
        text = getattr(q, "text", None) or getattr(q, "query", None) or ""
        ids.append(str(q.query_id))
        texts.append(_normalize_space(text) if normalize else text)
    return ids, texts, qrels


def _judged_only(ids, texts, qrels):
    keep = [i for i, q in enumerate(ids) if q in qrels]
    return [ids[i] for i in keep], [texts[i] for i in keep]


def _read_trec_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    return qrels


def _load_beir(name):
    import ir_datasets

    ir_key = BEIR[name[len("beir/"):]]
    ds = ir_datasets.load(ir_key)
    doc_ids, doc_texts = [], []
    for d in ds.docs_iter():
        doc_ids.append(str(d.doc_id))
        doc_texts.append(title_dot_text(getattr(d, "title", None), d.text))
    ids, texts, qrels = _irds_queries_and_qrels(ir_key)
    ids, texts = _judged_only(ids, texts, qrels)
    return Benchmark(name, doc_ids, doc_texts, [Split("test", ids, texts, qrels, BEIR_METRICS)])


def _load_msmarco():
    import ir_datasets

    doc_ids, doc_texts = [], []
    for d in ir_datasets.load("msmarco-passage").docs_iter():
        doc_ids.append(str(d.doc_id))
        doc_texts.append(d.text.strip())
    ids, texts, qrels = _irds_queries_and_qrels("msmarco-passage/dev/small", normalize=True)
    ids, texts = _judged_only(ids, texts, qrels)
    return Benchmark("msmarco-passage", doc_ids, doc_texts,
                     [Split("dev.small", ids, texts, qrels, MSMARCO_METRICS)])


TOT_TEST_DIR = os.path.join(REPO_ROOT, "data", "trec-tot-2025")


def _load_trec_tot():
    import ir_datasets

    doc_ids, doc_texts = [], []
    for d in ir_datasets.load("trec-tot/2025/train").docs_iter():
        body = getattr(d, "body", None) or getattr(d, "text", None)
        doc_ids.append(str(d.doc_id))
        doc_texts.append(title_dot_text(getattr(d, "title", None), body))

    splits = []
    for label in ("train", "dev1", "dev2", "dev3"):
        ids, texts, qrels = _irds_queries_and_qrels(f"trec-tot/2025/{label}", normalize=True)
        ids, texts = _judged_only(ids, texts, qrels)
        splits.append(Split(label, ids, texts, qrels, TOT_METRICS))

    ids, texts = [], []
    with open(os.path.join(TOT_TEST_DIR, "test-2025-queries.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            ids.append(str(row["query_id"]))
            texts.append(_normalize_space(row["query"]))
    qrels = _read_trec_qrels(os.path.join(TOT_TEST_DIR, "test-2025-qrel.txt"))
    ids, texts = _judged_only(ids, texts, qrels)
    splits.append(Split("test", ids, texts, qrels, TOT_METRICS))
    return Benchmark("trec-tot-2025", doc_ids, doc_texts, splits)


_BC_REPO = "https://github.com/texttron/BrowseComp-Plus.git"


def _load_browsecomp(cache_dir):
    import csv
    import subprocess
    import sys

    from datasets import load_dataset

    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    doc_ids, doc_texts = [], []
    for row in corpus:
        doc_ids.append(str(row.get("docid") or row.get("id") or row.get("_id") or ""))
        doc_texts.append(title_dot_text(row.get("title"), row.get("text") or row.get("contents")))

    repo = os.path.join(cache_dir, "BrowseComp-Plus")
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth=1", _BC_REPO, repo], check=True)
    queries_tsv = os.path.join(cache_dir, "browsecomp_queries.tsv")
    if not os.path.exists(queries_tsv):
        decrypted = os.path.join(cache_dir, "browsecomp_decrypted.jsonl")
        subprocess.run([sys.executable, os.path.join(repo, "scripts_build_index", "decrypt_dataset.py"),
                        "--output", decrypted, "--generate-tsv", queries_tsv], check=True)

    ids, texts = [], []
    with open(queries_tsv, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                ids.append(row[0].strip())
                texts.append(row[1].strip())
    splits = []
    for label, path in (("evidence", "qrel_evidence.txt"), ("gold", "qrel_golds.txt")):
        qrels = _read_trec_qrels(os.path.join(repo, "topics-qrels", path))
        split_ids, split_texts = _judged_only(ids, texts, qrels)
        splits.append(Split(label, split_ids, split_texts, qrels, BROWSECOMP_METRICS))
    return Benchmark("browsecomp-plus", doc_ids, doc_texts, splits)


def _load_crumb(name):
    from datasets import load_dataset

    task = CRUMB[name[len("crumb/"):]]
    corpus = load_dataset("jfkback/crumb", "passage_corpus", split=task)
    doc_ids, doc_texts, parent_of = [], [], {}
    for row in corpus:
        did = str(row["document_id"])
        doc_ids.append(did)
        doc_texts.append(row["document_content"])
        parent = row.get("parent_id")
        if parent not in (None, "", "None"):
            parent_of[did] = str(parent)

    queries = load_dataset("jfkback/crumb", "evaluation_queries", split=task)
    ids, texts, qrels, use_max_p = [], [], {}, False
    for row in queries:
        qid = str(row["query_id"])
        ids.append(qid)
        texts.append(row["query_content"])
        use_max_p = bool(row.get("use_max_p", False))
        for judgment in row.get("full_document_qrels" if use_max_p else "passage_qrels") or []:
            if judgment["label"] > 0:
                qrels.setdefault(qid, {})[str(judgment["id"])] = int(judgment["label"])
    ids, texts = _judged_only(ids, texts, qrels)
    return Benchmark(name, doc_ids, doc_texts, [Split("test", ids, texts, qrels, CRUMB_METRICS)],
                     parent_of=parent_of if use_max_p else None)
