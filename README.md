<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner.svg" alt="FLINT: Fixed-budget Late Interaction" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.27273"><img src="https://img.shields.io/badge/arXiv-2610.27273-c2580f" alt="arXiv: 2610.27273"></a>
  <a href="https://huggingface.co/collections/himankhanda/flint-6ad3ac31fafcb692ae534c3b"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20models-6%20checkpoints-2f6fb3" alt="Models on Hugging Face"></a>
  <img src="https://img.shields.io/badge/python-3.10-6e7781" alt="Python 3.10">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-3b8a3e" alt="License: MIT"></a>
</p>

<p align="center">Himank Handa · Negar Arabzadeh · Matei Zaharia</p>

FLINT is a late-interaction retriever that pools every query *and* every document into the same small,
fixed number of vector embeddings. Storage and search cost stop growing with text length, which matters most
for long, complex queries.

## How it works

The encoder gives one embedding per token. Attention-based pooling uses **V** learned latents to turn those
into V vectors, so a short query and a long document both end up as V vectors.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
  <img src="assets/architecture.svg" alt="Query and document token embeddings are pooled by attention-based pooling into V vectors each and compared with MaxSim" width="100%">
</picture>

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/highlights-dark.svg">
  <img src="assets/highlights.svg" alt="Fixed Budget: V embeddings per query or document; 5.3 to 9.3 times smaller index than ColBERTv2; 1.5 to 2.8 times faster than ConstBERT on long queries" width="100%">
</picture>

## Usage

Install the requirements for the backbone you want (`requirements/bert.txt` or
`requirements/modernbert.txt`) and `requirements/eval.txt` for evaluation.

Training:

```bash
python scripts/prepare_msmarco_bert.py --out-dir data/msmarco
python scripts/train_flint_bert.py --data-dir data/msmarco

python scripts/prepare_msmarco_bge.py --out-dir data/msmarco-en-bge
python scripts/train_flint_modernbert.py --data-dir data/msmarco-en-bge --latents 32
```

Evaluation:

```bash
python scripts/evaluate.py --checkpoint <checkpoint> --benchmarks beir/scifact crumb/paper
python scripts/measure_latency.py --checkpoint <checkpoint> --benchmark crumb/paper
```

The baselines and other analysis scripts are in `scripts/`. Trained checkpoints are in this
[Hugging Face collection](https://huggingface.co/collections/himankhanda/flint-6ad3ac31fafcb692ae534c3b).

## Acknowledgements

This code is built on [ColBERT](https://github.com/stanford-futuredata/ColBERT). It is released under
the MIT license; `scripts/train_tokenlevel_modernbert.py` is adapted from
[ModernBERT](https://github.com/AnswerDotAI/ModernBERT) under Apache 2.0. The ConstBERT baseline
follows [MacAvaney et al. (2025)](https://huggingface.co/pinecone/ConstBERT).
