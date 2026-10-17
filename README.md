# FLINT

Code for *FLINT: Fixed-budget Late Interaction* by Himank Handa, Negar Arabzadeh and Matei Zaharia.
FLINT is a late-interaction retriever that represents every query and document with a fixed number
of vectors.

Install the requirements for the backbone you want (`requirements/bert.txt` or
`requirements/modernbert.txt`) and `requirements/eval.txt` for evaluation. ModernBERT also needs
flash-attention 2.

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

This code is built on [ColBERT](https://github.com/stanford-futuredata/ColBERT). It is released under
the MIT license; `scripts/train_tokenlevel_modernbert.py` is adapted from
[ModernBERT](https://github.com/AnswerDotAI/ModernBERT) under Apache 2.0. The ConstBERT baseline
follows [MacAvaney et al. (2025)](https://huggingface.co/pinecone/ConstBERT).
