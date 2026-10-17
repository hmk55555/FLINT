# Adapted from examples/train_pylate.py of https://github.com/AnswerDotAI/ModernBERT
# Copyright 2024 onwards Answer.AI, LightOn, and contributors. Licensed under the Apache License,
# Version 2.0 (see licenses/ModernBERT-Apache-2.0.txt).
# Modified for multi-GPU training with our hyperparameters.

"""Train the token-level ColBERT baseline on ModernBERT-large with PyLate."""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-large")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--per-device-batch", type=int, default=4, help="Global batch = this x number of GPUs")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=2.0)
    args = ap.parse_args()

    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    from datasets import load_dataset
    from pylate import losses, models, utils
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    train = load_dataset("lightonai/ms-marco-en-bge", name="train")
    queries = load_dataset("lightonai/ms-marco-en-bge", name="queries")
    documents = load_dataset("lightonai/ms-marco-en-bge", name="documents")
    train.set_transform(utils.KDProcessing(queries=queries, documents=documents).transform)

    model = models.ColBERT(
        model_name_or_path=args.backbone,
        device=f"cuda:{local_rank}",
        model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        config_kwargs={"reference_compile": False},
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        fp16=False,
        logging_steps=50,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=2,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=losses.Distillation(model=model),
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )
    trainer.train()

    if trainer.is_world_process_zero():
        final = os.path.join(args.output_dir, "final")
        model.save_pretrained(final)
        with open(os.path.join(final, "training_state.json"), "w") as f:
            json.dump({"step": trainer.state.global_step, "maxsteps": trainer.state.max_steps,
                       "epoch": trainer.state.epoch, "finished": True}, f, indent=2)


if __name__ == "__main__":
    main()
