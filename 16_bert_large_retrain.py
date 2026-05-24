"""
16_bert_large_retrain.py — BERT-large fine-tune with corrected hyperparameters.

The original 07_train_bert_large.py used learning rate 2e-5 with no
warmup, batch size 2, and 3 epochs. It collapsed to all-neutral
predictions (test accuracy 0.6137 = exactly the neutral fraction
278/453), strongly suggesting the training run never moved the
classifier head off the majority-class fixed point.

This script retrains BERT-large on the SAME Financial PhraseBank
training split with corrections recommended for large transformer
fine-tuning:

  - Learning rate: 5e-6 (down from 2e-5 -- gentler updates for 340M
    parameters; the original LR was likely too aggressive)
  - Linear warmup over the first 10% of training steps, then linear
    decay (lets the optimizer stabilize before applying full LR)
  - Epochs: 5 (up from 3 -- gives the lower LR more iterations to
    converge on the 1811-sentence training set)
  - Per-seed RNG isolation (per-seed torch.Generator for the DataLoader)
  - Same batch_size 2, max_length 128, gradient_checkpointing (memory)

What this experiment tests:
  - If BERT-large STILL collapses, the deck's "bigger isn't better"
    framing is robust ("we tried with corrected hyperparameters too").
  - If BERT-large now reaches ~95-97%, the framing needs nuance
    ("bigger CAN match -- at much higher cost").

Saves to NEW paths (does NOT modify any existing files):
  models/bert-large-finetuned-v2/
  results/bert_large_retrain_predictions.csv
  results/bert_large_retrain_training_history.csv
  results/bert_large_retrain_eval.json

Run:
  conda run --no-capture-output -n diplomska-nlp python 16_bert_large_retrain.py
"""

from pathlib import Path
import json
import os
import time

import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models") / "bert-large-finetuned-v2"

MODEL_NAME = "bert-large-cased"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42
LOG_EVERY_BATCHES = 100

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
TARGET_NAMES = ["negative", "neutral", "positive"]


class FinancialPhraseBankDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.encodings = tokenizer(
            dataframe["sentence"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(dataframe["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            predicted_labels.extend(predictions.cpu().tolist())
            true_labels.extend(batch["labels"].cpu().tolist())
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )
    return (
        total_loss / len(loader),
        accuracy_score(true_labels, predicted_labels),
        report,
        predicted_labels,
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print("Starting BERT-large fine-tune (corrected hyperparameters).", flush=True)
    print(f"  Model:         {MODEL_NAME}", flush=True)
    print(f"  Train size:    {len(train_df)}", flush=True)
    print(f"  Test size:     {len(test_df)}", flush=True)
    print(f"  LR:            {LEARNING_RATE}  (original was 2e-5)", flush=True)
    print(f"  Warmup ratio:  {WARMUP_RATIO}   (original was none)", flush=True)
    print(f"  Weight decay:  {WEIGHT_DECAY}", flush=True)
    print(f"  Epochs:        {NUM_EPOCHS}    (original was 3)", flush=True)
    print(f"  Batch size:    {BATCH_SIZE}", flush=True)
    print(flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )
    # Gradient checkpointing trades compute for memory; disable KV cache to
    # silence the incompatibility warning during training.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_dataset = FinancialPhraseBankDataset(train_df, tokenizer)
    test_dataset = FinancialPhraseBankDataset(test_df, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model.to(device)
    print(f"Device: {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"  Total steps:   {total_steps}", flush=True)
    print(f"  Warmup steps:  {warmup_steps}", flush=True)
    print(flush=True)

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    training_history = []
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

            if batch_index % LOG_EVERY_BATCHES == 0:
                print(
                    f"  Epoch {epoch}/{NUM_EPOCHS}, "
                    f"batch {batch_index}/{len(train_loader)}, "
                    f"loss {loss.item():.4f}, "
                    f"lr {scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        avg_train_loss = total_train_loss / len(train_loader)
        eval_loss, eval_accuracy, eval_report, _ = evaluate(model, test_loader, device)

        training_history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "eval_f1_weighted": eval_report["weighted avg"]["f1-score"],
            "eval_f1_macro": eval_report["macro avg"]["f1-score"],
        })
        print(
            f"Epoch {epoch} finished: "
            f"train_loss={avg_train_loss:.4f}, "
            f"eval_acc={eval_accuracy:.4f}, "
            f"eval_f1_w={eval_report['weighted avg']['f1-score']:.4f}, "
            f"eval_f1_m={eval_report['macro avg']['f1-score']:.4f}",
            flush=True,
        )
        print(flush=True)

    training_time = time.time() - start

    eval_start = time.time()
    eval_loss, eval_accuracy, eval_report, predicted_labels = evaluate(model, test_loader, device)
    eval_time = time.time() - eval_start
    memory_after = process.memory_info().rss

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    predictions = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    predictions["bert_large_retrain_predicted_label"] = predicted_labels
    predictions["bert_large_retrain_prediction"] = [LABEL_NAMES[label] for label in predicted_labels]
    predictions_path = RESULTS_DIR / "bert_large_retrain_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    history_path = RESULTS_DIR / "bert_large_retrain_training_history.csv"
    pd.DataFrame(training_history).to_csv(history_path, index=False)

    evaluation = {
        "model": "Fine-tuned BERT-large (corrected hyperparameters)",
        "model_name": MODEL_NAME,
        "purpose": (
            "Retrain BERT-large with corrected hyperparameters (lower LR + "
            "linear warmup) to test whether the original collapse-to-neutral "
            "was a training artifact or an inherent limitation."
        ),
        "hyperparameters": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "max_length": MAX_LENGTH,
            "random_state": RANDOM_SEED,
            "gradient_checkpointing": True,
        },
        "comparison_with_original_07": {
            "original_lr": 2e-5,
            "original_warmup": "none",
            "original_epochs": 3,
            "original_test_accuracy": 0.6137,
            "original_test_f1_weighted": 0.4668,
            "note": "Original collapsed to predicting all-neutral.",
        },
        "device": str(device),
        "eval_loss": eval_loss,
        "accuracy": eval_accuracy,
        "f1_weighted": eval_report["weighted avg"]["f1-score"],
        "f1_macro": eval_report["macro avg"]["f1-score"],
        "report": eval_report,
        "training_time_seconds": training_time,
        "eval_time_seconds": eval_time,
        "memory_mb": (memory_after - memory_before) / (1024 * 1024),
        "num_train_examples": len(train_df),
        "num_test_examples": len(test_df),
    }
    evaluation_path = RESULTS_DIR / "bert_large_retrain_eval.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print("=" * 60, flush=True)
    print("Fine-tuned BERT-large (retrain) summary:", flush=True)
    print(f"  Accuracy:       {eval_accuracy:.4f}", flush=True)
    print(f"  F1 weighted:    {eval_report['weighted avg']['f1-score']:.4f}", flush=True)
    print(f"  F1 macro:       {eval_report['macro avg']['f1-score']:.4f}", flush=True)
    print(f"  vs original:    {eval_accuracy - 0.6137:+.4f} accuracy delta", flush=True)
    print(f"  Training time:  {training_time:.1f}s", flush=True)
    print(f"  Memory delta:   {(memory_after - memory_before) / (1024 * 1024):.2f} MB", flush=True)
    print(flush=True)
    print(f"Saved model to:      {MODEL_DIR}", flush=True)
    print(f"Saved predictions:   {predictions_path}", flush=True)
    print(f"Saved history:       {history_path}", flush=True)
    print(f"Saved evaluation:    {evaluation_path}", flush=True)


if __name__ == "__main__":
    main()
