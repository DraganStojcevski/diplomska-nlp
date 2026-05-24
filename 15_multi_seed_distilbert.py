"""
15_multi_seed_distilbert.py — Multi-seed DistilBERT FT for variance estimates.

Supplementary methodology experiment. Re-runs the fine-tuned DistilBERT
pipeline (same hyperparameters as 05_train_distilbert.py: 3 epochs, batch
8, lr 2e-5, max_length 128, full 1,811-sentence train set) across 5
random seeds. The single-seed headline result (accuracy 0.9713, seed 42)
in model_comparison.csv is a point estimate; this script reports mean ±
std across seeds so the headline number can be qualified honestly.

Does NOT save the trained models (only metrics) and does NOT touch any
existing files. New outputs only:

    results/multi_seed_distilbert.csv          (per-seed metrics)
    results/multi_seed_distilbert_summary.json (aggregate mean/std + per-seed)

Run:
  conda run --no-capture-output -n diplomska-nlp python 15_multi_seed_distilbert.py
"""

from pathlib import Path
import gc
import json
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SEEDS = [42, 1, 2, 3, 4]

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
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
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
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
    }


def train_one_seed(seed, train_df, test_df, tokenizer, device):
    torch.manual_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    train_dataset = FinancialPhraseBankDataset(train_df, tokenizer)
    test_dataset = FinancialPhraseBankDataset(test_df, tokenizer)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    train_time = time.time() - start

    metrics = evaluate(model, test_loader, device)
    metrics["seed"] = seed
    metrics["training_time_seconds"] = train_time

    del model, optimizer, train_loader, test_loader, train_dataset, test_dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = get_device()
    print(f"Device: {device}", flush=True)
    print()

    rows = []
    overall_start = time.time()
    for seed in SEEDS:
        print(f"[Seed {seed}] Starting...", flush=True)
        seed_start = time.time()
        metrics = train_one_seed(seed, train_df, test_df, tokenizer, device)
        elapsed = time.time() - seed_start
        rows.append(metrics)
        print(
            f"[Seed {seed}] Done in {elapsed:.1f}s — "
            f"acc={metrics['accuracy']:.4f}, "
            f"f1_w={metrics['f1_weighted']:.4f}, "
            f"f1_m={metrics['f1_macro']:.4f}",
            flush=True,
        )
        print()
    overall_time = time.time() - overall_start

    df = pd.DataFrame(rows)
    df = df[["seed", "accuracy", "f1_weighted", "f1_macro", "precision_weighted", "recall_weighted", "training_time_seconds"]]
    csv_path = RESULTS_DIR / "multi_seed_distilbert.csv"
    df.to_csv(csv_path, index=False)

    def agg(column):
        return {
            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
        }

    summary = {
        "model": "Fine-tuned DistilBERT (multi-seed)",
        "base_model_name": MODEL_NAME,
        "purpose": (
            "Estimate variance of the headline DistilBERT FT result across "
            "multiple random seeds so the single-seed number can be qualified honestly."
        ),
        "hyperparameters": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
        },
        "device": str(device),
        "seeds": SEEDS,
        "num_runs": len(rows),
        "total_time_seconds": overall_time,
        "accuracy": agg("accuracy"),
        "f1_weighted": agg("f1_weighted"),
        "f1_macro": agg("f1_macro"),
        "training_time_seconds": agg("training_time_seconds"),
        "per_seed": rows,
    }
    summary_path = RESULTS_DIR / "multi_seed_distilbert_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60, flush=True)
    print(f"Multi-seed DistilBERT (n={len(rows)}) summary:", flush=True)
    print(
        f"  Accuracy:    {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}  "
        f"(min {summary['accuracy']['min']:.4f}, max {summary['accuracy']['max']:.4f})",
        flush=True,
    )
    print(f"  F1 weighted: {summary['f1_weighted']['mean']:.4f} ± {summary['f1_weighted']['std']:.4f}", flush=True)
    print(f"  F1 macro:    {summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['std']:.4f}", flush=True)
    print(f"  Total time:  {overall_time:.1f}s", flush=True)
    print(flush=True)
    print(df.to_string(index=False), flush=True)
    print(flush=True)
    print(f"Saved per-seed: {csv_path}", flush=True)
    print(f"Saved summary:  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
