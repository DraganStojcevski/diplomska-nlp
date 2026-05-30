from pathlib import Path
import json
import os
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
VAL_FRACTION = 0.125

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


def evaluate(model, data_loader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
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
    return {
        "loss": total_loss / len(data_loader),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "report": report,
    }


def train_one_epoch(model, train_loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0

    for batch_index, batch in enumerate(train_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch_index % 50 == 0:
            print(
                f"  Epoch {epoch}/{total_epochs}, "
                f"batch {batch_index}/{len(train_loader)}, "
                f"loss {loss.item():.4f}",
                flush=True,
            )

    return total_loss / len(train_loader)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    print("Loading data...", flush=True)
    train_df_original = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_subset, val_subset = train_test_split(
        train_df_original,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=train_df_original["label"],
    )
    train_subset = train_subset.reset_index(drop=True)
    val_subset = val_subset.reset_index(drop=True)

    print(f"Original train.csv: {len(train_df_original)} sentences (untouched on disk).", flush=True)
    print(f"Subset train:       {len(train_subset)} sentences (used for training).", flush=True)
    print(f"Validation:         {len(val_subset)} sentences (held-out for hyperparameter selection).", flush=True)
    print(f"Test:               {len(test_df)} sentences (observation only during training).", flush=True)
    print(flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )

    train_dataset = FinancialPhraseBankDataset(train_subset, tokenizer)
    val_dataset = FinancialPhraseBankDataset(val_subset, tokenizer)
    test_dataset = FinancialPhraseBankDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model.to(device)
    print(f"Device: {device}", flush=True)
    print(flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    per_epoch = []
    best_val_accuracy = -1.0
    best_epoch = -1
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{NUM_EPOCHS}", flush=True)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, NUM_EPOCHS)
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        per_epoch.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_weighted": val_metrics["f1_weighted"],
            "val_f1_macro": val_metrics["f1_macro"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_weighted": test_metrics["f1_weighted"],
            "test_f1_macro": test_metrics["f1_macro"],
        })

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch

        print(
            f"  val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1w={val_metrics['f1_weighted']:.4f}",
            flush=True,
        )
        print(
            f"  test_acc (observation only)={test_metrics['accuracy']:.4f}",
            flush=True,
        )
        print(flush=True)

    total_time = time.time() - start
    best_epoch_metrics = per_epoch[best_epoch - 1]

    history_path = RESULTS_DIR / "validation_check_history.csv"
    pd.DataFrame(per_epoch).to_csv(history_path, index=False)

    summary = {
        "model": "Fine-tuned DistilBERT (validation-check)",
        "base_model_name": MODEL_NAME,
        "purpose": (
            "Verify the original hyperparameter choices (made on test) hold "
            "when selection is performed on a held-out validation set."
        ),
        "split": {
            "train": len(train_subset),
            "val": len(val_subset),
            "test": len(test_df),
            "val_fraction_of_original_train": VAL_FRACTION,
            "random_state": RANDOM_SEED,
        },
        "hyperparameters": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
        },
        "device": str(device),
        "training_time_seconds": total_time,
        "best_epoch_by_val": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy_at_best_epoch": best_epoch_metrics["test_accuracy"],
        "test_f1_weighted_at_best_epoch": best_epoch_metrics["test_f1_weighted"],
        "test_f1_macro_at_best_epoch": best_epoch_metrics["test_f1_macro"],
        "per_epoch": per_epoch,
    }

    summary_path = RESULTS_DIR / "validation_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60, flush=True)
    print("Validation-check summary:", flush=True)
    print(f"  Best epoch by validation accuracy: {best_epoch}", flush=True)
    print(f"  Validation accuracy at best epoch: {best_val_accuracy:.4f}", flush=True)
    print(f"  Test accuracy at chosen epoch:     {best_epoch_metrics['test_accuracy']:.4f}", flush=True)
    print(f"  Test weighted F1 at chosen epoch:  {best_epoch_metrics['test_f1_weighted']:.4f}", flush=True)
    print(f"  Test macro F1 at chosen epoch:     {best_epoch_metrics['test_f1_macro']:.4f}", flush=True)
    print(flush=True)
    print(f"Saved per-epoch history to: {history_path}", flush=True)
    print(f"Saved summary to:           {summary_path}", flush=True)


if __name__ == "__main__":
    main()
