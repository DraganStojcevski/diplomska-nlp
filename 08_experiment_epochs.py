from pathlib import Path
import os
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
TARGET_NAMES = ["negative", "neutral", "positive"]

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


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


def evaluate_model(model, data_loader, device):
    model.eval()

    total_loss = 0
    true_labels = []
    predicted_labels = []

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
        "eval_loss": total_loss / len(data_loader),
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
    }


def save_graphs(results):
    score_min = max(0, min(results["Accuracy"].min(), results["F1-score"].min()) - 0.03)
    score_max = min(1, max(results["Accuracy"].max(), results["F1-score"].max()) + 0.02)
    epoch_labels = [f"Epoch {epoch}" for epoch in results["Epoch"]]
    x_positions = list(range(len(results)))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    accuracy_bars = ax.bar(
        [position - bar_width / 2 for position in x_positions],
        results["Accuracy"],
        width=bar_width,
        color="#2563eb",
        label="Accuracy",
    )
    f1_bars = ax.bar(
        [position + bar_width / 2 for position in x_positions],
        results["F1-score"],
        width=bar_width,
        color="#10b981",
        label="F1-score",
    )
    ax.set_title("DistilBERT Scores by Number of Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(epoch_labels)
    ax.set_ylim(score_min, score_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment_epochs_scores.png")
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.plot(results["Epoch"], results["Train Loss"], marker="o", label="Train loss")
    plt.plot(results["Epoch"], results["Eval Loss"], marker="o", label="Eval loss")
    plt.title("DistilBERT Loss by Number of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(results["Epoch"])
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_epochs_loss.png")
    plt.close()

    per_epoch_time = results["Training Time (seconds)"].diff()
    per_epoch_time.iloc[0] = results["Training Time (seconds)"].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(epoch_labels, per_epoch_time, color="#f59e0b")
    ax.set_title("Training Time per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment_epochs_time.png")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )

    train_dataset = FinancialPhraseBankDataset(train_df, tokenizer)
    test_dataset = FinancialPhraseBankDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    rows = []
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_train_loss = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()

            if batch_index % 50 == 0:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS}, "
                    f"batch {batch_index}/{len(train_loader)}, "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        metrics = evaluate_model(model, test_loader, device)
        training_time = time.time() - start_time
        train_loss = total_train_loss / len(train_loader)

        rows.append(
            {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Eval Loss": metrics["eval_loss"],
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1_score"],
                "Training Time (seconds)": training_time,
            }
        )

        print(
            f"Epoch {epoch} finished: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_score']:.4f}, "
            f"time={training_time:.2f}s",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "experiment_epochs.csv"
    results.to_csv(results_path, index=False)
    save_graphs(results)

    print("Saved epoch experiment table to:", results_path)
    print("Saved score graph to:", RESULTS_DIR / "experiment_epochs_scores.png")
    print("Saved loss graph to:", RESULTS_DIR / "experiment_epochs_loss.png")
    print("Saved time graph to:", RESULTS_DIR / "experiment_epochs_time.png")


if __name__ == "__main__":
    main()
