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
TRAINING_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
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


def sample_train_data(train_df, fraction):
    if fraction == 1.0:
        return train_df.copy()

    sampled = train_df.groupby("label", group_keys=False).sample(
        frac=fraction,
        random_state=RANDOM_SEED,
    )

    return sampled.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def evaluate_model(model, data_loader, device):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in data_loader:
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
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"],
    }


def train_and_evaluate(train_subset, test_df, tokenizer, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )
    model.to(device)

    train_dataset = FinancialPhraseBankDataset(train_subset, tokenizer)
    test_dataset = FinancialPhraseBankDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_index % 50 == 0:
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS}, "
                    f"batch {batch_index}/{len(train_loader)}, "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

    training_time = time.time() - start_time
    metrics = evaluate_model(model, test_loader, device)

    return metrics, training_time


def save_graphs(results):
    labels = [
        f"{int(row['Training Fraction'] * 100)}%\n({int(row['Training Examples'])})"
        for _, row in results.iterrows()
    ]
    score_min = max(0, min(results["Accuracy"].min(), results["F1-score"].min()) - 0.03)
    score_max = min(1, max(results["Accuracy"].max(), results["F1-score"].max()) + 0.02)
    x_positions = list(range(len(results)))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
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
    ax.set_title("DistilBERT Scores by Training Data Size")
    ax.set_xlabel("Training data used")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(score_min, score_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "experiment_data_size_scores.png")
    plt.close(fig)

    plt.figure(figsize=(9, 5))
    plt.bar(labels, results["Training Time (seconds)"], color="#10b981")
    plt.title("Training Time by Dataset Size")
    plt.xlabel("Training data used")
    plt.ylabel("Training time (seconds)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_data_size_time.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        results["Training Time (seconds)"],
        results["F1-score"],
        s=90,
        color="#10b981",
    )
    for label, time_value, f1_value in zip(
        labels,
        results["Training Time (seconds)"],
        results["F1-score"],
    ):
        plt.annotate(label, (time_value, f1_value), textcoords="offset points", xytext=(8, 6))
    plt.title("Dataset Size: F1-score vs Training Time")
    plt.xlabel("Training time (seconds)")
    plt.ylabel("F1-score")
    plt.ylim(max(0, results["F1-score"].min() - 0.03), min(1, results["F1-score"].max() + 0.02))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_data_size_efficiency.png")
    plt.close()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = get_device()

    rows = []

    for fraction in TRAINING_FRACTIONS:
        train_subset = sample_train_data(train_df, fraction)

        print(
            f"Training DistilBERT with {int(fraction * 100)}% of training data "
            f"({len(train_subset)} examples).",
            flush=True,
        )

        metrics, training_time = train_and_evaluate(
            train_subset,
            test_df,
            tokenizer,
            device,
        )

        rows.append(
            {
                "Training Fraction": fraction,
                "Training Examples": len(train_subset),
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-score": metrics["f1_score"],
                "Training Time (seconds)": training_time,
            }
        )

        print(
            f"{int(fraction * 100)}% finished: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1_score']:.4f}, "
            f"time={training_time:.2f}s",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results_path = RESULTS_DIR / "experiment_data_size.csv"
    results.to_csv(results_path, index=False)
    save_graphs(results)

    print("Saved data-size experiment table to:", results_path)
    print("Saved score graph to:", RESULTS_DIR / "experiment_data_size_scores.png")
    print("Saved time graph to:", RESULTS_DIR / "experiment_data_size_time.png")
    print("Saved efficiency graph to:", RESULTS_DIR / "experiment_data_size_efficiency.png")


if __name__ == "__main__":
    main()
