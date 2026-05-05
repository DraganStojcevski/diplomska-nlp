from pathlib import Path
import json
import os
import time

import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models") / "distilbert-finetuned"

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
MAX_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

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

    average_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )

    return average_loss, accuracy, report, predicted_labels


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )

    train_dataset = FinancialPhraseBankDataset(train_df, tokenizer)
    test_dataset = FinancialPhraseBankDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    training_history = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
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
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, "
                    f"batch {batch_index}/{len(train_loader)}, "
                    f"loss {loss.item():.4f}"
                )

        train_loss = total_train_loss / len(train_loader)
        eval_loss, eval_accuracy, eval_report, _ = evaluate_model(
            model,
            test_loader,
            device,
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "eval_f1_weighted": eval_report["weighted avg"]["f1-score"],
            }
        )

        print(
            f"Epoch {epoch + 1} finished: "
            f"train_loss={train_loss:.4f}, "
            f"eval_accuracy={eval_accuracy:.4f}, "
            f"eval_f1={eval_report['weighted avg']['f1-score']:.4f}"
        )

    training_time = time.time() - start_time

    eval_start_time = time.time()
    eval_loss, eval_accuracy, eval_report, predicted_labels = evaluate_model(
        model,
        test_loader,
        device,
    )
    eval_time = time.time() - eval_start_time
    memory_after = process.memory_info().rss

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    predictions = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    predictions["distilbert_finetuned_predicted_label"] = predicted_labels
    predictions["distilbert_finetuned_prediction"] = [
        LABEL_NAMES[label] for label in predicted_labels
    ]

    predictions_path = RESULTS_DIR / "distilbert_finetuned_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    history_path = RESULTS_DIR / "distilbert_finetuned_training_history.csv"
    pd.DataFrame(training_history).to_csv(history_path, index=False)

    evaluation = {
        "model": "Fine-tuned DistilBERT",
        "model_name": MODEL_NAME,
        "num_labels": NUM_LABELS,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": str(device),
        "eval_loss": eval_loss,
        "accuracy": eval_accuracy,
        "report": eval_report,
        "training_time_seconds": training_time,
        "eval_time_seconds": eval_time,
        "memory_mb": (memory_after - memory_before) / (1024 * 1024),
        "num_train_examples": len(train_df),
        "num_test_examples": len(test_df),
    }

    evaluation_path = RESULTS_DIR / "distilbert_finetuned_eval.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print()
    print("Fine-tuned DistilBERT training complete.")
    print("Accuracy:", eval_accuracy)
    print("Weighted F1:", eval_report["weighted avg"]["f1-score"])
    print("Training time:", training_time)
    print("Evaluation time:", eval_time)
    print("Memory (MB):", evaluation["memory_mb"])
    print()
    print("Saved model to:", MODEL_DIR)
    print("Saved predictions to:", predictions_path)
    print("Saved training history to:", history_path)
    print("Saved evaluation to:", evaluation_path)


if __name__ == "__main__":
    main()
