from pathlib import Path
import os
import time

from datasets import load_dataset

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline


def map_model_label(label):
    if label == "POSITIVE":
        return 2
    if label == "NEGATIVE":
        return 0
    raise ValueError(f"Unknown model label: {label}")


def map_finbert_label(label):
    if label == "positive":
        return 2
    if label == "negative":
        return 0
    if label == "neutral":
        return 1
    raise ValueError(f"Unknown FinBERT label: {label}")


def save_bar_chart(models, values, title, ylabel, output_path):
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, values, color=["#6b7280", "#2563eb"])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


dataset = load_dataset(
    "financial_phrasebank",
    "sentences_allagree",
    trust_remote_code=True,
)

print("Dataset loaded successfully:")
print(dataset)
print()

df = pd.DataFrame(dataset["train"])

print("First 5 rows:")
print(df.head())
print()

print("Label counts:")
print(df["label"].value_counts().sort_index())
print()

label_names = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

df["label_name"] = df["label"].map(label_names)

print("First 5 rows with label names:")
print(df.head())
print()

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print()

print("Train label counts:")
print(train_df["label_name"].value_counts())
print()

print("Test label counts:")
print(test_df["label_name"].value_counts())
print()

classifier = pipeline("sentiment-analysis")

single_sentence = "The company reported strong profits this quarter"
single_result = classifier(single_sentence)

print("Single sentence prediction:")
print(single_sentence)
print(single_result)
print()

sample = test_df.sample(20, random_state=42)

predictions = []

for text in sample["sentence"]:
    result = classifier(text)
    predictions.append(result[0]["label"])

print("Predictions for 20 test sentences:")
print(predictions)
print()

sample = sample.copy()
sample["model_prediction"] = predictions
sample["predicted_label"] = [map_model_label(prediction) for prediction in predictions]

print("Sample predictions with true labels:")
print(sample[["sentence", "label_name", "model_prediction", "predicted_label"]])
print()

true_labels = sample["label"].tolist()
predicted_labels = sample["predicted_label"].tolist()

accuracy = accuracy_score(true_labels, predicted_labels)

print("Accuracy on 20-sentence sample:", accuracy)
print()

finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

finbert_single_result = finbert(single_sentence)

print("FinBERT single sentence prediction:")
print(single_sentence)
print(finbert_single_result)
print()

predictions_finbert = []

for text in sample["sentence"]:
    result = finbert(text)
    predictions_finbert.append(result[0]["label"])

print("FinBERT predictions for 20 test sentences:")
print(predictions_finbert)
print()

sample["finbert_prediction"] = predictions_finbert
sample["finbert_predicted_label"] = [
    map_finbert_label(prediction) for prediction in predictions_finbert
]

print("FinBERT sample predictions with true labels:")
print(sample[["sentence", "label_name", "finbert_prediction", "finbert_predicted_label"]])
print()

predicted_labels_finbert = sample["finbert_predicted_label"].tolist()
accuracy_finbert = accuracy_score(true_labels, predicted_labels_finbert)

print("FinBERT accuracy on 20-sentence sample:", accuracy_finbert)
print()

print("Running both models on the full test dataset...")
print("Full test size:", len(test_df))
print()

process = psutil.Process(os.getpid())

predictions_general_full = []

memory_before = process.memory_info().rss
start_time = time.time()

for text in test_df["sentence"]:
    result = classifier(text)
    predictions_general_full.append(result[0]["label"])

end_time = time.time()
memory_after = process.memory_info().rss
time_general = end_time - start_time
memory_general = (memory_after - memory_before) / (1024 * 1024)

predicted_general_numeric = [
    map_model_label(prediction) for prediction in predictions_general_full
]

predictions_finbert_full = []

memory_before = process.memory_info().rss
start_time = time.time()

for text in test_df["sentence"]:
    result = finbert(text)
    predictions_finbert_full.append(result[0]["label"])

end_time = time.time()
memory_after = process.memory_info().rss
time_finbert = end_time - start_time
memory_finbert = (memory_after - memory_before) / (1024 * 1024)

predicted_finbert_numeric = [
    map_finbert_label(prediction) for prediction in predictions_finbert_full
]

true_labels_full = test_df["label"].tolist()

accuracy_general_full = accuracy_score(true_labels_full, predicted_general_numeric)
accuracy_finbert_full = accuracy_score(true_labels_full, predicted_finbert_numeric)

print("Full test results:")
print("General Model Accuracy:", accuracy_general_full)
print("FinBERT Accuracy:", accuracy_finbert_full)
print("General Model Time:", time_general)
print("FinBERT Time:", time_finbert)
print("General Model Memory Usage (MB):", memory_general)
print("FinBERT Memory Usage (MB):", memory_finbert)
print()

target_names = ["negative", "neutral", "positive"]

print("General Model Report:")
print(
    classification_report(
        true_labels_full,
        predicted_general_numeric,
        target_names=target_names,
        zero_division=0,
    )
)

print("FinBERT Report:")
print(
    classification_report(
        true_labels_full,
        predicted_finbert_numeric,
        target_names=target_names,
        zero_division=0,
    )
)

report_general = classification_report(
    true_labels_full,
    predicted_general_numeric,
    target_names=target_names,
    zero_division=0,
    output_dict=True,
)

report_finbert = classification_report(
    true_labels_full,
    predicted_finbert_numeric,
    target_names=target_names,
    zero_division=0,
    output_dict=True,
)

results = pd.DataFrame(
    {
        "Model": ["General Model", "FinBERT"],
        "Accuracy": [accuracy_general_full, accuracy_finbert_full],
        "Precision": [
            report_general["weighted avg"]["precision"],
            report_finbert["weighted avg"]["precision"],
        ],
        "Recall": [
            report_general["weighted avg"]["recall"],
            report_finbert["weighted avg"]["recall"],
        ],
        "F1-score": [
            report_general["weighted avg"]["f1-score"],
            report_finbert["weighted avg"]["f1-score"],
        ],
        "Time (seconds)": [time_general, time_finbert],
        "Memory (MB)": [memory_general, memory_finbert],
    }
)

print("Comparison table:")
print(results)
print()

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

predictions_output = test_df.copy()
predictions_output["general_prediction"] = predictions_general_full
predictions_output["general_predicted_label"] = predicted_general_numeric
predictions_output["finbert_prediction"] = predictions_finbert_full
predictions_output["finbert_predicted_label"] = predicted_finbert_numeric

predictions_path = results_dir / "predictions.csv"
predictions_output.to_csv(predictions_path, index=False)

print("Saved predictions to:", predictions_path)

results_path = results_dir / "model_comparison.csv"
results.to_csv(results_path, index=False)

print("Saved comparison table to:", results_path)
print()

models = results["Model"].tolist()
accuracy_values = results["Accuracy"].tolist()
f1_values = results["F1-score"].tolist()

accuracy_chart_path = results_dir / "accuracy_comparison.png"
f1_chart_path = results_dir / "f1_score_comparison.png"

save_bar_chart(
    models,
    accuracy_values,
    "Model Accuracy Comparison",
    "Accuracy",
    accuracy_chart_path,
)

save_bar_chart(
    models,
    f1_values,
    "Model F1-score Comparison",
    "F1-score",
    f1_chart_path,
)

print("Saved accuracy graph to:", accuracy_chart_path)
print("Saved F1-score graph to:", f1_chart_path)
