from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report


RESULTS_DIR = Path("results")
TARGET_NAMES = ["negative", "neutral", "positive"]
LABELS = [0, 1, 2]

# (display name, file, predicted-label column)
MODELS = [
    ("General Model", "general_predictions.csv", "general_predicted_label"),
    ("FinBERT", "finbert_predictions.csv", "finbert_predicted_label"),
    ("Fine-tuned DistilBERT", "distilbert_finetuned_loaded_predictions.csv", "distilbert_finetuned_predicted_label"),
    ("Fine-tuned BERT-large", "bert_large_finetuned_predictions.csv", "bert_large_finetuned_predicted_label"),
    ("TF-IDF + LogReg", "classical_baseline_predictions.csv", "classical_baseline_predicted_label"),
]

# Display order in the report: best macro F1 first, failures last.
DISPLAY_ORDER = [
    "Fine-tuned DistilBERT",
    "FinBERT",
    "TF-IDF + LogReg",
    "Fine-tuned BERT-large",
    "General Model",
]

SHORT_NAMES = {
    "Fine-tuned DistilBERT": "DistilBERT FT",
    "Fine-tuned BERT-large": "BERT-large FT",
    "General Model": "General",
    "TF-IDF + LogReg": "Classical",
    "FinBERT": "FinBERT",
}


def compute_report(predictions_path, prediction_column):
    df = pd.read_csv(predictions_path)
    return classification_report(
        df["label"],
        df[prediction_column],
        labels=LABELS,
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for name, filename, column in MODELS:
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skipping {name} -- missing {path}")
            continue
        report = compute_report(path, column)
        rows.append({
            "Model": name,
            "F1 weighted": report["weighted avg"]["f1-score"],
            "F1 macro": report["macro avg"]["f1-score"],
            "F1 negative": report["negative"]["f1-score"],
            "F1 neutral": report["neutral"]["f1-score"],
            "F1 positive": report["positive"]["f1-score"],
            "Gap (weighted - macro)": report["weighted avg"]["f1-score"] - report["macro avg"]["f1-score"],
        })

    df = pd.DataFrame(rows)
    df["__order"] = df["Model"].map(lambda m: DISPLAY_ORDER.index(m) if m in DISPLAY_ORDER else 999)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    csv_path = RESULTS_DIR / "macro_f1_comparison.csv"
    df.to_csv(csv_path, index=False)

    short = [SHORT_NAMES.get(m, m) for m in df["Model"]]

    # Chart 1: weighted vs macro grouped bars
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_positions = list(range(len(df)))
    bar_width = 0.36
    bars_weighted = ax.bar(
        [position - bar_width / 2 for position in x_positions],
        df["F1 weighted"],
        width=bar_width,
        color="#6b7280",
        label="F1 weighted",
    )
    bars_macro = ax.bar(
        [position + bar_width / 2 for position in x_positions],
        df["F1 macro"],
        width=bar_width,
        color="#d4a55a",
        label="F1 macro",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Weighted vs Macro F1 on the 453-sentence test set")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    ax.bar_label(bars_weighted, fmt="%.2f", padding=3, fontsize=8)
    ax.bar_label(bars_macro, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    chart_path = RESULTS_DIR / "macro_f1_comparison.png"
    fig.savefig(chart_path)
    plt.close(fig)

    # Chart 2: per-class F1 grouped bars
    fig, ax = plt.subplots(figsize=(11, 5.5))
    class_columns = ["F1 negative", "F1 neutral", "F1 positive"]
    class_colors = ["#dc2626", "#6b7280", "#10b981"]
    class_labels = ["Negative", "Neutral", "Positive"]
    bar_width = 0.27
    for index, (column, color, label) in enumerate(zip(class_columns, class_colors, class_labels)):
        offsets = [position + (index - 1) * bar_width for position in x_positions]
        bars = ax.bar(offsets, df[column], width=bar_width, color=color, label=label)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 — where each model wins or fails")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    per_class_path = RESULTS_DIR / "per_class_f1.png"
    fig.savefig(per_class_path)
    plt.close(fig)

    print("Macro-F1 report:")
    print(df.to_string(index=False))
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {chart_path}")
    print(f"Saved: {per_class_path}")


if __name__ == "__main__":
    main()
