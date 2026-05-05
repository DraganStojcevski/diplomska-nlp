from pathlib import Path
import os
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix


RESULTS_DIR = Path("results")
LABELS = [0, 1, 2]
LABEL_NAMES = ["negative", "neutral", "positive"]

MODEL_FILES = [
    {
        "name": "FinBERT",
        "path": RESULTS_DIR / "finbert_predictions.csv",
        "prediction_column": "finbert_predicted_label",
        "output": RESULTS_DIR / "confusion_matrix_finbert.png",
    },
    {
        "name": "DistilBERT FT",
        "path": RESULTS_DIR / "distilbert_finetuned_loaded_predictions.csv",
        "prediction_column": "distilbert_finetuned_predicted_label",
        "output": RESULTS_DIR / "confusion_matrix_distilbert_ft.png",
    },
    {
        "name": "BERT-large",
        "path": RESULTS_DIR / "bert_large_finetuned_predictions.csv",
        "prediction_column": "bert_large_finetuned_predicted_label",
        "output": RESULTS_DIR / "confusion_matrix_bert_large.png",
    },
]


def save_confusion_matrix(model):
    if not model["path"].exists():
        print("Skipping confusion matrix. Missing:", model["path"])
        return

    data = pd.read_csv(model["path"])
    matrix = confusion_matrix(
        data["label"],
        data[model["prediction_column"]],
        labels=LABELS,
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(f"Confusion Matrix: {model['name']}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_yticklabels(LABEL_NAMES)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > matrix.max() / 2 else "black",
                fontsize=11,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(model["output"])
    plt.close(fig)
    print("Saved:", model["output"])


def save_architecture_diagram():
    steps = [
        "Financial\nPhraseBank",
        "Preprocessing\nTrain/Test Split",
        "Models\nBaseline, FinBERT,\nDistilBERT, BERT-large",
        "Fine-tuning\nDistilBERT + BERT-large",
        "Evaluation\nAccuracy, F1,\nTime, Memory",
        "Streamlit UI\nLive Demo + Results",
    ]

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    box_width = 0.14
    box_height = 0.42
    y = 0.45
    gap = 0.025

    for index, step in enumerate(steps):
        x = 0.02 + index * (box_width + gap)
        box = plt.Rectangle(
            (x, y),
            box_width,
            box_height,
            facecolor="#f8fafc",
            edgecolor="#2563eb",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + box_width / 2,
            y + box_height / 2,
            step,
            ha="center",
            va="center",
            fontsize=9,
        )

        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + box_width + gap * 0.75, y + box_height / 2),
                xytext=(x + box_width, y + box_height / 2),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.5},
            )

    ax.set_title("System Architecture and Experimental Pipeline", fontsize=14, pad=12)
    fig.tight_layout()
    output_path = RESULTS_DIR / "architecture_diagram.png"
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved:", output_path)


def save_main_summary_figure():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        print("Skipping main summary figure. Missing:", path)
        return

    data = pd.read_csv(path)
    display_names = {
        "General Model": "General",
        "Fine-tuned DistilBERT": "DistilBERT FT",
        "Fine-tuned BERT-large": "BERT-large",
        "FinBERT": "FinBERT",
    }
    data["Display Model"] = data["Model"].map(display_names).fillna(data["Model"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    plots = [
        ("Accuracy", "Accuracy", "#2563eb", 0, 1),
        ("F1-score", "F1-score", "#10b981", 0, 1),
        ("Time (seconds)", "Inference time (seconds)", "#f59e0b", 0, None),
        ("Memory (MB)", "Memory usage (MB)", "#ef4444", 0, None),
    ]

    for ax, (column, title, color, ymin, ymax) in zip(axes, plots):
        chart_data = data.dropna(subset=[column])
        bars = ax.bar(chart_data["Display Model"], chart_data[column], color=color)
        ax.set_title(title)
        ax.set_ylabel(column)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        if ymax is not None:
            ax.set_ylim(ymin, ymax)
        else:
            ax.set_ylim(ymin, chart_data[column].max() * 1.15)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    fig.suptitle("Main Experimental Results", fontsize=15)
    fig.tight_layout()
    output_path = RESULTS_DIR / "main_summary_figure.png"
    fig.savefig(output_path)
    plt.close(fig)
    print("Saved:", output_path)


def write_analysis_summary():
    summary = """
    # Analysis Summary

    ## Final Model Comparison

    The fine-tuned DistilBERT model and FinBERT achieved the strongest performance. Fine-tuned DistilBERT reached an accuracy and weighted F1-score of approximately 0.971, while FinBERT achieved approximately 0.969 on both metrics. This shows that a smaller model can become highly competitive when it is fine-tuned on the target dataset.

    BERT-large achieved lower performance in this experiment and predicted the majority class, neutral, for all test examples. This is an important finding: a larger model is not automatically better. Domain adaptation and task-specific fine-tuning are more important than parameter count alone.

    ## Epoch Experiment

    The epoch experiment shows that training for two epochs produced the best result. The third epoch did not improve performance and validation loss increased, which suggests that longer training can begin to overfit or become less useful after the model has already learned the task.

    ## Dataset Size Experiment

    The dataset-size experiment shows that performance improved strongly from 25% to 50% of the training data. After that point, the gains became smaller and were not strictly monotonic. This suggests that DistilBERT can learn efficiently from a moderate amount of task-specific data, but additional data does not always guarantee a proportional improvement.

    ## Limitations

    The dataset is relatively small and contains only one domain-specific sentiment task. BERT-large required significantly more computational resources and was trained under hardware constraints. The experiments used a limited set of hyperparameters, so the results should be interpreted as a controlled practical comparison rather than an exhaustive optimization study.

    ## Future Work

    Future work could use larger financial datasets, multilingual financial data, more systematic hyperparameter tuning, API deployment, and additional modern language models. A larger evaluation set would also make the conclusions more robust.
    """

    output_path = RESULTS_DIR / "analysis_summary.md"
    output_path.write_text(textwrap.dedent(summary).strip() + "\n", encoding="utf-8")
    print("Saved:", output_path)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    for model in MODEL_FILES:
        save_confusion_matrix(model)

    save_architecture_diagram()
    save_main_summary_figure()
    write_analysis_summary()


if __name__ == "__main__":
    main()
