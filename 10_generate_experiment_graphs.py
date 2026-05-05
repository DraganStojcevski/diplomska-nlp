from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
COLORS = {
    "accuracy": "#2563eb",
    "f1": "#10b981",
    "loss": "#ef4444",
    "time": "#f59e0b",
    "examples": "#6b7280",
}


def score_ylim(*series):
    values = pd.concat([pd.Series(item) for item in series])
    lower = max(0, values.min() - 0.03)
    upper = min(1, values.max() + 0.02)
    return lower, upper


def add_point_labels(x_values, y_values, decimals=3):
    for x_value, y_value in zip(x_values, y_values):
        plt.annotate(
            f"{y_value:.{decimals}f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )


def save_score_bar_chart(labels, accuracy, f1_score, title, output_path, xlabel):
    score_min, score_max = score_ylim(accuracy, f1_score)
    x_positions = list(range(len(labels)))
    bar_width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))
    accuracy_bars = ax.bar(
        [position - bar_width / 2 for position in x_positions],
        accuracy,
        width=bar_width,
        color=COLORS["accuracy"],
        label="Accuracy",
    )
    f1_bars = ax.bar(
        [position + bar_width / 2 for position in x_positions],
        f1_score,
        width=bar_width,
        color=COLORS["f1"],
        label="F1-score",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(score_min, score_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(accuracy_bars, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(f1_bars, fmt="%.3f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_epoch_graphs():
    path = RESULTS_DIR / "experiment_epochs.csv"
    if not path.exists():
        print("Skipping epoch graphs. Missing:", path)
        return

    results = pd.read_csv(path)
    epochs = results["Epoch"]
    epoch_labels = [f"Epoch {epoch}" for epoch in epochs]

    save_score_bar_chart(
        epoch_labels,
        results["Accuracy"],
        results["F1-score"],
        "DistilBERT Scores by Number of Epochs",
        RESULTS_DIR / "experiment_epochs_scores.png",
        "Epoch",
    )
    save_score_bar_chart(
        epoch_labels,
        results["Accuracy"],
        results["F1-score"],
        "DistilBERT Scores by Number of Epochs",
        RESULTS_DIR / "experiment_epochs_scores_zoomed.png",
        "Epoch",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        results["Train Loss"],
        marker="o",
        linewidth=2,
        color=COLORS["loss"],
        label="Train loss",
    )
    plt.plot(
        epochs,
        results["Eval Loss"],
        marker="o",
        linewidth=2,
        color=COLORS["time"],
        label="Eval loss",
    )
    plt.title("DistilBERT Loss by Number of Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_epochs_loss.png")
    plt.close()

    per_epoch_time = results["Training Time (seconds)"].diff()
    per_epoch_time.iloc[0] = results["Training Time (seconds)"].iloc[0]

    for output_name in ["experiment_epochs_time.png", "experiment_epochs_time_bar.png"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(epoch_labels, per_epoch_time, color=COLORS["time"])
        ax.set_title("Training Time per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Seconds")
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f s", padding=3, fontsize=9)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / output_name)
        plt.close(fig)

    print("Saved improved epoch graphs.")


def save_data_size_graphs():
    path = RESULTS_DIR / "experiment_data_size.csv"
    if not path.exists():
        print("Skipping data-size graphs. Missing:", path)
        return

    results = pd.read_csv(path)
    labels = [
        f"{int(row['Training Fraction'] * 100)}%\n({int(row['Training Examples'])})"
        for _, row in results.iterrows()
    ]

    save_score_bar_chart(
        labels,
        results["Accuracy"],
        results["F1-score"],
        "DistilBERT Scores by Training Data Size",
        RESULTS_DIR / "experiment_data_size_scores.png",
        "Training data used",
    )
    save_score_bar_chart(
        labels,
        results["Accuracy"],
        results["F1-score"],
        "DistilBERT Scores by Training Data Size",
        RESULTS_DIR / "experiment_data_size_scores_zoomed.png",
        "Training data used",
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        labels,
        results["Training Time (seconds)"],
        color=COLORS["time"],
    )
    plt.title("Training Time by Dataset Size")
    plt.xlabel("Training data used")
    plt.ylabel("Training time (seconds)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_data_size_time_bar.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(
        results["Training Time (seconds)"],
        results["F1-score"],
        s=90,
        color=COLORS["f1"],
    )
    for label, time_value, f1_value in zip(
        labels,
        results["Training Time (seconds)"],
        results["F1-score"],
    ):
        plt.annotate(
            label.replace("\n", " "),
            (time_value, f1_value),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
        )
    plt.title("Dataset Size: F1-score vs Training Time")
    plt.xlabel("Training time (seconds)")
    plt.ylabel("F1-score")
    plt.ylim(*score_ylim(results["F1-score"]))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "experiment_data_size_efficiency.png")
    plt.close()

    print("Saved improved data-size graphs.")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    save_epoch_graphs()
    save_data_size_graphs()


if __name__ == "__main__":
    main()
