from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / "matplotlib_cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report


RESULTS_DIR = Path("results")
TARGET_NAMES = ["negative", "neutral", "positive"]
MODEL_ORDER = [
    "General Model",
    "Fine-tuned DistilBERT",
    "Fine-tuned BERT-large",
    "FinBERT",
]
DISPLAY_NAMES = {
    "General Model": "General",
    "Fine-tuned DistilBERT": "DistilBERT FT",
    "Fine-tuned BERT-large": "BERT-large",
    "FinBERT": "FinBERT",
}
BAR_COLORS = ["#6b7280", "#10b981", "#f59e0b", "#2563eb"]
BERT_LARGE_FALLBACK_EVAL_TIME_SECONDS = 15.66961121559143
BERT_LARGE_FALLBACK_MEMORY_MB = 384.1875


def save_bar_chart(models, values, title, ylabel, output_path, score_chart=False):
    chart_df = pd.DataFrame({"Model": models, "Value": values})
    chart_df = chart_df.dropna(subset=["Value"])

    plt.figure(figsize=(10, 5))
    colors = BAR_COLORS[: len(chart_df)]
    bars = plt.bar(chart_df["Model"], chart_df["Value"], color=colors)
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(ylabel)

    if score_chart:
        plt.ylim(0, 1)
        label_offset = 0.02
    else:
        max_value = chart_df["Value"].max()
        plt.ylim(0, max_value * 1.15)
        label_offset = max_value * 0.03

    for bar, value in zip(bars, chart_df["Value"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def weighted_metrics_from_report(report):
    return {
        "Accuracy": report["accuracy"],
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-score": report["weighted avg"]["f1-score"],
    }


def build_report_from_predictions(path, prediction_column):
    predictions_df = pd.read_csv(path)
    true_labels = predictions_df["label"].tolist()
    predicted_labels = predictions_df[prediction_column].tolist()

    return classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )


def build_general_row(reports):
    runtime = load_json(RESULTS_DIR / "general_runtime.json")
    metrics = weighted_metrics_from_report(reports["General Model"])

    return {
        "Model": "General Model",
        **metrics,
        "Time (seconds)": runtime["time_seconds"],
        "Memory (MB)": runtime["memory_mb"],
    }


def build_finetuned_distilbert_row():
    evaluation = load_json(RESULTS_DIR / "distilbert_finetuned_eval.json")
    metrics = weighted_metrics_from_report(evaluation["report"])

    return {
        "Model": "Fine-tuned DistilBERT",
        **metrics,
        "Time (seconds)": evaluation["eval_time_seconds"],
        "Memory (MB)": evaluation["memory_mb"],
    }


def build_bert_large_row():
    evaluation_path = RESULTS_DIR / "bert_large_finetuned_eval.json"

    if evaluation_path.exists():
        evaluation = load_json(evaluation_path)
        report = evaluation["report"]
        time_seconds = evaluation.get("eval_time_seconds")
        memory_mb = evaluation.get("memory_mb")
    else:
        report = build_report_from_predictions(
            RESULTS_DIR / "bert_large_finetuned_predictions.csv",
            "bert_large_finetuned_predicted_label",
        )
        time_seconds = BERT_LARGE_FALLBACK_EVAL_TIME_SECONDS
        memory_mb = BERT_LARGE_FALLBACK_MEMORY_MB

    metrics = weighted_metrics_from_report(report)

    return {
        "Model": "Fine-tuned BERT-large",
        **metrics,
        "Time (seconds)": time_seconds,
        "Memory (MB)": memory_mb,
    }


def build_finbert_row(reports):
    runtime = load_json(RESULTS_DIR / "finbert_runtime.json")
    metrics = weighted_metrics_from_report(reports["FinBERT"])

    return {
        "Model": "FinBERT",
        **metrics,
        "Time (seconds)": runtime["time_seconds"],
        "Memory (MB)": runtime["memory_mb"],
    }


def main():
    reports = load_json(RESULTS_DIR / "evaluation_reports.json")
    rows = [
        build_general_row(reports),
        build_finetuned_distilbert_row(),
        build_bert_large_row(),
        build_finbert_row(reports),
    ]

    results = pd.DataFrame(rows)
    results["Model"] = pd.Categorical(
        results["Model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    results = results.sort_values("Model")
    results["Model"] = results["Model"].astype(str)

    results_path = RESULTS_DIR / "model_comparison.csv"
    results.to_csv(results_path, index=False)

    models = results["Model"].tolist()
    graph_models = [DISPLAY_NAMES[model] for model in models]

    save_bar_chart(
        graph_models,
        results["Accuracy"].tolist(),
        "Model Accuracy Comparison",
        "Accuracy",
        RESULTS_DIR / "accuracy_comparison.png",
        score_chart=True,
    )

    save_bar_chart(
        graph_models,
        results["F1-score"].tolist(),
        "Model F1-score Comparison",
        "F1-score",
        RESULTS_DIR / "f1_score_comparison.png",
        score_chart=True,
    )

    save_bar_chart(
        graph_models,
        results["Time (seconds)"].tolist(),
        "Inference Time Comparison",
        "Time (seconds)",
        RESULTS_DIR / "time_comparison.png",
    )

    save_bar_chart(
        graph_models,
        results["Memory (MB)"].tolist(),
        "Memory Usage Comparison",
        "Memory (MB)",
        RESULTS_DIR / "memory_comparison.png",
    )

    save_bar_chart(
        graph_models,
        results["Accuracy"].tolist(),
        "Accuracy Comparison (All Models)",
        "Accuracy",
        RESULTS_DIR / "final_accuracy.png",
        score_chart=True,
    )

    save_bar_chart(
        graph_models,
        results["F1-score"].tolist(),
        "F1-score Comparison (All Models)",
        "F1-score",
        RESULTS_DIR / "final_f1.png",
        score_chart=True,
    )

    save_bar_chart(
        graph_models,
        results["Time (seconds)"].tolist(),
        "Inference Time Comparison",
        "Time (seconds)",
        RESULTS_DIR / "final_time.png",
    )

    print("Final comparison table:")
    print(results)
    print()
    print("Saved comparison table to:", results_path)
    print("Saved accuracy graph to:", RESULTS_DIR / "accuracy_comparison.png")
    print("Saved F1-score graph to:", RESULTS_DIR / "f1_score_comparison.png")
    print("Saved time graph to:", RESULTS_DIR / "time_comparison.png")
    print("Saved memory graph to:", RESULTS_DIR / "memory_comparison.png")
    print("Saved final accuracy graph to:", RESULTS_DIR / "final_accuracy.png")
    print("Saved final F1-score graph to:", RESULTS_DIR / "final_f1.png")
    print("Saved final time graph to:", RESULTS_DIR / "final_time.png")


if __name__ == "__main__":
    main()
