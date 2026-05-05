from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import classification_report


RESULTS_DIR = Path("results")
TARGET_NAMES = ["negative", "neutral", "positive"]


def build_report(true_labels, predicted_labels):
    return classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )


def print_report(model_name, true_labels, predicted_labels):
    print(f"{model_name} Report:")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1, 2],
            target_names=TARGET_NAMES,
            zero_division=0,
        )
    )


def main():
    general_df = pd.read_csv(RESULTS_DIR / "general_predictions.csv")
    finbert_df = pd.read_csv(RESULTS_DIR / "finbert_predictions.csv")
    bert_df = pd.read_csv(RESULTS_DIR / "bert_predictions.csv")

    predictions = general_df.merge(
        bert_df[["row_id", "bert_prediction", "bert_predicted_label"]],
        on="row_id",
        how="inner",
    ).merge(
        finbert_df[["row_id", "finbert_prediction", "finbert_predicted_label"]],
        on="row_id",
        how="inner",
    )

    predictions_path = RESULTS_DIR / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    true_labels = predictions["label"].tolist()
    general_labels = predictions["general_predicted_label"].tolist()
    bert_labels = predictions["bert_predicted_label"].tolist()
    finbert_labels = predictions["finbert_predicted_label"].tolist()

    print_report("General Model", true_labels, general_labels)
    print_report("BERT", true_labels, bert_labels)
    print_report("FinBERT", true_labels, finbert_labels)

    reports = {
        "General Model": build_report(true_labels, general_labels),
        "BERT": build_report(true_labels, bert_labels),
        "FinBERT": build_report(true_labels, finbert_labels),
    }

    reports_path = RESULTS_DIR / "evaluation_reports.json"
    reports_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")

    print("Combined predictions saved to:", predictions_path)
    print("Evaluation reports saved to:", reports_path)


if __name__ == "__main__":
    main()
