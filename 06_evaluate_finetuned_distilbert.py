from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import pipeline


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models") / "distilbert-finetuned"

TARGET_NAMES = ["negative", "neutral", "positive"]

LABEL_TO_ID = {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2,
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def map_finetuned_label(label):
    if label in LABEL_TO_ID:
        return LABEL_TO_ID[label]

    normalized_label = label.lower()
    if normalized_label in LABEL_TO_ID:
        return LABEL_TO_ID[normalized_label]

    raise ValueError(f"Unknown fine-tuned DistilBERT label: {label}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / "test.csv")

    finetuned_distilbert = pipeline(
        "text-classification",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
    )

    predictions = []
    prediction_scores = []

    for text in test_df["sentence"]:
        result = finetuned_distilbert(text, truncation=True, max_length=128)
        predictions.append(result[0]["label"])
        prediction_scores.append(result[0]["score"])

    predicted_numeric = [map_finetuned_label(label) for label in predictions]
    true_labels = test_df["label"].tolist()

    accuracy = accuracy_score(true_labels, predicted_numeric)
    report = classification_report(
        true_labels,
        predicted_numeric,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print("Fine-tuned DistilBERT Accuracy:", accuracy)
    print()
    print("Fine-tuned DistilBERT Report:")
    print(
        classification_report(
            true_labels,
            predicted_numeric,
            labels=[0, 1, 2],
            target_names=TARGET_NAMES,
            zero_division=0,
        )
    )

    output = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    output["distilbert_finetuned_prediction"] = predictions
    output["distilbert_finetuned_predicted_label"] = predicted_numeric
    output["distilbert_finetuned_score"] = prediction_scores

    predictions_path = RESULTS_DIR / "distilbert_finetuned_loaded_predictions.csv"
    output.to_csv(predictions_path, index=False)

    evaluation = {
        "model": "Fine-tuned DistilBERT",
        "model_path": str(MODEL_DIR),
        "accuracy": accuracy,
        "report": report,
        "num_test_examples": len(test_df),
    }

    evaluation_path = RESULTS_DIR / "distilbert_finetuned_loaded_eval.json"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print("Saved loaded-model predictions to:", predictions_path)
    print("Saved loaded-model evaluation to:", evaluation_path)


if __name__ == "__main__":
    main()
