from pathlib import Path
import json
import os
import time

import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

RANDOM_SEED = 42
TFIDF_NGRAMS = (1, 2)
TFIDF_MAX_FEATURES = 20000
LOGREG_C = 1.0
LOGREG_MAX_ITER = 2000

TARGET_NAMES = ["negative", "neutral", "positive"]
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)} sentences")
    print(f"Test:  {len(test_df)} sentences")
    print()

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=TFIDF_NGRAMS,
            max_features=TFIDF_MAX_FEATURES,
            lowercase=True,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )),
    ])

    train_start = time.time()
    pipeline.fit(train_df["sentence"].tolist(), train_df["label"].tolist())
    train_time = time.time() - train_start

    inference_start = time.time()
    predicted_labels = pipeline.predict(test_df["sentence"].tolist())
    inference_time = time.time() - inference_start

    memory_after = process.memory_info().rss
    memory_mb = (memory_after - memory_before) / (1024 * 1024)

    true_labels = test_df["label"].tolist()

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print(classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=TARGET_NAMES,
        zero_division=0,
    ))
    print()

    predictions = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    predictions["classical_baseline_predicted_label"] = predicted_labels
    predictions["classical_baseline_prediction"] = [LABEL_NAMES[label] for label in predicted_labels]

    predictions_path = RESULTS_DIR / "classical_baseline_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    vocabulary_size = len(pipeline.named_steps["tfidf"].vocabulary_)

    summary = {
        "model": "TF-IDF + Logistic Regression",
        "purpose": (
            "Cheap classical baseline. Reframes the transformer comparison: "
            "shows what non-transformer ML gives you for free, so the "
            "transformers' extra cost can be judged against a real floor."
        ),
        "hyperparameters": {
            "tfidf_ngram_range": list(TFIDF_NGRAMS),
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_sublinear_tf": True,
            "tfidf_lowercase": True,
            "logreg_C": LOGREG_C,
            "logreg_max_iter": LOGREG_MAX_ITER,
            "logreg_solver": "lbfgs",
            "random_state": RANDOM_SEED,
        },
        "vocabulary_size": vocabulary_size,
        "training_time_seconds": train_time,
        "inference_time_seconds": inference_time,
        "memory_mb": memory_mb,
        "num_train_examples": len(train_df),
        "num_test_examples": len(test_df),
        "accuracy": accuracy,
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "report": report,
    }

    summary_path = RESULTS_DIR / "classical_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Extended comparison CSV (additive — does NOT touch model_comparison.csv).
    existing_comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    classical_row = pd.DataFrame([{
        "Model": "TF-IDF + LogReg (baseline)",
        "Accuracy": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-score": report["weighted avg"]["f1-score"],
        "Time (seconds)": inference_time,
        "Memory (MB)": memory_mb,
    }])
    extended = pd.concat([existing_comparison, classical_row], ignore_index=True)
    extended_path = RESULTS_DIR / "model_comparison_extended.csv"
    extended.to_csv(extended_path, index=False)

    print("=" * 60)
    print("Classical baseline (TF-IDF + LogReg) summary:")
    print(f"  Accuracy:           {accuracy:.4f}")
    print(f"  F1 (weighted):      {report['weighted avg']['f1-score']:.4f}")
    print(f"  F1 (macro):         {report['macro avg']['f1-score']:.4f}")
    print(f"  Vocabulary size:    {vocabulary_size:,}")
    print(f"  Train time:         {train_time:.3f}s")
    print(f"  Inference time:     {inference_time:.4f}s ({len(test_df)} sentences)")
    print(f"  Memory delta:       {memory_mb:.2f} MB")
    print()
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved summary to:     {summary_path}")
    print(f"Saved extended table: {extended_path}")


if __name__ == "__main__":
    main()
