from pathlib import Path
import json
import os
import time

import pandas as pd
import psutil
from transformers import pipeline


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_NAME = "ProsusAI/finbert"


def map_finbert_label(label):
    if label == "positive":
        return 2
    if label == "negative":
        return 0
    if label == "neutral":
        return 1
    raise ValueError(f"Unknown FinBERT label: {label}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / "test.csv")

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    finbert = pipeline("sentiment-analysis", model=MODEL_NAME)

    predictions = []
    start_time = time.time()

    for text in test_df["sentence"]:
        result = finbert(text)
        predictions.append(result[0]["label"])

    end_time = time.time()
    memory_after = process.memory_info().rss

    output = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    output["finbert_prediction"] = predictions
    output["finbert_predicted_label"] = [
        map_finbert_label(prediction) for prediction in predictions
    ]

    output_path = RESULTS_DIR / "finbert_predictions.csv"
    output.to_csv(output_path, index=False)

    runtime = {
        "model": "FinBERT",
        "model_name": MODEL_NAME,
        "time_seconds": end_time - start_time,
        "memory_mb": (memory_after - memory_before) / (1024 * 1024),
        "num_examples": len(test_df),
    }

    runtime_path = RESULTS_DIR / "finbert_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    print("FinBERT predictions saved to:", output_path)
    print("FinBERT runtime saved to:", runtime_path)
    print("FinBERT time:", runtime["time_seconds"])
    print("FinBERT memory (MB):", runtime["memory_mb"])


if __name__ == "__main__":
    main()

