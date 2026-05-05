from pathlib import Path
import json
import os
import time

import pandas as pd
import psutil
from transformers import pipeline


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def map_general_label(label):
    if label == "POSITIVE":
        return 2
    if label == "NEGATIVE":
        return 0
    raise ValueError(f"Unknown general model label: {label}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / "test.csv")

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    classifier = pipeline("sentiment-analysis", model=MODEL_NAME)

    predictions = []
    start_time = time.time()

    for text in test_df["sentence"]:
        result = classifier(text)
        predictions.append(result[0]["label"])

    end_time = time.time()
    memory_after = process.memory_info().rss

    output = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    output["general_prediction"] = predictions
    output["general_predicted_label"] = [
        map_general_label(prediction) for prediction in predictions
    ]

    output_path = RESULTS_DIR / "general_predictions.csv"
    output.to_csv(output_path, index=False)

    runtime = {
        "model": "General Model",
        "model_name": MODEL_NAME,
        "time_seconds": end_time - start_time,
        "memory_mb": (memory_after - memory_before) / (1024 * 1024),
        "num_examples": len(test_df),
    }

    runtime_path = RESULTS_DIR / "general_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    print("General model predictions saved to:", output_path)
    print("General model runtime saved to:", runtime_path)
    print("General model time:", runtime["time_seconds"])
    print("General model memory (MB):", runtime["memory_mb"])


if __name__ == "__main__":
    main()


