from pathlib import Path
import json
import os
import time

import pandas as pd
import psutil
from transformers import pipeline


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"


def map_bert_label(label):
    rating = int(label.split()[0])

    if rating == 1:
        return 0
    if rating in (2, 3):
        return 1
    if rating in (4, 5):
        return 2

    raise ValueError(f"Unknown BERT label: {label}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    test_df = pd.read_csv(DATA_DIR / "test.csv")

    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    bert_model = pipeline("sentiment-analysis", model=MODEL_NAME)

    test_sentence = "The company reported strong profits this quarter"
    print("BERT test sentence result:")
    print(bert_model(test_sentence))

    predictions = []
    start_time = time.time()

    for text in test_df["sentence"]:
        result = bert_model(text)
        predictions.append(result[0]["label"])

    end_time = time.time()
    memory_after = process.memory_info().rss

    output = test_df[["row_id", "sentence", "label", "label_name"]].copy()
    output["bert_prediction"] = predictions
    output["bert_predicted_label"] = [
        map_bert_label(prediction) for prediction in predictions
    ]

    output_path = RESULTS_DIR / "bert_predictions.csv"
    output.to_csv(output_path, index=False)

    runtime = {
        "model": "BERT",
        "model_name": MODEL_NAME,
        "time_seconds": end_time - start_time,
        "memory_mb": (memory_after - memory_before) / (1024 * 1024),
        "num_examples": len(test_df),
    }

    runtime_path = RESULTS_DIR / "bert_runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    print("BERT predictions saved to:", output_path)
    print("BERT runtime saved to:", runtime_path)
    print("BERT time:", runtime["time_seconds"])
    print("BERT memory (MB):", runtime["memory_mb"])


if __name__ == "__main__":
    main()
