from pathlib import Path

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def main():
    DATA_DIR.mkdir(exist_ok=True)

    dataset = load_dataset(
        "financial_phrasebank",
        "sentences_allagree",
        trust_remote_code=True,
    )

    df = pd.DataFrame(dataset["train"])
    df = df.reset_index().rename(columns={"index": "row_id"})
    df["label_name"] = df["label"].map(LABEL_NAMES)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print("Dataset loaded and split successfully.")
    print("Train size:", len(train_df))
    print("Test size:", len(test_df))
    print()
    print("Train label counts:")
    print(train_df["label_name"].value_counts())
    print()
    print("Test label counts:")
    print(test_df["label_name"].value_counts())
    print()
    print("Saved training data to:", DATA_DIR / "train.csv")
    print("Saved testing data to:", DATA_DIR / "test.csv")


if __name__ == "__main__":
    main()

