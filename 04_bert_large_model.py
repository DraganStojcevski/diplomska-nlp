from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "bert-large-cased"
NUM_LABELS = 3

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def load_bert_large_model():
    tokenizer_large = AutoTokenizer.from_pretrained(MODEL_NAME)

    model_large = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id=LABEL_TO_ID,
    )

    return tokenizer_large, model_large


def main():
    tokenizer_large, model_large = load_bert_large_model()

    print("BERT-large model loaded successfully.")
    print("Model name:", MODEL_NAME)
    print("Tokenizer:", tokenizer_large.__class__.__name__)
    print("Model:", model_large.__class__.__name__)
    print("Number of labels:", model_large.config.num_labels)
    print("Labels:", model_large.config.id2label)


if __name__ == "__main__":
    main()
