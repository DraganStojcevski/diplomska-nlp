# Local Model Artifacts

Fine-tuned model binaries are not committed to GitHub because they are large.

Expected local folders:

- `models/distilbert-finetuned/` — produced by `05_train_distilbert.py`, read by `06_evaluate_finetuned_distilbert.py`
- `models/bert-large-finetuned/` — produced by `07_train_bert_large.py` (original recipe)
- `models/bert-large-finetuned-v2/` — produced by `16_bert_large_retrain.py` (corrected hyperparameters; intended to be trained on Colab)

The saved-prediction CSVs and summary JSONs under `results/` are sufficient for all downstream analysis scripts (evaluation, macro-F1 report, McNemar significance, error analysis). The model folders above are only needed if you want to re-run a script that loads a previously fine-tuned model rather than reading its saved predictions.
