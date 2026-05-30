# Project Files — A Beginner's Guide

A plain-language walkthrough of every file in this project. If you're new to NLP, Python, or machine learning, start here.

---

## What does this project ask?

Can a computer read a financial sentence and tell whether it's good news, bad news, or just normal information? And which kind of model is best — a small model, a big model, a specialized financial model, or a simple word-counting approach?

To answer it, the project tries six different model approaches on the **same** sentences and compares them carefully.

---

## The simplest possible glossary

- **Sentiment** — the feeling or opinion expressed in text. "Profits doubled!" is positive. "Lawsuit filed" is negative. "The meeting is on Tuesday" is neutral.
- **Model** — a trained program that makes predictions. You give it text, it gives you a label.
- **Dataset** — a collection of labeled examples used for training and testing the models.
- **Train / training** — showing a model thousands of examples until it learns the pattern.
- **Fine-tuning** — taking a model that already understands general language and teaching it your specific task on top of that.
- **Hyperparameter** — a choice you make *before* training, like how many times to go through the data or how big a step to take when learning.
- **Accuracy** — what percent of predictions were correct.
- **F1 score** — a more careful version of accuracy that handles unbalanced data (some classes are rare).
- **Macro F1** — the average of the F1 scores per class, treating each class equally.
- **Weighted F1** — the average of the F1 scores per class, with bigger weight for bigger classes.
- **Confusion matrix** — a grid showing, for each true class, what the model predicted. The diagonal cells are correct predictions; off-diagonal cells are mistakes.
- **Statistical significance** — whether a difference between two models is real or could be due to chance.

---

## The dataset we use

**Financial PhraseBank** — 2,264 short financial sentences from real news, hand-labeled by experts as negative, neutral, or positive. We use the version where all experts agreed (`sentences_allagree`). The split is:

- **1,811 sentences** for training (80%)
- **453 sentences** for testing (20%)

Every model uses the **same** training set and is tested on the **same** test set, so the comparisons are fair.

---

## Folders

- `data/` — the dataset, already split into `train.csv` and `test.csv`.
- `models/` — fine-tuned model files (too big for git, kept locally).
- `results/` — everything the scripts produce: predictions, tables, charts, summaries.

---

## The main pipeline — files `01_` through `11_`

These produce the original headline numbers and charts.

### `01_data.py`

Reads the Financial PhraseBank dataset, mixes it well, and splits it into `data/train.csv` (1,811 sentences for teaching the models) and `data/test.csv` (453 sentences for grading them). After this, every other script reads the same two files, which is why all comparisons in the project are fair.

### `02_general_model.py`

Loads a *generic* sentiment model — one that was trained on movie/product reviews, not finance. Runs it on all 453 test sentences and saves the predictions. Spoiler: it does badly. Why? Because it can only output "positive" or "negative" — it doesn't know the "neutral" class even exists, and most of the test sentences are neutral.

### `03_finbert_model.py`

Loads **FinBERT**, a sentiment model that was specifically pretrained on financial text. Runs it on the test set. This is the "domain expert" baseline — it should do well because it speaks finance.

### `04_bert_model.py`

Loads a multilingual BERT model that was pretrained to predict movie ratings as 1–5 stars. We map 1 star → negative, 2–3 stars → neutral, 4–5 stars → positive, then run it on the test set. Another non-financial baseline used for comparison.

### `04_bert_large_model.py`

Just a quick smoke test that confirms BERT-large (the big model) can be loaded on this computer. Doesn't predict anything yet; later scripts actually train it.

> Some numbers have two scripts at the same prefix (like `04_bert_model.py` and `04_bert_large_model.py`). The number reflects the *stage* in the pipeline narrative, not the strict execution order. For the exact run order, see the README.

### `05_train_distilbert.py`

**Fine-tunes** DistilBERT (a small, fast transformer — 66 million parameters) on the financial training set. After running, DistilBERT has been taught the specific patterns of financial sentiment. The trained model is saved to `models/distilbert-finetuned/`. Takes a few minutes on a normal laptop.

### `05_evaluation.py`

Reads the prediction CSV files from scripts `02`, `03`, and `04`, computes accuracy / precision / recall / F1 / per-class scores for each model, and saves a combined report (`evaluation_reports.json`) plus a merged predictions table (`predictions.csv`).

### `06_evaluate_finetuned_distilbert.py`

Loads the fine-tuned DistilBERT from disk (the one `05_train_distilbert.py` saved) and tests it again on the test set, to confirm it still works after saving and reloading. Saves a fresh `distilbert_finetuned_loaded_predictions.csv`.

### `06_results.py`

Builds the **final comparison table** (`model_comparison.csv`) and the headline bar charts (accuracy, F1, time, memory). This is the "scoreboard" of the original four models.

### `07_train_bert_large.py`

Fine-tunes BERT-large (a much bigger transformer — 340 million parameters, ~5× the size of DistilBERT). Saves the trained model. This original run uses the *default* learning rate (2e-5) — which, surprisingly, made the model collapse to predicting only "neutral" for every test sentence. The collapse is preserved as the original finding; script `16_bert_large_retrain.py` later shows it was a hyperparameter problem.

### `08_experiment_epochs.py`

**Question: how many times should we go through the training data?** Trains DistilBERT for 1, then 2, then 3 epochs (passes through the data) and records the accuracy after each. Conclusion: **2 epochs is the sweet spot** — after 3, the model starts to overfit (memorize the training data rather than learn the pattern).

### `09_experiment_data_size.py`

**Question: how much training data does DistilBERT really need?** Trains it on 25%, 50%, 75%, and 100% of the training data. Conclusion: **the biggest jump is 25% → 50%**, after that the gains shrink.

### `10_generate_experiment_graphs.py`

Reads the CSVs that `08_experiment_epochs.py` and `09_experiment_data_size.py` produced and creates prettier, presentation-ready versions of the charts.

### `11_generate_presentation_assets.py`

Produces the final visuals: **confusion matrices** (which classes does each model confuse with which?), an architecture diagram of the project pipeline, and a one-page summary figure.

---

## The supplementary experiments — files `12_` through `17_`

These were added later to strengthen the methodology. Each one tests a specific concern about the original headline numbers. **None of them change the original results** — they only add new files to `results/`. This is the "additive" principle: original numbers remain comparable, new analyses sit alongside.

### `12_validation_check.py`

> **Concern:** the epoch sweep in `08_` picked "2 epochs is best" by looking at the test set. Strictly, that's peeking — you should not use your test set to tune choices.

This script holds out 12.5% of the training data as a *validation* set, picks the best epoch using **only validation accuracy** (never touching test for the selection), and confirms the original choice. Conclusion: epoch 2 still wins on the held-out validation set — the original methodology was actually defensible.

### `13_classical_baseline.py`

> **Concern:** all the models we compared are transformers. What does a simple, non-transformer approach get?

Trains **TF-IDF + Logistic Regression** — essentially "count which words appear, run a simple equation, predict the class". It's almost instant. Result: 86.5% accuracy, ~700× faster inference than DistilBERT and ~60× lower memory. This is the floor against which the transformers should be judged honestly.

### `14_macro_f1_report.py`

> **Concern:** weighted F1 is dominated by the majority class (neutral). It can hide problems.

This script computes **macro F1** (which treats every class equally) and per-class F1 (one number per class) for every model. Reveals: the original BERT-large run scored 0 on negative and 0 on positive — it really was a total collapse, just hidden by the weighted-F1 number.

### `15_multi_seed_distilbert.py`

> **Concern:** the headline DistilBERT number comes from one lucky training run with seed 42. How much would it vary if we used different random seeds?

Re-trains DistilBERT 5 times with 5 different random seeds. Result: **96.16 ± 0.62% accuracy** across 5 seeds. The original 97.13% headline turns out to be on the lucky side of the distribution. More importantly: 0.62% is *wider* than the 0.2% gap between DistilBERT and FinBERT, so those two are statistically a tie within seed noise.

### `16_bert_large_retrain.py`

> **Concern:** BERT-large collapsed in `07_train_bert_large.py`. Was that the model's fault, or the training recipe's fault?

Retrains BERT-large with three changes: a **4× lower learning rate (5e-6 instead of 2e-5)**, a **linear warmup** at the start, and **5 epochs instead of 3**. Result: **98.5% accuracy** — the highest in the entire study. So the original collapse was a training artifact (a bad learning rate for a big model), not a property of BERT-large itself.

> Because BERT-large is too heavy for a laptop, this one is intended to be run on Google Colab with a free T4 GPU (~10–15 minutes). The output JSON is the only thing you need to copy back.

### `17_significance_and_errors.py`

> **Concern:** are the gaps between models statistically real, or could they be noise?

Runs **McNemar's test** (the standard statistical test for paired classifiers on the same test set) on every interesting pair. Two conclusions:

1. The three top transformers (BERT-large retrain, DistilBERT FT, FinBERT) are **statistically tied** with each other — the test set is too small (n = 453) to separate them.
2. All three significantly outperform the classical baseline (p < 0.001 for every pair).

Also pulls out the test sentences that even the three best models all get wrong — only 2 of them — those are the genuinely hard cases.

---

## Support files

- **`requirements.txt`** — the Python packages this project depends on (pandas, torch, transformers, scikit-learn, scipy, matplotlib, datasets, psutil).
- **`README.md`** — top-level project intro, setup, run order, results table, references.
- **`THESIS_DRAFT.md`** / **`.docx`** — the full thesis write-up, including the layered methodology chapter (§11) that documents every supplementary experiment with tables and findings.

---

## What gets saved in `results/`

Each script writes specific files into the `results/` folder. The most important ones:

| File | What it contains |
|---|---|
| `model_comparison.csv` | The headline scoreboard. One row per model; columns for accuracy, precision, recall, F1, time, memory. |
| `model_comparison_extended.csv` | Same as above plus the classical baseline row. |
| `*_predictions.csv` | For each model: the test sentence, true label, predicted label, and (sometimes) the confidence. |
| `*_summary.json` | Clean summary numbers for each run. |
| `validation_check_summary.json` | The held-out-validation per-epoch results plus the best-epoch decision. |
| `multi_seed_distilbert_summary.json` | Per-seed and aggregate (mean ± std) DistilBERT results. |
| `bert_large_retrain_eval.json` | The corrected BERT-large result (98.5%) from Colab. |
| `mcnemar_results.csv` | Pairwise statistical-significance tests with p-values and verdicts. |
| `macro_f1_comparison.csv` | Per-model weighted F1, macro F1, and per-class F1. |
| `error_analysis.csv` | Test sentences misclassified by at least one of the top three transformers. |
| `confusion_matrix_*.png` | One per model — a 3×3 grid showing what was predicted vs the true label. |
| `final_accuracy.png`, `final_f1.png`, `final_time.png` | The headline bar charts. |
| `experiment_epochs_*`, `experiment_data_size_*` | Charts for the two original controlled experiments. |
| `analysis_summary.md` | Auto-generated narrative summary. |

---

## How the scripts depend on each other (mental map)

```
01_data.py
   │  (writes data/train.csv, data/test.csv)
   ▼
   ├──► 02_general_model.py     ┐
   │                             │
   ├──► 03_finbert_model.py      │ writes per-model
   │                             │ prediction CSVs
   ├──► 04_bert_model.py         ┘
   │
   ├──► 05_train_distilbert.py ──► 06_evaluate_finetuned_distilbert.py
   │
   ├──► 07_train_bert_large.py   (Colab-friendly; original recipe)
   │
   ├──► 05_evaluation.py  (reads the 02/03/04 prediction CSVs)
   │       │
   │       ▼
   │   06_results.py     (builds model_comparison.csv + headline charts)
   │
   ├──► 08_experiment_epochs.py
   ├──► 09_experiment_data_size.py
   ├──► 10_generate_experiment_graphs.py
   └──► 11_generate_presentation_assets.py

Supplementary experiments (additive — read existing files, write new ones):

01 data + 05_train_distilbert ──► 12_validation_check.py
01 data ────────────────────────► 13_classical_baseline.py
all prediction CSVs ────────────► 14_macro_f1_report.py
01 data ────────────────────────► 15_multi_seed_distilbert.py
01 data (run on Colab) ─────────► 16_bert_large_retrain.py
all prediction CSVs ────────────► 17_significance_and_errors.py
```

---

## How to actually run it

The short version (the full version is in the README):

1. **Set up the conda environment:**

   ```bash
   conda create -n diplomska-nlp python=3.11
   conda activate diplomska-nlp
   pip install -r requirements.txt
   ```

2. **Run the main pipeline** in numbered order. Most scripts take seconds to a few minutes; `05_train_distilbert.py` and `07_train_bert_large.py` are the slower ones.

3. **Run the supplementary experiments** in any order — they are independent of each other.

4. For **`16_bert_large_retrain.py`**, use a Google Colab notebook with a T4 GPU runtime (about 10–15 minutes there).

---

## What the project tries to convince you of (the headline finding)

After all the layered experiments, the conclusion is:

> **Bigger is fragile, and, when handled, leads.** Large transformer models (like BERT-large) are more sensitive to hyperparameter choice than smaller ones. With default settings that work fine for a small model, the big one can collapse completely. With the *right* settings, the same big model produces the highest accuracy. Smaller fine-tuned models and domain-pretrained models sit just below in raw accuracy, are statistically indistinguishable from the big one at this sample size, and are far less fragile to set up. So the choice between transformer approaches in this study is more about cost and hyperparameter risk than about accuracy.

For the detailed evidence and tables, see `THESIS_DRAFT.md` / `.docx`.
