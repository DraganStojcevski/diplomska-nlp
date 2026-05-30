# Performance Comparison Between Small Language Models and Large Language Models for Natural Language Processing Tasks

> Bachelor thesis · practical implementation and empirical analysis  
> Author **Dragan Stojchevski** · Software Engineering and Innovation

A layered empirical study comparing **classical**, **small fine-tuned**, **domain-pretrained**, and **large fine-tuned** language models on three-class financial sentiment classification. Uses the Financial PhraseBank dataset (`sentences_allagree`, 2,264 sentences; same 80/20 stratified split for every model).

The study runs an initial four-model comparison, two controlled experiments (epoch sweep, dataset-size sweep), and six supplementary methodology experiments (held-out validation, classical baseline, macro-F1, multi-seed variance, BERT-large retraining, and McNemar significance tests).

---

## Headline finding

> **Bigger is fragile, and, when handled, leads.** Large transformer models are more sensitive to hyperparameter choice than smaller ones. With default hyperparameters that work for smaller models, BERT-large collapses to predicting only the majority class; with a 4× lower learning rate and linear warmup, the same architecture reaches the highest accuracy in the study. Smaller fine-tuned and domain-pretrained models cluster slightly below it in raw accuracy, are statistically indistinguishable from it at this sample size, and are far less fragile to set up.

## Results

All numbers on the same 453-sentence stratified test set.

| Model | Accuracy | F1 macro | Inference | Notes |
|---|---:|---:|---:|---|
| General Model (binary baseline) | 25.4% | 24.4% | 7.0 s | Cannot predict the neutral class |
| TF-IDF + Logistic Regression | **86.5%** | **80.5%** | **0.008 s** | Classical reference; ~700× faster than transformers |
| FinBERT | 96.9% | 96.3% | 8.3 s | Domain pretraining, no fine-tuning |
| Fine-tuned DistilBERT (multi-seed) | 96.2 ± 0.6% | 94.7 ± 1.0% | 5.6 s | Mean ± std over 5 seeds |
| Fine-tuned BERT-large (original) | 61.4% | 25.4% | 15.7 s | Collapsed to all-neutral (training artifact) |
| **Fine-tuned BERT-large (retrained)** | **98.5%** | **98.0%** | — | LR 5e-6 + linear warmup, Colab T4 |

**McNemar pairwise tests (n = 453):** the three top transformers — BERT-large retrained (98.5%), DistilBERT FT (~96–97%), and FinBERT (96.9%) — are statistically indistinguishable from each other (all pairwise *p* > 0.05). All three significantly outperform the classical baseline (all pairwise *p* < 0.001).

---

## Setup

```bash
conda create -n diplomska-nlp python=3.11
conda activate diplomska-nlp
pip install -r requirements.txt
```

## Run order

**Main pipeline** (initial four-model comparison + two original controlled experiments):

```bash
conda run -n diplomska-nlp python 01_data.py
conda run -n diplomska-nlp python 02_general_model.py
conda run -n diplomska-nlp python 03_finbert_model.py
conda run -n diplomska-nlp python 04_bert_model.py
conda run -n diplomska-nlp python 05_train_distilbert.py
conda run -n diplomska-nlp python 06_evaluate_finetuned_distilbert.py
conda run --no-capture-output -n diplomska-nlp python 07_train_bert_large.py
conda run -n diplomska-nlp python 05_evaluation.py
conda run -n diplomska-nlp python 06_results.py
conda run --no-capture-output -n diplomska-nlp python 08_experiment_epochs.py
conda run --no-capture-output -n diplomska-nlp python 09_experiment_data_size.py
conda run -n diplomska-nlp python 10_generate_experiment_graphs.py
conda run -n diplomska-nlp python 11_generate_presentation_assets.py
```

**Supplementary methodology experiments** (all additive — they read existing outputs and write new files; nothing in `data/`, `models/`, or pre-existing `results/*` files is modified):

```bash
conda run --no-capture-output -n diplomska-nlp python 12_validation_check.py
conda run -n diplomska-nlp python 13_classical_baseline.py
conda run -n diplomska-nlp python 14_macro_f1_report.py
conda run --no-capture-output -n diplomska-nlp python 15_multi_seed_distilbert.py
conda run -n diplomska-nlp python 17_significance_and_errors.py
```

`16_bert_large_retrain.py` is intended for Colab/GPU (~10–15 min on T4). On local Apple MPS expect ~90+ min with possible memory pressure.

---

## Project structure

```text
.
├── 01_data.py                          Load dataset, create stratified train/test split
├── 02_general_model.py                 Run the generic binary sentiment baseline
├── 03_finbert_model.py                 Run FinBERT (domain pretraining)
├── 04_bert_model.py                    Run the multilingual BERT star-rating baseline
├── 04_bert_large_model.py              Load bert-large-cased (smoke test)
├── 05_evaluation.py                    Combine predictions, compute classification reports
├── 05_train_distilbert.py              Fine-tune DistilBERT on Financial PhraseBank
├── 06_evaluate_finetuned_distilbert.py Reload and evaluate the saved fine-tuned DistilBERT
├── 06_results.py                       Final comparison table + headline graphs
├── 07_train_bert_large.py              Fine-tune bert-large-cased (original recipe)
├── 08_experiment_epochs.py             DistilBERT epoch-count experiment
├── 09_experiment_data_size.py          DistilBERT training-data-size experiment
├── 10_generate_experiment_graphs.py    Regenerate improved graphs from experiment CSVs
├── 11_generate_presentation_assets.py  Confusion matrices, architecture diagram, summary
│
├── 12_validation_check.py              Held-out validation verification of epoch choice
├── 13_classical_baseline.py            TF-IDF + Logistic Regression reference baseline
├── 14_macro_f1_report.py               Macro F1 and per-class F1 across all models
├── 15_multi_seed_distilbert.py         5-seed variance estimate of DistilBERT FT
├── 16_bert_large_retrain.py            BERT-large retrain with corrected hyperparameters (Colab)
├── 17_significance_and_errors.py       McNemar pairwise tests + per-model error analysis
│
├── data/                               train.csv, test.csv
├── models/                             Local fine-tuned model artifacts (not committed)
├── results/                            Predictions, metrics, graphs, analysis outputs
│
├── requirements.txt                    Python dependencies
├── README.md                           This file
├── THESIS_DRAFT.md                     Thesis write-up (Markdown)
└── THESIS_DRAFT.docx                   Thesis write-up (Word, regenerated from MD via pandoc)
```

---

## What lands in `results/`

| File | Contents |
|---|---|
| `model_comparison.csv` | Original four-model headline table (accuracy, precision, recall, F1, time, memory) |
| `model_comparison_extended.csv` | Same plus the classical baseline row |
| `multi_seed_distilbert_summary.json` | Per-seed and aggregate (mean ± std) DistilBERT FT metrics |
| `bert_large_retrain_eval.json` | Colab T4 result of the corrected BERT-large retraining |
| `validation_check_summary.json` | Held-out val per-epoch + best-epoch selection |
| `macro_f1_comparison.csv` | Weighted vs macro F1 and per-class F1 for every model |
| `mcnemar_results.csv` | Pairwise McNemar test results with p-values and verdicts |
| `error_analysis.csv` | Test sentences misclassified by at least one of the top three transformers |
| `analysis_summary.md` | Auto-generated narrative summary |
| `final_accuracy.png`, `final_f1.png`, `final_time.png` | Headline comparison charts |
| `confusion_matrix_*.png` | Per-model confusion matrices |
| `experiment_epochs_*`, `experiment_data_size_*` | Charts for the two original controlled experiments |

---

## Companion materials

- **`THESIS_DRAFT.md` / `.docx`** in this repo — the full thesis write-up, including the layered methodology chapter (§11) that documents every supplementary experiment.
- **`diplomska-presentation/`** (separate repo on Desktop) — React-based defense deck with the live demo, comparison panels, and the headline finding.

---

## Reproducibility

- All scripts seed at `random_state=42` (and per-seed for the multi-seed experiment).
- The train/test split is saved to `data/train.csv` and `data/test.csv` so every model evaluates on identical examples.
- Supplementary experiments write to new output files only — they never modify the original `model_comparison.csv` or any other artifact from the main pipeline.

---

## References

### Related work — small vs large language models

- Belcak, P., Heinrich, G., Diao, S., Fu, Y., Dong, X., Muralidharan, S., Lin, Y. C., & Molchanov, P. (2025). *Small Language Models are the Future of Agentic AI.* arXiv:2506.02153.
- Chen, Y., Zhao, J., & Han, H. (2025). *A Survey on Collaborative Mechanisms Between Large and Small Language Models.* arXiv:2505.07460.
- Gupta, A., Thomas, B., Asnani, H., Madduru, P. R., Feroze, S., Subramanian, S., Elango, V., & Gungor, M. (2025). *Small Language Models (SLMs) Can Still Pack a Punch: A survey.* arXiv:2501.05465.
- Sakib, T. H., Hosain, M. T., & Morol, M. K. (2025). *Small Language Models: Architectures, Techniques, Evaluation, Problems and Future Adaptation.* arXiv:2505.19529.

### Methods and dataset

- **Financial PhraseBank**: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.*
- **BERT**: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *Pre-training of deep bidirectional transformers for language understanding.*
- **DistilBERT**: Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *A distilled version of BERT: smaller, faster, cheaper and lighter.*
- **FinBERT**: Araci, D. (2019). *Financial sentiment analysis with pre-trained language models.*
- **Transformer**: Vaswani, A. et al. (2017). *Attention is all you need.*
- **Hugging Face Transformers**: Wolf, T. et al. (2020).
