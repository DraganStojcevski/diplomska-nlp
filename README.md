# Diploma NLP Project

## Step 1: Load and Inspect the Dataset

Run:

```bash
conda run -n diplomska-nlp python 01_load_dataset.py
```

This loads the Financial PhraseBank dataset and prints:

- the dataset structure
- the first rows
- label counts
- readable label names

Labels:

```text
0 = negative
1 = neutral
2 = positive
```

## Step 2: Split the Dataset

The same script also splits the dataset into:

```text
80% training data
20% testing data
```

The split uses:

- `random_state=42` so the result is reproducible
- `stratify=df["label"]` so the label balance is preserved

Expected sizes:

```text
Train size: 1811
Test size: 453
```

## Step 3: Run a Baseline Sentiment Model

The same script also loads a pretrained Hugging Face sentiment model:

```python
classifier = pipeline("sentiment-analysis")
```

It tests:

- one custom financial sentence
- 20 random sentences from the test set

This first baseline model predicts only:

```text
POSITIVE
NEGATIVE
```

That is okay for this step. The goal is only to confirm that a pretrained model can read the dataset sentences and return predictions.

## Step 4: Calculate Accuracy

The script compares:

```text
true dataset label vs model prediction
```

The baseline model predicts `POSITIVE` and `NEGATIVE`, so the script maps them to dataset labels:

```text
NEGATIVE -> 0
POSITIVE -> 2
```

Then it calculates accuracy on the 20-sentence sample.

## Step 5: Use FinBERT

The script also loads a financial-domain sentiment model:

```python
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
```

FinBERT predicts:

```text
negative
neutral
positive
```

The script maps those labels to the dataset format:

```text
negative -> 0
neutral -> 1
positive -> 2
```

Then it calculates FinBERT accuracy on the same 20-sentence sample.

## Step 6: Evaluate Both Models on the Full Test Dataset

The script now runs both models on all test sentences:

```text
453 test sentences
```

It calculates:

```text
General Model Accuracy
FinBERT Accuracy
```

This gives the first real full-test comparison between a general sentiment model and a financial-domain model.

## Step 7: Calculate Precision, Recall, and F1-Score

The script now prints a classification report for both models:

```python
classification_report(true_labels_full, predicted_labels)
```

This includes:

- precision
- recall
- F1-score
- support

The report is printed separately for:

```text
General Model Report
FinBERT Report
```

## Step 8: Create a Results Table

The script creates a comparison table with:

- accuracy
- weighted precision
- weighted recall
- weighted F1-score

The table is saved to:

```text
results/model_comparison.csv
```

## Step 9: Create Graphs

The script creates two bar charts:

```text
results/accuracy_comparison.png
results/f1_score_comparison.png
```

These graphs compare:

- model accuracy
- model F1-score

## Step 10: Measure Inference Time

The script measures how long each model takes to process the full test dataset.

It adds this column to the comparison table:

```text
Time (seconds)
```

This makes it possible to compare:

```text
accuracy vs speed
```

## Step 11: Measure Memory Usage

The script measures memory usage during full-test inference.

It adds this column to the comparison table:

```text
Memory (MB)
```

This helps compare model resource usage, not only accuracy and speed.

## Step 12: Save Predictions and Results

The script saves full test-set predictions to:

```text
results/predictions.csv
```

It also keeps saving the final comparison table to:

```text
results/model_comparison.csv
```

The predictions file includes:

- original sentence
- true label
- general model prediction
- BERT prediction
- FinBERT prediction

## Step 13: Clean Project Structure

The project is now split into clear files:

```text
01_data.py
02_general_model.py
03_finbert_model.py
04_bert_model.py
04_bert_large_model.py
05_train_distilbert.py
06_evaluate_finetuned_distilbert.py
07_train_bert_large.py
05_evaluation.py
06_results.py
07_finetune_distilbert.py

data/
models/
results/
```

Run the structured pipeline in this order:

```bash
conda run -n diplomska-nlp python 01_data.py
conda run -n diplomska-nlp python 02_general_model.py
conda run -n diplomska-nlp python 03_finbert_model.py
conda run -n diplomska-nlp python 04_bert_model.py
conda run -n diplomska-nlp python 05_train_distilbert.py
conda run -n diplomska-nlp python 06_evaluate_finetuned_distilbert.py
conda run -n diplomska-nlp python 07_train_bert_large.py
conda run -n diplomska-nlp python 05_evaluation.py
conda run -n diplomska-nlp python 06_results.py
conda run -n diplomska-nlp python 07_finetune_distilbert.py
```

Each file has one responsibility:

```text
01_data.py           Load dataset and create train/test split
02_general_model.py  Run the general sentiment model
03_finbert_model.py  Run FinBERT
04_bert_model.py     Run the sentiment-ready BERT star-rating baseline
04_bert_large_model.py  Load bert-large-cased for sequence classification
05_train_distilbert.py  Fine-tune DistilBERT for sequence classification
06_evaluate_finetuned_distilbert.py  Evaluate the saved fine-tuned DistilBERT
07_train_bert_large.py  Fine-tune bert-large-cased for sequence classification
05_evaluation.py     Calculate metrics and save predictions
06_results.py        Create final table and graphs
07_finetune_distilbert.py  Fine-tune DistilBERT on Financial PhraseBank
```

## Step 14: Add BERT as the Larger Model

The project now includes a third model:

```text
nlptown/bert-base-multilingual-uncased-sentiment
```

This model returns star labels:

```text
1 star
2 stars
3 stars
4 stars
5 stars
```

The script maps them into the dataset labels:

```text
1 star       -> 0 negative
2 or 3 stars -> 1 neutral
4 or 5 stars -> 2 positive
```

The comparison now contains:

```text
General Model
BERT
FinBERT
```

## Step 16: Create Graphs for All Models

The results script creates four graphs:

```text
results/accuracy_comparison.png
results/f1_score_comparison.png
results/time_comparison.png
results/memory_comparison.png
```

These graphs compare all three models:

```text
General Model
BERT
FinBERT
```

## Step 17: Fine-Tune DistilBERT

The project now trains a small model on the Financial PhraseBank training data:

```text
distilbert-base-uncased
```

Run:

```bash
conda run -n diplomska-nlp python 07_finetune_distilbert.py
```

This script saves:

```text
models/distilbert_finetuned/
results/distilbert_finetuned_predictions.csv
results/distilbert_finetuned_training_history.csv
results/distilbert_finetuned_eval.json
```

This step checks whether a small model improves after being trained on the financial sentiment dataset.

## Step 18: Add BERT-Large Sequence Classification

The project now includes the true large model requested for the дипломска:

```text
bert-large-cased
```

The loader is in:

```text
04_bert_large_model.py
```

This file uses:

```python
AutoTokenizer.from_pretrained("bert-large-cased")
AutoModelForSequenceClassification.from_pretrained(
    "bert-large-cased",
    num_labels=3
)
```

This model is not used with `pipeline("sentiment-analysis")`, because it is not already trained for financial sentiment. The next step is to fine-tune it on the Financial PhraseBank dataset.

Optional loader smoke test:

```bash
conda run -n diplomska-nlp python 04_bert_large_model.py
```

This command may download a large model the first time it runs.

## Step 19: Fine-Tune DistilBERT for Sequence Classification

The project now has a clean small-model training file:

```text
05_train_distilbert.py
```

It trains:

```text
distilbert-base-uncased
```

on the Financial PhraseBank training split and saves the trained model to:

```text
models/distilbert-finetuned/
```

It also saves:

```text
results/distilbert_finetuned_predictions.csv
results/distilbert_finetuned_training_history.csv
results/distilbert_finetuned_eval.json
```

Run:

```bash
conda run -n diplomska-nlp python 05_train_distilbert.py
```

## Step 20: Evaluate Fine-Tuned DistilBERT

The project now reloads the saved small model from disk:

```text
models/distilbert-finetuned/
```

The evaluation file is:

```text
06_evaluate_finetuned_distilbert.py
```

It calculates:

```text
Accuracy
Precision
Recall
F1-score
```

and saves:

```text
results/distilbert_finetuned_loaded_predictions.csv
results/distilbert_finetuned_loaded_eval.json
```

Run:

```bash
conda run -n diplomska-nlp python 06_evaluate_finetuned_distilbert.py
```

## Step 21: Fine-Tune BERT-Large

The project now has a large-model training file:

```text
07_train_bert_large.py
```

It trains:

```text
bert-large-cased
```

on the same Financial PhraseBank training split. Because BERT-large is much heavier than DistilBERT, this file uses:

```text
batch_size = 2
max_length = 128
num_epochs = 3
gradient checkpointing
```

It saves the trained model to:

```text
models/bert-large-finetuned/
```

It also saves:

```text
results/bert_large_finetuned_predictions.csv
results/bert_large_finetuned_training_history.csv
results/bert_large_finetuned_eval.json
```

Run:

```bash
conda run --no-capture-output -n diplomska-nlp python 07_train_bert_large.py
```

The `--no-capture-output` option is recommended for this step because BERT-large training is slow and you should see progress logs while it runs.

## Step 22: Add Fine-Tuned Models to Final Results

The final results table now compares:

```text
General Model
Fine-tuned DistilBERT
Fine-tuned BERT-large
FinBERT
```

The results script reads the BERT-large Colab predictions from:

```text
results/bert_large_finetuned_predictions.csv
```

Then it updates:

```text
results/model_comparison.csv
results/accuracy_comparison.png
results/f1_score_comparison.png
results/time_comparison.png
results/memory_comparison.png
```

Run:

```bash
conda run -n diplomska-nlp python 06_results.py
```

## Step 23: Create Final Comparison Graphs

The results script now creates final presentation graphs with short model names:

```text
General
DistilBERT FT
BERT-large
FinBERT
```

The final graph files are:

```text
results/final_accuracy.png
results/final_f1.png
results/final_time.png
```

Run:

```bash
conda run -n diplomska-nlp python 06_results.py
```

## Step 25: Create a Streamlit UI

The project now includes an interactive demo:

```text
app.py
```

The UI includes:

```text
Text input
Analyze button
Model selection
Prediction label
Confidence score
Model comparison
Saved results table
Final graphs
```

Run:

```bash
streamlit run app.py
```

The UI supports live predictions with FinBERT and the fine-tuned DistilBERT model. Fine-tuned BERT-large is shown in the saved results table and will be available for live prediction if the saved model folder is copied into `models/bert-large-finetuned/`.

## Step 26: Multi-Model Comparison UI

The Streamlit UI now runs all available live models for the same input sentence and displays the predictions side by side.

Live comparison includes:

```text
FinBERT
DistilBERT FT
BERT-large
```

Run:

```bash
streamlit run app.py
```

Enter one financial sentence and click `Analyze`. The UI shows each model's predicted label and confidence score in one table.

The UI is organized into tabs:

```text
Live Demo
Final Comparison
Epoch Experiment
Dataset Size Experiment
Prediction Files
```

This makes the Streamlit app usable as a complete дипломска presentation dashboard.

## Step 27: Controlled Experiments

The project now includes two deeper experiments for the дипломска analysis.

Epoch experiment:

```bash
conda run --no-capture-output -n diplomska-nlp python 08_experiment_epochs.py
```

This saves:

```text
results/experiment_epochs.csv
results/experiment_epochs_scores.png
results/experiment_epochs_loss.png
results/experiment_epochs_time.png
```

Dataset size experiment:

```bash
conda run --no-capture-output -n diplomska-nlp python 09_experiment_data_size.py
```

This trains DistilBERT with:

```text
25% training data
50% training data
75% training data
100% training data
```

and saves:

```text
results/experiment_data_size.csv
results/experiment_data_size_scores.png
results/experiment_data_size_time.png
results/experiment_data_size_efficiency.png
```

The Streamlit UI also includes three ready-made finance demo sentences for positive, negative, and neutral examples.

If you already have the experiment CSV files and only want to regenerate improved graphs, run:

```bash
conda run -n diplomska-nlp python 10_generate_experiment_graphs.py
```

This creates presentation-friendly versions such as:

```text
results/experiment_epochs_scores_zoomed.png
results/experiment_epochs_loss.png
results/experiment_epochs_time_bar.png
results/experiment_data_size_scores_zoomed.png
results/experiment_data_size_time_bar.png
results/experiment_data_size_efficiency.png
```

## Step 28: Presentation Assets and Analysis

The project includes a final presentation-asset script:

```bash
conda run -n diplomska-nlp python 11_generate_presentation_assets.py
```

This creates:

```text
results/confusion_matrix_finbert.png
results/confusion_matrix_distilbert_ft.png
results/confusion_matrix_bert_large.png
results/architecture_diagram.png
results/main_summary_figure.png
results/analysis_summary.md
```

The Streamlit UI displays these in the `Analysis` tab.
# diplomska-nlp
