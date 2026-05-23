# Performance Comparison Between Small Language Models and Large Language Models for Natural Language Processing Tasks

**Student:** Dragan Stojchevski  
**Index number:** A23071  
**Study program:** Software Engineering and Innovation  
**Project type:** Bachelor thesis practical implementation and analysis  

> Note: This is a Word-ready thesis draft. After copying it into Microsoft Word, add the official university title page, table of contents, page numbers, figure numbering, and final citation formatting required by the faculty.

---

## Abstract

Natural Language Processing is one of the most important areas of artificial intelligence because it enables computers to process and interpret human language. In recent years, transformer-based models such as BERT, DistilBERT, and domain-specific variants such as FinBERT have become widely used for text classification tasks. However, choosing the best model for a practical application is not only a question of accuracy. It also involves efficiency, execution time, memory usage, domain adaptation, training cost, and usability.

This thesis investigates the performance difference between smaller and larger language models on a financial sentiment classification task. The practical part uses the Financial PhraseBank dataset, which contains financial sentences labeled as negative, neutral, or positive. Several models were implemented and evaluated: a general pretrained sentiment model, a fine-tuned DistilBERT model, a fine-tuned BERT-large model, and FinBERT, a model trained specifically for financial sentiment analysis. The models were compared using accuracy, precision, recall, weighted F1-score, inference time, and memory usage.

The results show that model size alone does not guarantee better performance. Fine-tuned DistilBERT achieved the best overall result with an accuracy of 0.9713 and weighted F1-score of 0.9713, while FinBERT achieved very similar performance with an accuracy of 0.9691 and weighted F1-score of 0.9695. The fine-tuned BERT-large model performed significantly worse, with an accuracy of 0.6137 and weighted F1-score of 0.4668, because it predicted the majority neutral class for all test examples. This finding is important because it shows that larger models are not automatically better when the dataset is small or when training is not sufficiently optimized. Domain-specific training and task-specific fine-tuning can be more important than parameter count.

In addition to the model comparison, two controlled experiments were performed. The first experiment tested the effect of training epochs on DistilBERT performance, while the second tested the effect of dataset size. These experiments showed that two epochs produced the best balance between performance and training cost, and that performance improved strongly when the training data increased from 25% to 50%, but gains became smaller afterward.

Finally, an interactive Streamlit prototype was developed. The prototype allows users to enter financial text, compare predictions from multiple models, analyze batches of sentences, view confidence scores, inspect saved results, and present the system as a practical financial sentiment intelligence tool. Therefore, the project combines research, software engineering, model evaluation, and an interactive user-facing application.

**Keywords:** Natural Language Processing, Sentiment Analysis, Financial PhraseBank, DistilBERT, BERT-large, FinBERT, Small Language Models, Large Language Models, Fine-tuning, Model Evaluation

---

## 1. Introduction

Natural Language Processing, commonly abbreviated as NLP, is a field of artificial intelligence that focuses on enabling computers to understand, process, analyze, and generate human language. Human language is complex because it contains grammar, context, tone, ambiguity, domain-specific terminology, and implicit meaning. Because of this, NLP tasks are challenging but also extremely useful in real-world systems.

One important NLP task is sentiment analysis. Sentiment analysis is the process of identifying whether a given text expresses a positive, negative, or neutral opinion. It is used in many areas such as product reviews, social media monitoring, customer feedback analysis, political analysis, and financial market monitoring. In the financial domain, sentiment analysis is especially useful because company announcements, financial reports, and news headlines can influence decision-making. Investors, analysts, and business users often need to quickly understand whether a piece of financial text indicates positive, negative, or neutral information.

The development of transformer-based language models has significantly improved NLP performance. Models such as BERT introduced contextual word representations, meaning that the meaning of a word can be interpreted based on the full sentence. This is very important in financial language, where words can have different meanings depending on context. For example, the sentence "The company reduced losses" is positive even though the word "losses" usually has a negative meaning. A simple keyword-based model may classify this sentence incorrectly, while a transformer-based model can understand the context more effectively.

However, transformer models vary greatly in size and purpose. Some models are smaller and faster, such as DistilBERT. Other models are larger and more computationally expensive, such as BERT-large. There are also domain-specific models, such as FinBERT, which are trained or adapted for financial language. This creates an important practical question: should a system use a smaller model, a larger model, or a domain-specific model?

This thesis focuses on that question. The project compares multiple transformer-based models on the same financial sentiment classification dataset. The goal is not only to measure which model is most accurate, but also to understand the trade-offs between accuracy, speed, memory usage, fine-tuning, and practical usability.

The project also includes a working prototype application. The application allows a user to enter a financial sentence and compare predictions from multiple models. It also includes batch analysis, result tables, graphs, confusion matrices, and interpretation. This makes the project more than a simple experiment; it becomes a complete experimental NLP system with a usable interface.

### 1.1 Problem Statement

Many NLP systems focus only on accuracy. However, in real software systems, accuracy is not the only important factor. A model may be accurate but too slow. A model may be large but not adapted to the target domain. A smaller model may perform well if it is fine-tuned correctly. A domain-specific model may outperform a larger general model because it understands the language of the domain better.

In financial sentiment analysis, this problem is especially important because financial language is different from everyday language. General sentiment models are often trained on movie reviews, product reviews, or general internet text. These models may not correctly interpret financial sentences. For example, a sentence may contain words that seem negative in general language but are actually neutral or positive in finance.

The central problem addressed in this thesis is:

**How do small, large, and domain-specific language models compare on financial sentiment classification when evaluated using accuracy, F1-score, inference time, memory usage, and practical usability?**

### 1.2 Research Objectives

The main objective of this thesis is to build and evaluate a complete NLP system for financial sentiment analysis. The specific objectives are:

1. To load and inspect the Financial PhraseBank dataset.
2. To split the dataset into training and testing subsets using a reproducible method.
3. To evaluate a general pretrained sentiment model as a baseline.
4. To evaluate FinBERT as a financial-domain sentiment model.
5. To fine-tune DistilBERT as a small model for financial sentiment classification.
6. To fine-tune BERT-large as a larger model for the same task.
7. To compare all models using accuracy, precision, recall, F1-score, inference time, and memory usage.
8. To perform additional controlled experiments using different numbers of training epochs and different training dataset sizes.
9. To create graphs, confusion matrices, and analysis outputs for presentation.
10. To develop an interactive Streamlit user interface that demonstrates the system in a practical way.

### 1.3 Research Questions

The thesis is guided by the following research questions:

**RQ1:** Does a domain-specific financial sentiment model perform better than a general sentiment model?

**RQ2:** Can a smaller model such as DistilBERT become competitive after fine-tuning on the target dataset?

**RQ3:** Does a larger model such as BERT-large automatically outperform a smaller model?

**RQ4:** How does the number of fine-tuning epochs affect the performance of DistilBERT?

**RQ5:** How does the amount of training data affect the performance of DistilBERT?

**RQ6:** Can the final system be presented as an interactive prototype useful for financial text analysis?

### 1.4 Thesis Contributions

This thesis makes several practical contributions:

1. A structured NLP project pipeline was implemented, starting from dataset loading and ending with an interactive application.
2. Multiple transformer-based models were compared under the same dataset and evaluation conditions.
3. A small fine-tuned model was shown to be highly competitive with a financial-domain model.
4. A larger model was shown not to be automatically better, which is an important practical conclusion.
5. Additional experiments were performed to analyze training epochs and training dataset size.
6. Results were saved in reusable CSV, JSON, and image formats.
7. A Streamlit prototype was developed to demonstrate live and batch financial sentiment analysis.

---

## 2. Theoretical Background

### 2.1 Natural Language Processing

Natural Language Processing is a subfield of artificial intelligence that deals with the interaction between computers and human language. The goal of NLP is to enable machines to process language in a way that is useful for tasks such as classification, translation, question answering, summarization, information extraction, and text generation.

Traditional NLP systems often depended on manually designed features. For example, a sentiment analysis system could count positive and negative words in a sentence. This approach is easy to understand, but it has major limitations. It does not understand context, grammar, or domain-specific meaning. If the sentence contains sarcasm, negation, or specialized terminology, the result may be incorrect.

Modern NLP systems use machine learning and deep learning. Instead of manually defining all rules, the model learns patterns from data. Transformer-based models are especially powerful because they can process a sentence as a whole and learn relationships between words. This allows them to capture context more effectively than earlier models.

### 2.2 Sentiment Analysis

Sentiment analysis is the task of identifying the emotional or opinion-based orientation of text. In the simplest case, sentiment can be classified as positive or negative. In more advanced cases, a neutral class is also included. In this thesis, sentiment is classified into three categories:

- negative
- neutral
- positive

Financial sentiment analysis is more specialized than general sentiment analysis. A general sentence such as "The movie was excellent" is clearly positive. However, financial language can be more subtle. The sentence "The company reduced its operating loss" may be positive because the company is improving, even though it contains the word "loss". Similarly, "The company remained stable" is usually neutral, even though it does not express strong emotion.

This is why domain-specific models are important. A model trained on general sentiment data may not perform well on financial text because the vocabulary and meaning patterns are different.

### 2.3 Transformer Models

Transformer models are neural network architectures designed to process sequential data such as text. The transformer architecture uses a mechanism called self-attention. Self-attention allows the model to examine relationships between words in a sentence, regardless of their distance from each other.

For example, in the sentence "The company increased revenue, but profit decreased," the model must understand both positive and negative signals. Self-attention helps the model weigh different parts of the sentence and understand the overall meaning.

Transformers became a foundation for many modern NLP models. They improved performance across many tasks because they are able to learn contextual representations of words. A contextual representation means that the same word can have a different meaning depending on the sentence. This is important for financial sentiment because many financial terms are context-dependent.

### 2.4 BERT

BERT stands for Bidirectional Encoder Representations from Transformers. It is a transformer-based language model that reads text bidirectionally, meaning it uses both the left and right context of a word. This allows BERT to understand sentences more deeply than models that process text in only one direction.

BERT can be adapted to many tasks, including text classification. For sentiment classification, a classification layer is added on top of the model. The input sentence is tokenized, processed by BERT, and then classified into one of the sentiment categories.

In this thesis, BERT-large-cased was used as the large model. The model is larger than DistilBERT and has more parameters. In theory, a larger model can learn more complex patterns. However, larger models also require more computational resources, more memory, and more careful training. This thesis investigates whether the larger model actually performs better in this specific task.

### 2.5 DistilBERT

DistilBERT is a smaller and faster version of BERT. It was created using knowledge distillation, a technique where a smaller model learns from a larger model. The goal is to keep much of the performance of BERT while reducing model size and computational cost.

DistilBERT is useful in practical applications because it is faster and easier to deploy. For many real-world systems, especially prototypes or applications with limited hardware, smaller models can be more realistic than very large models. In this thesis, DistilBERT was evaluated in two forms:

1. A general pretrained sentiment model based on DistilBERT.
2. A fine-tuned DistilBERT model trained on Financial PhraseBank.

This comparison shows the difference between using a model directly and adapting it to the target task.

### 2.6 FinBERT

FinBERT is a BERT-based model adapted for financial language. It is designed for financial sentiment analysis and can predict positive, negative, and neutral sentiment. Because it has been trained on financial text, it is expected to understand financial terminology better than a general sentiment model.

In this thesis, FinBERT is used as the domain-specific model. It provides an important comparison point because it represents a model already adapted to the financial domain. The results show whether fine-tuning a general model can compete with a specialized financial model.

### 2.7 Small Language Models and Large Language Models

The topic of this thesis is related to the comparison between small language models and large language models. In practical software engineering, larger models are not always the best choice. A large model may produce strong results, but it may also require more memory, more time, more expensive hardware, and more complex deployment.

Small language models are attractive because they can be faster, cheaper, and easier to integrate into applications. If a small model can be fine-tuned to perform well on a specific task, it may be a better engineering choice than a larger model.

The main idea tested in this thesis is that model selection should consider the task, the domain, the dataset, and the available resources. The best model is not always the biggest model. The best model is the one that provides the best balance between performance and practical constraints.

### 2.8 Evaluation Metrics

Several evaluation metrics were used in this project.

**Accuracy** measures the percentage of correct predictions. It is calculated as the number of correct predictions divided by the total number of predictions. Accuracy is easy to understand, but it can be misleading when the dataset is imbalanced.

**Precision** measures how many predictions for a class were correct. For example, if a model predicts many sentences as positive, precision tells how many of those positive predictions were actually positive.

**Recall** measures how many real examples of a class were found by the model. For example, if there are many truly negative sentences, recall measures how many of them the model correctly identified.

**F1-score** is the harmonic mean of precision and recall. It is useful because it balances both metrics.

**Weighted F1-score** was used because the dataset is imbalanced. The neutral class has more examples than the negative and positive classes. Weighted F1 accounts for the number of examples in each class.

**Inference time** measures how long the model takes to process the test dataset. This is important for practical systems where speed matters.

**Memory usage** measures the resource usage of the model during execution. This is important because models with high memory requirements may be difficult to deploy.

---

## 3. Dataset and Data Preparation

### 3.1 Financial PhraseBank Dataset

The dataset used in this thesis is Financial PhraseBank. It contains sentences from financial news and reports, labeled according to sentiment. The selected configuration was `sentences_allagree`, which contains examples where annotators agreed on the sentiment label. This configuration was chosen because it provides cleaner labels and reduces ambiguity.

The dataset labels are:

| Label ID | Label Name |
|---:|---|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

The dataset is suitable for this thesis because it is directly related to financial sentiment analysis. Instead of using a general sentiment dataset, the project uses financial sentences, which makes the experiment relevant to the research goal.

### 3.2 Dataset Loading

The dataset was loaded using the Hugging Face `datasets` library. The first step of the project was to load and inspect the dataset:

```python
from datasets import load_dataset

dataset = load_dataset("financial_phrasebank", "sentences_allagree")
```

After loading, the dataset was converted into a Pandas DataFrame. This made it easier to inspect the sentences, labels, and distribution of examples.

The initial inspection was important because a machine learning project should not start with model training immediately. The data must first be understood. This includes checking the available columns, label format, number of rows, and example sentences.

### 3.3 Train-Test Split

After loading the dataset, the data was split into training and testing subsets. The project used an 80/20 split:

- 80% training data
- 20% testing data

The split used `random_state=42` so that the result is reproducible. Reproducibility is important because the same experiment should produce the same train-test split every time it is run.

The split also used stratification by label. Stratification preserves the label distribution in both the training and testing sets. This matters because the dataset is imbalanced, with more neutral examples than positive or negative examples.

The final split was:

| Dataset Part | Number of Examples |
|---|---:|
| Training set | 1811 |
| Test set | 453 |

The label distribution was:

| Dataset Part | Negative | Neutral | Positive |
|---|---:|---:|---:|
| Training set | 242 | 1113 | 456 |
| Test set | 61 | 278 | 114 |

The test set contains 453 examples. The neutral class is the largest class, with 278 examples. This means that a model predicting only neutral would already achieve approximately 61.37% accuracy. This fact becomes important later when interpreting the BERT-large result.

### 3.4 Saved Data Files

The prepared data was saved into CSV files:

- `data/train.csv`
- `data/test.csv`

Saving the data makes the pipeline reproducible. Later scripts do not need to reload and split the dataset again. Instead, they use the same saved training and testing files. This ensures that all models are evaluated on the same test set.

---

## 4. System Architecture and Implementation

### 4.1 Project Structure

The project was organized into multiple files instead of one long script. This is important from a software engineering perspective because it makes the system easier to understand, maintain, and extend.

The main files are:

| File | Purpose |
|---|---|
| `01_data.py` | Loads the dataset and creates train/test CSV files |
| `02_general_model.py` | Runs the general sentiment baseline model |
| `03_finbert_model.py` | Runs the FinBERT domain-specific model |
| `04_bert_model.py` | Runs an additional sentiment-ready BERT model |
| `04_bert_large_model.py` | Loads BERT-large for sequence classification |
| `05_train_distilbert.py` | Fine-tunes DistilBERT |
| `06_evaluate_finetuned_distilbert.py` | Evaluates the saved fine-tuned DistilBERT model |
| `07_train_bert_large.py` | Fine-tunes BERT-large |
| `08_experiment_epochs.py` | Runs the epoch experiment |
| `09_experiment_data_size.py` | Runs the dataset-size experiment |
| `10_generate_experiment_graphs.py` | Regenerates experiment graphs |
| `11_generate_presentation_assets.py` | Generates confusion matrices and presentation figures |
| `06_results.py` | Creates the final comparison table and graphs |
| `app.py` | Streamlit interactive user interface |

The project also contains:

| Folder | Purpose |
|---|---|
| `data/` | Prepared training and test CSV files |
| `results/` | Predictions, metrics, graphs, and analysis outputs |
| `models/` | Local fine-tuned model folders |

This structure separates data preparation, model execution, evaluation, experiments, results, and user interface. This makes the project more professional and closer to a real software system.

### 4.2 Processing Pipeline

The overall processing pipeline is:

```text
Dataset -> Train/Test Split -> Models -> Predictions -> Evaluation -> Results -> UI
```

The steps are:

1. Load Financial PhraseBank.
2. Convert the dataset into a DataFrame.
3. Split the dataset into training and testing data.
4. Run each model on the test set.
5. Convert model labels into the dataset label format.
6. Compare predictions with true labels.
7. Calculate evaluation metrics.
8. Save predictions and metrics to files.
9. Generate graphs and confusion matrices.
10. Display results in the Streamlit application.

This pipeline was chosen because it is clear, reproducible, and easy to explain. Every stage produces outputs that are saved and can be inspected.

### 4.3 Label Mapping

Different models return labels in different formats. For example, the general sentiment model returns:

```text
POSITIVE
NEGATIVE
```

FinBERT returns:

```text
positive
neutral
negative
```

Fine-tuned sequence classification models return:

```text
LABEL_0
LABEL_1
LABEL_2
```

The dataset uses numeric labels:

```text
0 = negative
1 = neutral
2 = positive
```

Therefore, label mapping was necessary. Without mapping, the predictions could not be compared with the dataset labels. The project maps each model output to the same numeric label system. This allows all models to be evaluated fairly with the same metrics.

### 4.4 Evaluation and Saved Outputs

The project saves predictions and results instead of only printing them in the terminal. This is important because results should be reusable, inspectable, and reproducible.

The saved outputs include:

- prediction CSV files
- runtime JSON files
- final comparison CSV file
- accuracy graphs
- F1-score graphs
- time graphs
- memory graphs
- confusion matrices
- analysis summary

This makes it possible to use the results in the thesis document and in the Streamlit prototype.

---

## 5. Models Used in the Experiment

### 5.1 General Sentiment Model

The first model was a general pretrained sentiment model:

```text
distilbert/distilbert-base-uncased-finetuned-sst-2-english
```

This model is trained for general English sentiment analysis. It predicts only two classes:

- positive
- negative

It does not predict neutral. This is a limitation because the Financial PhraseBank dataset contains three classes. However, this model is useful as a baseline because it shows what happens when a general sentiment model is used directly on financial text.

The general model was expected to perform poorly because it is not trained for financial language and cannot produce the neutral class. This expectation was confirmed in the results.

### 5.2 FinBERT

The second model was FinBERT:

```text
ProsusAI/finbert
```

FinBERT is a financial-domain sentiment model. It predicts:

- negative
- neutral
- positive

This model was included because the task is financial sentiment analysis. It provides a domain-specific comparison point. If FinBERT performs better than the general model, it shows that domain adaptation is important.

### 5.3 Fine-tuned DistilBERT

The third model was DistilBERT fine-tuned on the Financial PhraseBank dataset:

```text
distilbert-base-uncased
```

DistilBERT is a smaller model than BERT. It was fine-tuned for three-class sequence classification using the training set. The classification task had three output labels:

- negative
- neutral
- positive

The fine-tuning configuration included:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 128 |
| Batch size | 8 |
| Number of epochs | 3 |
| Learning rate | 2e-5 |
| Random seed | 42 |

The purpose of this model was to test whether a smaller model can become competitive after task-specific training.

### 5.4 Fine-tuned BERT-large

The fourth model was BERT-large fine-tuned on the same dataset:

```text
bert-large-cased
```

BERT-large was used as the large model in the experiment. It was also trained for three-class sequence classification:

- negative
- neutral
- positive

The fine-tuning configuration included:

| Parameter | Value |
|---|---:|
| Maximum sequence length | 128 |
| Batch size | 2 |
| Number of epochs | 3 |
| Learning rate | 2e-5 |
| Random seed | 42 |

The batch size was smaller because BERT-large is much heavier than DistilBERT. The model was trained in Google Colab using a T4 GPU because local training would be slower and more resource-intensive.

BERT-large was included to test whether a larger model provides better performance than a smaller fine-tuned model.

---

## 6. Fine-tuning Methodology

### 6.1 Why Fine-tuning Was Needed

Pretrained language models learn general language patterns from large text corpora. However, a pretrained model is not automatically optimized for every task. Fine-tuning is the process of continuing training on a smaller task-specific dataset.

In this project, fine-tuning was necessary because the task is financial sentiment classification. The model must learn to classify financial sentences into negative, neutral, and positive categories. This is different from general sentiment classification because financial language has specialized meaning.

For example:

```text
The company narrowed its losses.
```

This sentence may be positive in a financial context, because losses are decreasing. A general sentiment model may focus on the word "losses" and classify it as negative. Fine-tuning helps the model learn the correct domain-specific patterns.

### 6.2 Tokenization

Before a transformer model can process text, the text must be tokenized. Tokenization converts text into numerical input IDs that the model can understand.

In this project, each sentence was tokenized with:

- truncation enabled
- padding to maximum length
- maximum length of 128 tokens

The maximum length of 128 was chosen because the dataset contains relatively short financial sentences. This value is sufficient for the task while keeping training and inference efficient.

### 6.3 Training Process

The training process consisted of the following steps:

1. Load the pretrained tokenizer.
2. Load the pretrained model with a classification head.
3. Convert the dataset into tokenized tensors.
4. Train the model on the training set.
5. Evaluate the model on the test set after training.
6. Save the trained model and tokenizer.
7. Save predictions and evaluation metrics.

The loss function used by sequence classification models is cross-entropy loss. This is standard for multi-class classification tasks.

### 6.4 Reproducibility

The project used a fixed random seed of 42. This improves reproducibility. The train-test split was also saved to CSV files so that all experiments used the same data.

Reproducibility is important in research because results should not depend on a random split that changes each time the code is executed. By saving the split and using the same test set, the comparison between models becomes fairer.

---

## 7. Final Model Comparison

### 7.1 Evaluation Setup

All models were evaluated on the same test set of 453 financial sentences. The same true labels were used for every model. The evaluation metrics were:

- accuracy
- weighted precision
- weighted recall
- weighted F1-score
- inference time
- memory usage

The final comparison table is shown below.

**Table 1. Final model comparison**

| Model | Accuracy | Precision | Recall | F1-score | Time (s) | Memory (MB) |
|---|---:|---:|---:|---:|---:|---:|
| General Model | 0.2539 | 0.1015 | 0.2539 | 0.1395 | 7.04 | 309.34 |
| Fine-tuned DistilBERT | 0.9713 | 0.9717 | 0.9713 | 0.9713 | 5.55 | 2053.06 |
| Fine-tuned BERT-large | 0.6137 | 0.3766 | 0.6137 | 0.4668 | 15.67 | 384.19 |
| FinBERT | 0.9691 | 0.9712 | 0.9691 | 0.9695 | 8.28 | 776.08 |

![Final accuracy comparison](results/final_accuracy.png)

![Final F1-score comparison](results/final_f1.png)

![Final time comparison](results/final_time.png)

### 7.2 General Model Results

The general sentiment model achieved an accuracy of 0.2539 and a weighted F1-score of 0.1395. This was the weakest result in the final comparison.

The poor result is expected for two reasons. First, the model was trained for general sentiment, not financial sentiment. Second, it predicts only positive and negative labels. Since the dataset includes a large neutral class, the model cannot correctly predict many examples.

This result demonstrates that a general pretrained sentiment model should not be used blindly for a domain-specific task. Even if the model works well on general English sentiment, it may fail when applied to financial language.

### 7.3 FinBERT Results

FinBERT achieved an accuracy of 0.9691 and a weighted F1-score of 0.9695. This is a very strong result and confirms that domain-specific models are highly effective for financial sentiment analysis.

FinBERT performed well because it is adapted to financial language. It can classify all three labels: negative, neutral, and positive. Its predictions were well balanced across the test set:

| Predicted Label | Count |
|---|---:|
| negative | 62 |
| neutral | 267 |
| positive | 124 |

This distribution is close to the true test distribution:

| True Label | Count |
|---|---:|
| negative | 61 |
| neutral | 278 |
| positive | 114 |

The similarity between predicted and true distributions suggests that FinBERT learned the structure of financial sentiment well.

### 7.4 Fine-tuned DistilBERT Results

Fine-tuned DistilBERT achieved the best result in the final comparison, with an accuracy of 0.9713 and a weighted F1-score of 0.9713. This result is very important because DistilBERT is a smaller model than BERT-large.

The strong performance shows that a small model can become highly competitive when it is fine-tuned on the correct dataset. Fine-tuned DistilBERT even slightly outperformed FinBERT in this experiment. This does not mean that DistilBERT is always better than FinBERT, but it shows that task-specific fine-tuning can be extremely effective.

The prediction distribution was:

| Predicted Label | Count |
|---|---:|
| negative | 65 |
| neutral | 277 |
| positive | 111 |

This distribution is also very close to the true test distribution, which explains the high performance.

### 7.5 Fine-tuned BERT-large Results

Fine-tuned BERT-large achieved an accuracy of 0.6137 and weighted F1-score of 0.4668. At first, this result may seem surprising because BERT-large is larger than DistilBERT. However, the prediction file showed that BERT-large predicted neutral for all 453 test examples.

The test set contains 278 neutral examples out of 453. This means that predicting neutral for every sentence gives:

```text
278 / 453 = 0.6137
```

This exactly matches the BERT-large accuracy. Therefore, the BERT-large model collapsed into predicting the majority class.

This is one of the most important findings of the thesis. A larger model is not automatically better. Several factors may explain this result:

1. The dataset is relatively small for fine-tuning a large model.
2. The class distribution is imbalanced, with neutral as the majority class.
3. The model may require more careful hyperparameter tuning.
4. The training setup used a small batch size because of hardware limits.
5. The model may have learned that predicting the majority class minimizes loss enough to achieve moderate accuracy.

This result is useful academically because it shows that evaluation must include more than accuracy. If only accuracy were considered, 0.6137 might appear acceptable. However, the F1-score of 0.4668 and the prediction distribution reveal that the model did not learn the minority classes properly.

### 7.6 Main Interpretation

The main conclusion from the final comparison is that model size alone is not enough. The fine-tuned small model and the domain-specific model performed best. The large model performed worse because it did not learn a balanced classification behavior.

This supports the idea that practical NLP model selection should consider:

- domain adaptation
- dataset size
- class balance
- fine-tuning strategy
- inference speed
- memory usage
- deployment needs

The best model for this task was not the largest model. The best models were those that were either adapted to the financial domain or fine-tuned effectively on the target dataset.

---

## 8. Epoch Experiment

### 8.1 Purpose of the Experiment

The epoch experiment tested how the number of training epochs affects fine-tuned DistilBERT performance. An epoch means one full pass through the training dataset. Training for more epochs can improve performance, but it can also lead to overfitting.

The experiment tested:

- 1 epoch
- 2 epochs
- 3 epochs

The same training and test data were used for each run.

### 8.2 Results

**Table 2. DistilBERT epoch experiment**

| Epochs | Train Loss | Eval Loss | Accuracy | Precision | Recall | F1-score | Training Time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4443 | 0.1441 | 0.9603 | 0.9612 | 0.9603 | 0.9604 | 46.14 |
| 2 | 0.1156 | 0.0931 | 0.9735 | 0.9734 | 0.9735 | 0.9734 | 92.77 |
| 3 | 0.0553 | 0.1372 | 0.9603 | 0.9618 | 0.9603 | 0.9607 | 139.78 |

![Epoch score comparison](results/experiment_epochs_scores.png)

![Epoch loss comparison](results/experiment_epochs_loss.png)

![Epoch time comparison](results/experiment_epochs_time.png)

### 8.3 Analysis

The best result was achieved after two epochs. At two epochs, the model reached an accuracy of 0.9735 and a weighted F1-score of 0.9734. This was better than both one epoch and three epochs.

After three epochs, the training loss continued to decrease, but the evaluation loss increased from 0.0931 to 0.1372. This suggests that the model may have started to overfit. Overfitting happens when a model learns the training data too specifically and becomes less effective on unseen data.

This experiment shows that more training is not always better. Training for three epochs took more time than two epochs but produced worse evaluation results. Therefore, two epochs provided the best balance between performance and training cost.

This is an important software engineering conclusion. In practical systems, it is not enough to train as long as possible. The training process should be evaluated carefully to find the point where performance and cost are balanced.

---

## 9. Dataset Size Experiment

### 9.1 Purpose of the Experiment

The dataset size experiment tested how much training data DistilBERT needs to perform well. The model was trained using different fractions of the training set:

- 25%
- 50%
- 75%
- 100%

This experiment answers the question: how much data does a small model need to improve?

### 9.2 Results

**Table 3. DistilBERT dataset size experiment**

| Training Fraction | Training Examples | Accuracy | Precision | Recall | F1-score | Training Time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 25% | 452 | 0.9272 | 0.9300 | 0.9272 | 0.9277 | 33.64 |
| 50% | 905 | 0.9647 | 0.9656 | 0.9647 | 0.9649 | 67.68 |
| 75% | 1359 | 0.9470 | 0.9520 | 0.9470 | 0.9463 | 113.63 |
| 100% | 1811 | 0.9647 | 0.9648 | 0.9647 | 0.9647 | 152.93 |

![Dataset size score comparison](results/experiment_data_size_scores.png)

![Dataset size time comparison](results/experiment_data_size_time_bar.png)

![Dataset size efficiency comparison](results/experiment_data_size_efficiency.png)

### 9.3 Analysis

The results show that performance improved strongly from 25% to 50% of the training data. With only 25% of the training data, the model achieved an accuracy of 0.9272. With 50%, accuracy increased to 0.9647.

However, the results were not strictly increasing after that. The 75% training size produced lower performance than 50% and 100%. This can happen because fine-tuning performance depends not only on the amount of data but also on the exact examples included, optimization behavior, class distribution, and random variation.

The 100% dataset size reached the same accuracy as the 50% size, but required more training time. This suggests that DistilBERT can learn efficiently from a moderate amount of financial sentiment data. Additional data may still be useful, but the improvement is not always proportional to the cost.

This experiment is useful because it shows the trade-off between training cost and model performance. In real systems, data collection and labeling can be expensive. If a smaller amount of labeled data produces strong results, that can reduce development cost.

---

## 10. Confusion Matrices and Error Analysis

### 10.1 Purpose of Confusion Matrices

A confusion matrix shows how predictions are distributed across the true classes. It is useful because it shows which classes the model confuses.

Accuracy alone gives only one number. A confusion matrix gives more detail. For example, a model may have high accuracy because it predicts the majority class correctly, but it may fail on minority classes. This was the case with BERT-large.

### 10.2 FinBERT Confusion Matrix

![FinBERT confusion matrix](results/confusion_matrix_finbert.png)

FinBERT performed strongly across all classes. Its predictions were close to the true label distribution. The confusion matrix shows that the model was able to identify negative, neutral, and positive examples effectively.

### 10.3 Fine-tuned DistilBERT Confusion Matrix

![Fine-tuned DistilBERT confusion matrix](results/confusion_matrix_distilbert_ft.png)

Fine-tuned DistilBERT also performed very strongly. The model learned the task well and produced balanced predictions. This supports the conclusion that smaller models can be powerful when fine-tuned on the correct dataset.

### 10.4 BERT-large Confusion Matrix

![BERT-large confusion matrix](results/confusion_matrix_bert_large.png)

BERT-large predicted the neutral class for all test examples. The confusion matrix clearly shows that negative and positive examples were misclassified as neutral. This explains why the F1-score was much lower than accuracy.

This is a strong example of why multiple evaluation metrics are necessary. If only accuracy were reported, the BERT-large result might look moderate. The confusion matrix reveals the real problem.

---

## 11. Interactive Streamlit Prototype

### 11.1 Purpose of the User Interface

The project includes a Streamlit application. The purpose of the application is to make the system interactive and easier to demonstrate. Instead of only showing static results, the user can type financial text and see model predictions immediately.

The prototype makes the practical part stronger because it shows that the project is not only a research experiment. It is also a working software system.

### 11.2 Main Features

The Streamlit UI includes:

- product dashboard
- live demo
- model comparison
- confidence scores
- analyst summary
- batch analyzer
- final results table
- graphs
- epoch experiment results
- dataset size experiment results
- confusion matrices
- prediction file previews
- CSV export

The live demo allows the user to enter a financial sentence such as:

```text
The company reported strong revenue growth.
```

The system then shows predictions from the available models. It displays the predicted label, confidence, and whether the models agree.

### 11.3 Product-Oriented Design

The UI was improved to make it feel like a financial sentiment intelligence prototype. It includes a product dashboard that explains possible use cases:

- financial news monitoring
- analyst decision support
- automatic sentiment tagging
- model comparison and governance

The batch analyzer is especially useful because it shows how the system could be used in practice. A user can enter multiple financial sentences, analyze them together, view sentiment distribution, and export the result.

### 11.4 Value of the Prototype

The prototype demonstrates the practical value of the thesis. It allows the committee or a potential user to test the system directly. This helps explain the results more clearly than tables alone.

The UI also connects the research part with a software engineering outcome. It shows how model evaluation can become a usable application.

---

## 12. Discussion

### 12.1 Small Models Can Be Competitive

One of the most important findings is that fine-tuned DistilBERT performed extremely well. It achieved the highest accuracy and F1-score in the final comparison.

This shows that small models should not be underestimated. A smaller model can be easier to deploy, faster to run, and cheaper to use. If it is fine-tuned on the correct dataset, it can match or even outperform larger or domain-specific models.

This finding is important for software engineering because many real systems have resource constraints. A company may not want to deploy a very large model if a smaller model provides similar results.

### 12.2 Domain Adaptation Matters

FinBERT also performed very well. This confirms that domain adaptation is important. Financial language has special patterns that general models may not understand.

The large difference between the general sentiment model and FinBERT shows that using a model trained for the correct domain can dramatically improve results.

### 12.3 Larger Models Are Not Automatically Better

BERT-large did not perform as expected. Instead of outperforming DistilBERT, it predicted neutral for all examples. This result is not a failure of the thesis; it is actually an important research finding.

The result shows that larger models require careful training. They may need:

- more data
- better class balancing
- hyperparameter tuning
- different learning rates
- better validation strategy
- longer or more controlled training

The result also shows that evaluation must go beyond accuracy. The BERT-large accuracy matched the neutral majority class percentage, while the F1-score and confusion matrix showed poor class behavior.

### 12.4 Accuracy vs Efficiency

The project compared not only accuracy but also inference time and memory usage. This is important because real software systems must run efficiently.

Fine-tuned DistilBERT achieved the best accuracy and also had fast inference time. FinBERT achieved almost the same accuracy but took longer. BERT-large took the longest evaluation time and performed worse.

This supports the conclusion that fine-tuned small models can be attractive for practical deployment.

### 12.5 Importance of Controlled Experiments

The epoch and dataset size experiments made the thesis stronger because they tested model behavior under different conditions.

The epoch experiment showed that two epochs were better than three. The dataset size experiment showed that 50% of the training data already produced strong performance. These findings provide deeper analysis than a single final comparison.

---

## 13. Limitations

Every practical experiment has limitations. The main limitations of this thesis are:

### 13.1 Dataset Size

The dataset is relatively small. The training set contains 1811 examples and the test set contains 453 examples. This is enough for a controlled thesis experiment, but larger datasets would make the conclusions more robust.

### 13.2 Class Imbalance

The neutral class is much larger than the negative and positive classes. This affects model behavior. BERT-large collapsed into predicting the neutral class for all examples, showing that class imbalance can strongly influence results.

### 13.3 Limited Hyperparameter Tuning

The models were not tuned exhaustively. Only a limited set of learning rates, batch sizes, epoch counts, and training sizes were tested. More extensive tuning could improve BERT-large or change the final comparison.

### 13.4 Hardware Constraints

BERT-large required more computational resources and was trained in Google Colab using a T4 GPU. Hardware limits affected the batch size and training setup.

### 13.5 Memory Measurement

Memory usage was measured using process memory differences. This provides useful practical information, but it is not the same as full GPU peak memory usage. Therefore, memory results should be interpreted carefully.

### 13.6 Single Dataset

The project used one financial sentiment dataset. Testing on additional financial datasets would make the conclusions stronger.

---

## 14. Future Work

Several improvements could be made in future work.

### 14.1 Larger Financial Datasets

Future experiments could use larger financial datasets. A larger dataset may help BERT-large learn better and avoid majority-class collapse.

### 14.2 Class Balancing

Future work could apply class weighting, oversampling, or undersampling to reduce the effect of class imbalance. This may improve negative and positive class performance.

### 14.3 More Hyperparameter Tuning

Additional experiments could test different learning rates, batch sizes, optimizers, and training schedules. This may improve the large model performance.

### 14.4 More Modern Models

Future work could include newer transformer models or instruction-tuned models. However, this should be done carefully to avoid adding models without clear research purpose.

### 14.5 API Deployment

The Streamlit prototype could be extended into a full API-based system. A backend API could receive financial text and return predictions in JSON format.

### 14.6 Multilingual Financial Sentiment

Future work could explore multilingual financial sentiment analysis, including Macedonian or regional financial text.

### 14.7 Real-Time News Monitoring

The prototype could be extended to process live news feeds. This would make the system more useful for analysts and business users.

---

## 15. Conclusion

This thesis presented a complete practical NLP project for financial sentiment analysis. The project compared small, large, and domain-specific language models using the Financial PhraseBank dataset.

The models evaluated were:

- a general pretrained sentiment model
- fine-tuned DistilBERT
- fine-tuned BERT-large
- FinBERT

The results showed that the general sentiment model performed poorly because it was not adapted to financial language and could not predict the neutral class. FinBERT performed very strongly because it is domain-specific. Fine-tuned DistilBERT achieved the best result, showing that a smaller model can become highly competitive when trained on the target dataset. BERT-large performed worse because it predicted the neutral majority class for all examples.

The final results support the main conclusion:

**The best model is not always the largest model. For financial sentiment classification, domain adaptation and task-specific fine-tuning are more important than model size alone.**

The epoch experiment showed that two epochs produced the best result for DistilBERT. The dataset size experiment showed that performance improved strongly from 25% to 50% of the training data, but additional data did not always produce proportional improvement.

The project also included a Streamlit prototype that makes the system interactive. The prototype allows live sentiment analysis, multi-model comparison, batch analysis, result visualization, and CSV export. This makes the practical part more complete and demonstrates how the research can be transformed into a usable software system.

Overall, the thesis combines theoretical understanding, machine learning implementation, software engineering structure, experimental evaluation, and interactive presentation. It demonstrates that careful model selection and evaluation are essential when applying NLP models to real-world domain-specific tasks.

---

## References

> Format these according to the official citation style required by the faculty.

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding.

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing.

---

## Appendix A: Project Files

The implementation is organized as follows:

```text
01_data.py
02_general_model.py
03_finbert_model.py
04_bert_model.py
04_bert_large_model.py
05_train_distilbert.py
06_evaluate_finetuned_distilbert.py
07_train_bert_large.py
08_experiment_epochs.py
09_experiment_data_size.py
10_generate_experiment_graphs.py
11_generate_presentation_assets.py
06_results.py
app.py
```

## Appendix B: Main Result Files

The main result files are:

```text
results/model_comparison.csv
results/experiment_epochs.csv
results/experiment_data_size.csv
results/predictions.csv
results/final_accuracy.png
results/final_f1.png
results/final_time.png
results/memory_comparison.png
results/main_summary_figure.png
```

## Appendix C: How to Run the Application

The Streamlit application can be started with:

```bash
streamlit run app.py
```

The application includes:

- product dashboard
- live model comparison
- batch analyzer
- final comparison tables
- experiment graphs
- confusion matrices
- prediction file previews
