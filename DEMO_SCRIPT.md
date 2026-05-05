# Product Demo Script

## 1. Opening

This project is a financial sentiment intelligence prototype.

It analyzes financial text and compares multiple language models so an analyst can see:

- the predicted sentiment
- the model confidence
- whether models agree or disagree
- which model is most accurate and efficient

The goal is not only to classify text, but to support a real decision workflow.

## 2. Problem

Financial news, reports, and announcements are produced very quickly.

Manually reading every sentence is slow, especially when analysts need to identify positive, negative, or neutral signals quickly.

This system helps by automatically classifying financial text and highlighting uncertain cases for review.

## 3. Live Demo

Open the Live Demo tab.

Test these examples:

- The company reported strong revenue growth.
- The company announced major losses.
- The company will hold its annual meeting next week.

Explain:

- FinBERT is domain-specific.
- DistilBERT is the smaller fine-tuned model.
- BERT-large is the larger fine-tuned model.
- The app compares their predictions on the same input.

## 4. Batch Analyzer

Open the Batch Analyzer tab.

This shows how the system could be used on multiple financial headlines or report sentences.

The result can be exported as CSV, which makes the system useful for further reporting or analysis.

## 5. Results

Open the Final Comparison tab.

Explain that the project evaluates models using:

- accuracy
- F1-score
- inference time
- memory usage

The most important conclusion is:

Large models are not automatically better. Domain-specific training and fine-tuning matter.

## 6. Business Value

This prototype could be used for:

- financial news monitoring
- earnings report analysis
- analyst decision support
- automatic sentiment tagging
- research comparison of NLP models

## 7. Closing

The project combines software engineering and machine learning:

- dataset preparation
- model evaluation
- fine-tuning
- controlled experiments
- visual analysis
- interactive product UI

This makes it more than an experiment. It is a working prototype that demonstrates practical value.
