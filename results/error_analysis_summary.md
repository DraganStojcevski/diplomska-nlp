# Error Analysis

Test set size: 453 sentences. Top three models analyzed: FinBERT, DistilBERT FT, BERT-large retrain.

Total sentences mis-classified by at least one of these models: **24**.

## Per-model error counts

| Model | Errors | Error rate |
|-------|------:|-----------:|
| FinBERT | 14 | 3.1% |
| DistilBERT FT | 13 | 2.9% |
| BERT-large retrain | 7 | 1.5% |

## Error patterns (true → predicted, top 5 per model)

### FinBERT

| True | Predicted | Count |
|------|-----------|------:|
| neutral | positive | 9 |
| negative | positive | 2 |
| neutral | negative | 2 |
| positive | negative | 1 |

### DistilBERT FT

| True | Predicted | Count |
|------|-----------|------:|
| positive | negative | 5 |
| neutral | positive | 4 |
| positive | neutral | 3 |
| negative | positive | 1 |

### BERT-large retrain

| True | Predicted | Count |
|------|-----------|------:|
| neutral | positive | 3 |
| positive | negative | 2 |
| neutral | negative | 1 |
| positive | neutral | 1 |

## Sentences mis-classified by ALL 3 top models

These are the hardest sentences in the test set — three independent model approaches all fail. (2 total; first 10 shown.)

- **True: positive** — "Unit costs for flight operations fell by 6.4 percent ."
  - FinBERT: predicted *negative*
  - DistilBERT FT: predicted *negative*
  - BERT-large retrain: predicted *negative*

- **True: neutral** — "`` These developments partly reflect the government 's higher activity in the field of dividend policy . ''"
  - FinBERT: predicted *positive*
  - DistilBERT FT: predicted *positive*
  - BERT-large retrain: predicted *positive*

