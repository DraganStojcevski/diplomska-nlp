# McNemar Pairwise Tests

Two-sided exact binomial McNemar test on the 453-sentence test set.

- `b` = #(Model A correct, Model B wrong)
- `c` = #(Model A wrong, Model B correct)
- p < 0.05 = statistically significant difference between the two classifiers' errors

## Results

| Pair | b | c | p-value | Significant? | Winner |
|------|---:|---:|---:|:---:|:---|
| DistilBERT FT vs FinBERT | 9 | 8 | 1.000 | no | DistilBERT FT |
| BERT-large retrain vs DistilBERT FT | 8 | 2 | 0.109 | no | BERT-large retrain |
| BERT-large retrain vs FinBERT | 12 | 5 | 0.143 | no | BERT-large retrain |
| BERT-large retrain vs Classical baseline | 59 | 5 | < 0.001 | yes | BERT-large retrain |
| DistilBERT FT vs Classical baseline | 55 | 7 | < 0.001 | yes | DistilBERT FT |
| FinBERT vs Classical baseline | 58 | 11 | < 0.001 | yes | FinBERT |

## Questions answered

- **Are these statistically equivalent? (multi-seed suggested tie)**
  - Verdict: no (p = 1.000), winner: DistilBERT FT
- **Does corrected BERT-large beat the small fine-tuned model?**
  - Verdict: no (p = 0.109), winner: BERT-large retrain
- **Does corrected BERT-large beat the domain-pretrained model?**
  - Verdict: no (p = 0.143), winner: BERT-large retrain
- **Does the best transformer beat the classical floor?**
  - Verdict: yes (p = < 0.001), winner: BERT-large retrain
- **Does fine-tuned DistilBERT beat the classical floor?**
  - Verdict: yes (p = < 0.001), winner: DistilBERT FT
- **Does FinBERT beat the classical floor?**
  - Verdict: yes (p = < 0.001), winner: FinBERT
