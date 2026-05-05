# Analysis Summary

## Final Model Comparison

The fine-tuned DistilBERT model and FinBERT achieved the strongest performance. Fine-tuned DistilBERT reached an accuracy and weighted F1-score of approximately 0.971, while FinBERT achieved approximately 0.969 on both metrics. This shows that a smaller model can become highly competitive when it is fine-tuned on the target dataset.

BERT-large achieved lower performance in this experiment and predicted the majority class, neutral, for all test examples. This is an important finding: a larger model is not automatically better. Domain adaptation and task-specific fine-tuning are more important than parameter count alone.

## Epoch Experiment

The epoch experiment shows that training for two epochs produced the best result. The third epoch did not improve performance and validation loss increased, which suggests that longer training can begin to overfit or become less useful after the model has already learned the task.

## Dataset Size Experiment

The dataset-size experiment shows that performance improved strongly from 25% to 50% of the training data. After that point, the gains became smaller and were not strictly monotonic. This suggests that DistilBERT can learn efficiently from a moderate amount of task-specific data, but additional data does not always guarantee a proportional improvement.

## Limitations

The dataset is relatively small and contains only one domain-specific sentiment task. BERT-large required significantly more computational resources and was trained under hardware constraints. The experiments used a limited set of hyperparameters, so the results should be interpreted as a controlled practical comparison rather than an exhaustive optimization study.

## Future Work

Future work could use larger financial datasets, multilingual financial data, more systematic hyperparameter tuning, API deployment, and additional modern language models. A larger evaluation set would also make the conclusions more robust.
