# Checkpoint Answer Key

## Answers

1. Why does a PyTorch classifier emit logits instead of probabilities?
   Because `CrossEntropyLoss` expects raw logits and applies the stable softmax logic internally.
2. What problem does a `DataLoader` solve?
   It batches, shuffles, and iterates training data so the loop stays simple and repeatable.
3. Why can Adam outperform SGD early in training?
   It adapts update sizes per parameter, which often speeds up initial progress.
4. What does weight decay do?
   It discourages overly large weights, which can reduce overfitting.
5. How do you spot a helpful regularizer in this project?
   Validation accuracy improves or holds up better without destabilizing training loss.
