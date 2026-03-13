# Checkpoint Answer Key

## Answers

1. Why can a perceptron fail on XOR?
   Because a single linear boundary cannot isolate the alternating class pattern.
2. Why is zero initialization a problem in hidden layers?
   Every neuron starts identically, receives identical gradients, and stays redundant.
3. Why does cross-entropy usually fit classification better than MSE?
   It reacts more strongly when the model is confidently wrong about a class probability.
4. What does backpropagation actually pass backward?
   It passes error-derived gradient signals so each parameter knows how changing it would affect loss.
5. How do you spot overfitting in this project?
   Training accuracy keeps rising while validation accuracy stalls or falls, creating a widening gap.
