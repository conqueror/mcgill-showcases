# Concept Learning Map

## Concept Map

| Concept | Code Anchor | Artifact | Question To Ask |
| --- | --- | --- | --- |
| Perceptron | `src/neural_network_foundations_showcase/networks.py` | `decision_boundary_summary.csv` | When is one linear boundary enough? |
| Activation functions | `src/neural_network_foundations_showcase/activations.py` | `activation_comparison.csv` | Which functions saturate or keep gradients alive? |
| Loss functions | `src/neural_network_foundations_showcase/losses.py` | `loss_function_comparison.csv` | How harshly does each loss punish the same mistake? |
| Backpropagation | `src/neural_network_foundations_showcase/backprop.py` | `backprop_gradient_trace.csv` | Which layer gets the strongest update signal? |
| Initialization | `src/neural_network_foundations_showcase/networks.py` | `initialization_comparison.csv` | Why does zero initialization create symmetry problems? |
| Fitting regimes | `src/neural_network_foundations_showcase/training.py` | `underfit_overfit_examples.csv` | Are both train and validation improving together? |
| Decision boundaries | `src/neural_network_foundations_showcase/plots.py` | `decision_boundaries.png` | What shape can the model carve into feature space? |

## Study Hint

If a result feels surprising, go back one layer of abstraction:

- from plot to CSV,
- from CSV to helper function,
- from helper function to the math idea it encodes.
