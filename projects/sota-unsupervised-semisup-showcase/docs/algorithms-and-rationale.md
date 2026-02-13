# Algorithm Intuition Guide

Use this file when you want quick intuition, practical use cases, and "what to try next".

## 1) Unsupervised Learning

### K-Means
What it does:
- Groups points around center points (centroids).

When it works well:
- Clusters are roughly compact/spherical.
- You can guess a reasonable number of groups.

Where you might use it:
- Retail: customer segments from spend behavior.
- Healthcare: patient groups from lab measurements.
- Finance: account usage profiles.

Why it is in this project:
- It is the simplest strong baseline and makes cluster intuition clear.

### MiniBatch K-Means
What it does:
- Same idea as K-Means, but with mini-batches for speed.

When to use:
- Larger datasets where full K-Means is too slow.

### DBSCAN / HDBSCAN
What they do:
- Find dense regions and mark sparse points as noise.

When to use:
- Cluster shapes are irregular.
- You care about outliers as a first-class output.

Domain examples:
- Fraud detection (finance), defect detection (manufacturing), unusual visit patterns (healthcare).

### Agglomerative / BIRCH / Mean-Shift / Affinity / Spectral
These give different clustering assumptions:
- Agglomerative: hierarchical merges.
- BIRCH: memory-efficient summary tree.
- Mean-Shift: density modes.
- Affinity: exemplar-based clusters.
- Spectral: graph/eigen-space clusters.

Why include many methods:
- Real data rarely follows one perfect cluster shape.

### Gaussian Mixture + Bayesian Gaussian Mixture
What they do:
- Soft clustering: each point gets probabilities for each cluster.

When useful:
- You want uncertainty, not just hard assignments.
- You need density estimates and cluster overlap handling.

## 2) Anomaly Detection

### Isolation Forest
Intuition:
- Anomalies are easier to isolate with random tree splits.

### Local Outlier Factor (LOF)
Intuition:
- A point is odd if its local neighborhood is much less dense than nearby neighborhoods.

### One-Class SVM
Intuition:
- Learn a boundary around "normal" data; outside is suspicious.

### Fast-MCD / Elliptic Envelope
Intuition:
- Fit a robust covariance ellipse and flag distant points.

### PCA Reconstruction Error
Intuition:
- If a point reconstructs poorly from a low-dimensional subspace, it may be unusual.

### GMM Density Thresholding
Intuition:
- Very low likelihood under the learned density model suggests anomaly.

## 3) Semi-Supervised Learning

### Supervised-Only-Labeled Baseline
Why start here:
- You need a fair baseline before claiming semi-supervised gains.

### Self-Training
How it works:
- Train on labeled data, pseudo-label high-confidence unlabeled points, retrain.

### Co-Training (Two Views)
How it works:
- Split features into two views, keep points where both models agree confidently.

### Label Spreading
How it works:
- Build a similarity graph; spread labels across nearby points.

### Cluster Pseudo-Labeling
How it works:
- Cluster unlabeled data and assign majority labels in each cluster.

### Autoencoder Features + Classifier
How it works:
- Learn compact representations from all data first, then classify with few labels.

## 4) Active Learning

### Random Sampling (baseline)
- Pick unlabeled points at random for labeling.

### Uncertainty Sampling
- Ask for labels where model confidence is lowest.

What you should look for:
- At same label budget, uncertainty usually reaches higher accuracy sooner.

Domain examples:
- Medical triage: review uncertain cases first.
- Credit review: prioritize uncertain loan approvals.
- Product moderation: label ambiguous content first.

## 5) Self-Supervised Learning

### Contrastive Representation Learning
Intuition:
- Teach model that two augmented views of the same sample should be close in embedding space.

Why it helps:
- Learns strong features before labels are used.

### Linear Probe
Intuition:
- Freeze embedding, train a simple classifier.
- If this works well, embeddings are genuinely useful.

### LightGBM on Embeddings
Intuition:
- Test whether tree models can extract more signal from learned embeddings.

## 6) Deep Embedded Clustering (DEC)

How to think about DEC:
1. Pretrain an autoencoder.
2. Cluster in latent space.
3. Fine-tune latent space to sharpen cluster assignments.

Why it matters:
- Bridges deep representation learning and clustering in one objective.

## 7) Which Method Should You Try First?

1. Start with K-Means + anomaly baselines.
2. Add semi-supervised methods if labels are scarce.
3. Add active learning if labeling is expensive.
4. Add self-supervised learning when feature quality limits performance.
5. Use DEC when you want deeper cluster structure refinement.
