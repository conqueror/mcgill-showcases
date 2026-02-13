# Self-Study Learning Guide (75-90 Minutes)

This guide helps you learn the full workflow on your own.

## Your Goal

You should finish this guide able to:
- explain each learning paradigm in plain language,
- run all modules end-to-end,
- interpret metrics and figures,
- decide which method to use for a new real-world problem.

## Before You Start

1. `cd projects/sota-unsupervised-semisup-showcase`
2. `uv sync --extra dev`
3. Run one mode:
   - `uv run sota-showcase --dataset digits`
   - `uv run sota-showcase --dataset business`

## Learning Flow

### 1. Understand the dataset first (10 minutes)
- Open run summary:
  - `artifacts/reports/<mode>_run_summary.json`
- Check:
  - number of samples,
  - number of features,
  - number of classes.

### 2. Learn unsupervised clustering (15 minutes)
- Open:
  - `artifacts/reports/<mode>_clustering_metrics.csv`
  - `artifacts/figures/<mode>_kmeans_selection.png`
- Questions to answer:
  1. Which algorithm wins on ARI?
  2. What `k` region looks strongest in silhouette?
  3. Why might the winner differ between digits and business data?

### 3. Learn anomaly detection (10 minutes)
- Open `artifacts/reports/<mode>_anomaly_metrics.csv`
- Questions to answer:
  1. Which method has best recall?
  2. Which method has best precision?
  3. Which one would you choose if false alarms are expensive?

### 4. Learn semi-supervised learning (15 minutes)
- Open `artifacts/reports/<mode>_semi_supervised_metrics.csv`
- Compare with low-label baseline.
- Questions to answer:
  1. Which method gives the biggest gain over supervised-only-labeled?
  2. Is label propagation stable for this dataset?

### 5. Learn active learning (15 minutes)
- Open:
  - `artifacts/reports/<mode>_active_learning_metrics.csv`
  - `artifacts/figures/<mode>_active_learning_curve.png`
- Questions to answer:
  1. Does `uncertainty` beat `random` at the same label budget?
  2. At what budget do improvements slow down?

### 6. Learn self-supervised transfer (15 minutes)
- Open:
  - `artifacts/reports/<mode>_self_supervised_metrics.csv`
  - `artifacts/figures/<mode>_contrastive_embeddings_train_pca.png`
- Questions to answer:
  1. Do learned embeddings help linear models?
  2. When do raw features still win?

### 7. Learn DEC as a capstone (10 minutes)
- Open `artifacts/reports/<mode>_dec_metrics.csv`
- Questions to answer:
  1. Does DEC beat KMeans on pretrained latent space?
  2. How sensitive is DEC to training epochs?

## Try These Experiments

1. Less labels:
   - `uv run sota-showcase --dataset digits --labeled-fraction 0.05`
2. Faster active learning loop:
   - `uv run sota-showcase --dataset business --active-learning-rounds 4 --active-learning-query-size 15`
3. Faster deep modules:
   - `uv run sota-showcase --dataset business --contrastive-epochs 2 --dec-pretrain-epochs 2 --dec-finetune-epochs 2`

## What Good Understanding Looks Like

You can explain:
- why clustering works even without labels,
- why active learning can reduce labeling cost,
- why representation learning can help downstream tasks,
- why different datasets favor different methods.
