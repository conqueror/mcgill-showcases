# Domain Use Cases and Intuitive Examples

This guide helps you transfer each concept to real problems.

## 1) Clustering

What it means:
- "Find natural groups without labels."

### Retail
- Problem: shoppers behave differently but no segment labels exist.
- Input: spend frequency, basket size, discount usage.
- Output: shopper groups for personalized campaigns.

### Healthcare
- Problem: symptom patterns vary, diagnosis labels are incomplete.
- Input: labs, vitals, visit counts.
- Output: patient phenotype clusters for care pathways.

### Finance
- Problem: accounts have different behavior profiles.
- Input: payment habits, utilization, transaction patterns.
- Output: risk-style groups for monitoring and policy tuning.

### Manufacturing
- Problem: machines run in multiple operational states.
- Input: sensor traces, temperature, vibration statistics.
- Output: operating-state clusters and maintenance groups.

## 2) Anomaly Detection

What it means:
- "Find points that do not look normal."

### Retail
- Example: unusual basket combinations that may indicate promo abuse.

### Healthcare
- Example: rare vital-sign trajectory indicating early deterioration.

### Finance
- Example: transaction pattern unlike account history (potential fraud).

### Manufacturing
- Example: sensor spike patterns not seen during healthy operation.

## 3) Semi-Supervised Learning

What it means:
- "Use a small labeled set plus a large unlabeled set."

### Retail
- Few manually tagged product intents, many untagged session logs.

### Healthcare
- Few expert-labeled clinical outcomes, many unlabeled records.

### Finance
- Few verified default/fraud labels, many unresolved cases.

### Manufacturing
- Few confirmed fault labels, many unlabeled event windows.

## 4) Active Learning

What it means:
- "Spend labeling budget where it gives most model improvement."

### Retail
- Ask humans to label the most uncertain customer sessions.

### Healthcare
- Ask clinicians to review uncertain patient cases first.

### Finance
- Send uncertain loan applications to expert underwriters.

### Manufacturing
- Ask engineers to label uncertain fault events before routine cases.

## 5) Self-Supervised Learning

What it means:
- "Learn useful representations before using labels."

### Retail
- Learn embeddings from session behavior sequences.

### Healthcare
- Learn patient timeline embeddings from event order and context.

### Finance
- Learn account embeddings from transaction histories.

### Manufacturing
- Learn sensor embeddings from unlabeled time windows.

## 6) DEC (Deep Embedded Clustering)

What it means:
- "Learn a representation and cluster it jointly to sharpen group structure."

### Retail
- Sharpen broad shopper groups into actionable micro-segments.

### Healthcare
- Refine broad phenotypes into clinically meaningful subtypes.

### Finance
- Refine coarse risk groups into more precise monitoring tiers.

### Manufacturing
- Refine coarse machine states into detailed fault-precursor states.

## 7) Practical "Which One First?" Rule

1. Start with clustering + anomaly baselines.
2. Add semi-supervised methods when labels are scarce.
3. Add active learning when labeling costs are high.
4. Add self-supervised learning when feature quality is bottleneck.
5. Use DEC when cluster boundaries are still weak.
