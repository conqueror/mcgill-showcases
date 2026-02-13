# Concept Transfer: Different Domains and Use Cases

Use this guide to connect each concept to real-world problems beyond digit datasets.

## 1. Binary Classification

What it means intuitively:
- A yes/no decision.

Examples:
- Healthcare: "Will this patient be readmitted within 30 days?"
- Finance: "Will this borrower default?"
- Cybersecurity: "Is this login attempt malicious?"
- Manufacturing: "Will this machine fail in the next 24 hours?"

## 2. Multi-Class Classification

What it means intuitively:
- Exactly one label among many categories.

Examples:
- Retail: classify a product into one category.
- Support operations: route a ticket to billing, technical, or logistics.
- Agriculture: classify plant disease type from sensor/image features.
- Media: classify article topic.

## 3. Multi-Label Classification

What it means intuitively:
- One example can belong to multiple labels at once.

Examples:
- HR: employee skills tagging (`python`, `sql`, `ml`).
- Legal: contract tags (`nda`, `termination`, `payment_terms`).
- Content moderation: post can be `spam` and `toxicity` simultaneously.
- Music: track can be `jazz`, `instrumental`, and `live`.

## 4. Multi-Output Prediction

What it means intuitively:
- Predict several outputs together for each input.

Examples:
- Supply chain: predict demand for each region at once.
- Energy: forecast hourly consumption for multiple zones.
- Computer vision: denoise/reconstruct full image pixel vectors.
- Robotics: predict multiple actuator values in one step.

## 5. Regression

What it means intuitively:
- Predict a continuous number.

Examples:
- Real estate: home price estimation.
- Insurance: claim amount prediction.
- E-commerce: expected order value.
- Transportation: trip duration prediction.

## 6. Imbalanced Classes

What it means intuitively:
- Important events are rare, so naive models may ignore them.

Examples:
- Fraud detection: fraud is usually a tiny fraction of transactions.
- Healthcare screening: severe disease cases are rarer than healthy cases.
- Equipment failure: outages are infrequent but high cost.
- Safety incidents: accidents are rare compared to normal operations.

Why it matters:
- High accuracy can still be useless if rare but critical cases are missed.

## 7. Ensemble Methods

What it means intuitively:
- Combine multiple imperfect models to get a stronger overall decision.

Examples:
- Credit scoring: blend linear and tree models for stability and nonlinear capture.
- Recommendations: combine collaborative and content-based predictors.
- Forecasting: blend global and local models for better robustness.
- Diagnosis support: ensemble improves consistency across patient subgroups.

## 8. Model Selection Curves

What it means intuitively:
- Curves tell you whether to tune model complexity or collect more data.

Examples:
- Marketing uplift model: validation curve picks useful depth/regularization.
- Churn model: learning curve shows if more labeled data may improve recall.
- Demand forecasting: curve behavior reveals if model is capacity-limited.

## How to Use This File While Learning

For each artifact you inspect:
1. Name the task type.
2. Map it to one domain example above.
3. Explain metric choice in one sentence.
4. State one business action if the model is good.
