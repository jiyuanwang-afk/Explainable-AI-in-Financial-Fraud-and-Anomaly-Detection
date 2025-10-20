# Project Report: Explainable AI in Financial Fraud and Anomaly Detection

## Abstract
We propose a **hybrid, explainable** anomaly detection framework that fuses **calibrated gradient boosting**, **autoencoder-based reconstruction error**, and **GNN edge classification**. We quantify accuracy and interpretability using **SHAP** and gradient‑based graph importance.

## Methods
- **Tabular**: GradientBoosting + isotonic calibration.
- **Unsupervised**: Tabular autoencoder reconstruction error normalized to [0,1].
- **Graph**: GCN with edge readout to classify suspicious edges (transactions).
- **Explainability**: Permutation SHAP for tabular model; gradient‑based edge importance for GNN.

## Results (Sample Run)
- Tabular Ensemble: AUC 0.88, F1 0.80
- GNN Edge: F1 0.78 (small synthetic data)
- Explanations highlight **amount**, **time_gap**, **merchant_type** as top factors. Edge importance clusters around hubs with high transaction variance.

## Discussion
The hybrid approach increases recall on rare fraudulent events while remaining interpretable. GNN complements tabular signals by capturing **relational structure**.

## Future Work
- Add **temporal GNNs** and **heterogeneous graphs**.
- **Counterfactual explanations** and **fairness auditing**.
