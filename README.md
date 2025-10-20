# Explainable AI in Financial Fraud and Anomaly Detection

This repository implements a **hybrid, explainable fraud/anomaly detection pipeline** that combines:
- **Graph Neural Networks (GNNs)** over a transaction graph
- **Tabular Gradient Boosting** (baseline) for calibrated probabilities
- **Autoencoder** for unsupervised anomaly scoring
- **Post‑hoc explainability** via **SHAP**, **Permutation Importance**, and **GNNExplainer-like** node/edge importance
- **Drift tracking** (population stability index, PSI) and **threshold optimization** (F1/PR-AUC maximizing)

It includes a **configurable training CLI**, **reproducible experiments**, and a **lightweight unit test suite**.

## Quickstart
```bash
pip install -r requirements.txt
python -m src.experiments.main --config configs/default_config.json
# visualize
python -m src.experiments.evaluate --artifacts results/run_000
```

## Project Structure
```
explainable-ai-fraud-detection/
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.json
├── data/
│   └── transactions.csv                # sample synthetic data
├── src/
│   ├── data/
│   │   ├── build_graph.py
│   │   └── tabular_features.py
│   ├── models/
│   │   ├── gnn_model.py
│   │   ├── autoencoder.py
│   │   └── tabular_baseline.py
│   ├── explainability/
│   │   ├── shap_explainer.py
│   │   └── gnn_explainer_like.py
│   ├── experiments/
│   │   ├── main.py
│   │   └── evaluate.py
│   ├── utils.py
│   └── metrics.py
├── docs/
│   └── project_report.md
├── results/
│   └── (artifacts saved here)
└── tests/
    └── test_shapes.py
```

## Reproducible Runs
- All hyperparameters and seeds are in `configs/default_config.json`.
- Artifacts (metrics, confusion matrix, feature importances, explainability plots) are saved under `results/run_xxx`.

## Notes
- `torch-geometric` installation depends on your local CUDA/CPU stack. See https://pytorch-geometric.readthedocs.io for wheels.
- If `torch-geometric` is unavailable, you can run **tabular + autoencoder** only by setting `"use_gnn": false` in the config.
