from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true))>1 else 0.0,
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def optimal_threshold(y_true, y_prob, strategy="f1_max"):
    best_t, best_f1 = 0.5, -1
    for t in [i/100 for i in range(1,100)]:
        f1 = f1_score(y_true, (y_prob>=t).astype(int))
        if f1>best_f1: best_f1, best_t = f1, t
    return best_t, best_f1
