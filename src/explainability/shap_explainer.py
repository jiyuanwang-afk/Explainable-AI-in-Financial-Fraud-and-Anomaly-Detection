import shap, numpy as np

def shap_feature_importance(model, X_background, X_sample, max_display=10):
    e = shap.Explainer(model.predict_proba, X_background, algorithm='permutation')
    sv = e(X_sample)
    # Return mean |shap| as importances
    importances = np.abs(sv.values).mean(axis=0).tolist()
    order = np.argsort(importances)[::-1][:max_display].tolist()
    return {'mean_abs_shap': importances, 'top_indices': order}
