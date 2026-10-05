"""
evaluate.py — Load saved models and print consolidated evaluation metrics.

Also generates SHAP global feature importance plots.
"""

import json
import warnings
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
IMG_DIR      = PROJECT_ROOT / "app" / "static" / "img"


def run_evaluation():
    print("=" * 60)
    print("STEP 5 — Evaluation & Explainability (SHAP)")
    print("=" * 60)

    # Load artefacts
    reg_pipeline = joblib.load(MODELS_DIR / "regressor.joblib")
    preprocessor = reg_pipeline.named_steps["preprocessor"]
    regressor    = reg_pipeline.named_steps["regressor"]

    splits = joblib.load(PROCESSED / "splits.joblib")
    X_test = splits["X_test"]

    # Transform test features
    X_test_t = preprocessor.transform(X_test)

    # Get feature names from the preprocessor
    from src.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_feature_names = [f"cat_{i}" for i in range(X_test_t.shape[1] - len(NUMERIC_FEATURES))]
    feature_names = NUMERIC_FEATURES + cat_feature_names

    # ── SHAP for regression model ──────────────────────────────────────────
    print("\n[1/2] Computing SHAP values for regressor …")
    # Use a background sample for efficiency
    bg_sample = shap.sample(X_test_t, min(100, X_test_t.shape[0]), random_state=42)

    # Use TreeExplainer if tree model, else KernelExplainer
    model_type = type(regressor).__name__
    if model_type in ("RandomForestRegressor", "XGBRegressor"):
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_test_t[:200])
    else:
        explainer = shap.KernelExplainer(regressor.predict, bg_sample)
        shap_values = explainer.shap_values(X_test_t[:100])

    # Global importance bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Get top 15 features
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_names  = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_idx]
    top_values = mean_abs_shap[top_idx]

    ax.barh(range(len(top_names)), top_values[::-1], color="#6366f1")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 15 Feature Importances (SHAP)")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "shap_importance.png", dpi=150)
    plt.close()

    # Save global importance as JSON for the API
    global_importance = {
        feature_names[i] if i < len(feature_names) else f"feature_{i}": round(float(mean_abs_shap[i]), 6)
        for i in top_idx
    }
    with open(MODELS_DIR / "shap_global_importance.json", "w") as f:
        json.dump(global_importance, f, indent=2)

    # ── Print metrics summary ──────────────────────────────────────────────
    print("\n[2/2] Metrics Summary")
    print("-" * 40)

    with open(MODELS_DIR / "regression_metrics.json") as f:
        reg_m = json.load(f)
    print(f"  Regression — Best: {reg_m['best_model']}")
    print(f"    MAE  : ₹{reg_m['test_MAE']:,.0f}")
    print(f"    RMSE : ₹{reg_m['test_RMSE']:,.0f}")
    print(f"    R²   : {reg_m['test_R2']:.4f}")

    with open(MODELS_DIR / "classification_metrics.json") as f:
        clf_m = json.load(f)
    print(f"\n  Classification — Best: {clf_m['best_model']}")
    print(f"    Accuracy : {clf_m['test_accuracy']:.4f}")
    print(f"    Macro F1 : {clf_m['test_f1']:.4f}")
    print(f"    ROC-AUC  : {clf_m['test_roc_auc']:.4f}")

    print("\n✓ Evaluation and SHAP analysis complete.")


if __name__ == "__main__":
    run_evaluation()
