"""
train_regression.py — Train & compare regression models for used-car price
prediction, tune the best one, and persist the final pipeline.

Models compared: Linear Regression, Ridge, Random Forest, XGBoost.
Tuning: RandomizedSearchCV (5-fold CV) on the best candidate.
Metrics: MAE, RMSE, R² — all computed on the *original* price scale
         (after expm1 inverse of log1p target).
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
IMG_DIR      = PROJECT_ROOT / "app" / "static" / "img"

IMG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _original_scale(y_log):
    """Reverse log1p transformation."""
    return np.expm1(y_log)


def run_regression():
    print("=" * 60)
    print("STEP 3 — Regression Training")
    print("=" * 60)

    # Load artefacts
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    splits       = joblib.load(PROCESSED / "splits.joblib")

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    # Transform features
    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── 1. Compare base models with 5-fold CV ──────────────────────────────
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge":            Ridge(alpha=1.0),
        "RandomForest":     RandomForestRegressor(
                                n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":          XGBRegressor(
                                n_estimators=200, learning_rate=0.1,
                                random_state=42, n_jobs=-1,
                                verbosity=0),
    }

    results = {}
    print("\n[1/4] 5-fold CV comparison (neg-MAE on log scale) …")
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train_t, y_train,
            cv=5, scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        cv_mae = -scores.mean()
        print(f"  {name:25s}  CV MAE (log): {cv_mae:.4f}")
        results[name] = {"cv_mae_log": round(float(cv_mae), 4)}

    # Pick best by CV MAE (log scale)
    best_name = min(results, key=lambda k: results[k]["cv_mae_log"])
    print(f"\n  ➜ Best base model: {best_name}")

    # ── 2. Hyperparameter tuning on best model ─────────────────────────────
    print(f"\n[2/4] Tuning {best_name} with RandomizedSearchCV …")
    if best_name == "RandomForest":
        param_dist = {
            "n_estimators":  [100, 200, 400, 600],
            "max_depth":     [10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2", 0.5],
        }
        base = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif best_name == "XGBoost":
        param_dist = {
            "n_estimators":  [200, 400, 600, 800],
            "max_depth":     [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample":     [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha":     [0, 0.01, 0.1],
            "reg_lambda":    [1, 1.5, 2],
        }
        base = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    elif best_name == "Ridge":
        param_dist = {"alpha": [0.01, 0.1, 1, 10, 100]}
        base = Ridge()
    else:
        # Linear Regression has no hyper-params — just refit
        param_dist = {}
        base = LinearRegression()

    if param_dist:
        search = RandomizedSearchCV(
            base, param_dist,
            n_iter=30, cv=5,
            scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=-1, verbose=0,
        )
        search.fit(X_train_t, y_train)
        best_model = search.best_estimator_
        print(f"  Best params: {search.best_params_}")
    else:
        best_model = base
        best_model.fit(X_train_t, y_train)

    # ── 3. Evaluate on test set (original scale) ───────────────────────────
    print("\n[3/4] Evaluating on test set (original ₹ scale) …")
    y_pred_log  = best_model.predict(X_test_t)
    y_pred_orig = _original_scale(y_pred_log)
    y_test_orig = _original_scale(y_test)

    mae  = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    r2   = r2_score(y_test_orig, y_pred_orig)

    print(f"  MAE  : ₹{mae:,.0f}")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  R²   : {r2:.4f}")

    # Also evaluate all models on test set for comparison table
    comparison = []
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        yp = _original_scale(model.predict(X_test_t))
        yt = y_test_orig
        comparison.append({
            "model": name,
            "MAE":   round(float(mean_absolute_error(yt, yp)), 2),
            "RMSE":  round(float(np.sqrt(mean_squared_error(yt, yp))), 2),
            "R2":    round(float(r2_score(yt, yp)), 4),
        })

    # Add tuned best
    comparison.append({
        "model": f"{best_name} (tuned)",
        "MAE":   round(float(mae), 2),
        "RMSE":  round(float(rmse), 2),
        "R2":    round(float(r2), 4),
    })

    metrics_out = {
        "best_model": best_name,
        "test_MAE":   round(float(mae), 2),
        "test_RMSE":  round(float(rmse), 2),
        "test_R2":    round(float(r2), 4),
        "comparison": comparison,
    }

    with open(MODELS_DIR / "regression_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ── 4. Save the best pipeline ──────────────────────────────────────────
    print("\n[4/4] Saving best regression pipeline …")
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    best_model),
    ])
    joblib.dump(full_pipeline, MODELS_DIR / "regressor.joblib")

    # ── 5. Plots ───────────────────────────────────────────────────────────
    # Actual vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_test_orig, y_pred_orig, alpha=0.3, s=10, color="#6366f1")
    lim = max(y_test_orig.max(), y_pred_orig.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1)
    ax.set_xlabel("Actual Price (₹)")
    ax.set_ylabel("Predicted Price (₹)")
    ax.set_title("Actual vs Predicted")
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)

    # Residuals
    residuals = y_test_orig - y_pred_orig
    ax = axes[1]
    ax.scatter(y_pred_orig, residuals, alpha=0.3, s=10, color="#f59e0b")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Price (₹)")
    ax.set_ylabel("Residual (₹)")
    ax.set_title("Residual Plot")

    plt.tight_layout()
    plt.savefig(IMG_DIR / "regression_plots.png", dpi=150)
    plt.close()

    print("\n✓ Regression complete. Model saved to models/regressor.joblib")
    return best_model, metrics_out


if __name__ == "__main__":
    run_regression()
