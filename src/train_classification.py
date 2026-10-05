"""
train_classification.py — Deal classifier: Underpriced / Fair / Overpriced.

Approach:
  1. Use cross_val_predict (5-fold) on the training set with the regressor
     to get leak-free predicted prices.
  2. ratio = actual_price / predicted_price
       - ratio < 0.90  → Underpriced  (seller is asking less than fair value)
       - 0.90 ≤ ratio ≤ 1.10 → Fair
       - ratio > 1.10  → Overpriced
  3. Train classifiers on car features + listed_price.
  4. Compare Logistic Regression, Random Forest, SVM, XGBoost.
  5. Handle class imbalance with class_weight="balanced" or scale_pos_weight.
  6. Report accuracy, macro P/R/F1, OvR ROC-AUC, confusion matrix.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
IMG_DIR      = PROJECT_ROOT / "app" / "static" / "img"


def _label_deal(ratio):
    if ratio < 0.90:
        return "Underpriced"
    elif ratio > 1.10:
        return "Overpriced"
    else:
        return "Fair"


def run_classification():
    print("=" * 60)
    print("STEP 4 — Classification Training")
    print("=" * 60)

    # Load artefacts
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    regressor_pipeline = joblib.load(MODELS_DIR / "regressor.joblib")
    splits = joblib.load(PROCESSED / "splits.joblib")

    X_train = splits["X_train"].copy()
    X_test  = splits["X_test"].copy()
    y_train_log = splits["y_train"]
    y_test_log  = splits["y_test"]

    # ── 1. Out-of-fold predictions on training data ────────────────────────
    print("\n[1/5] Generating out-of-fold predictions …")
    # We use the raw regressor (not the pipeline) since data is already
    # preprocessed, but cross_val_predict needs the raw model
    # Actually, let's use the full pipeline so the preprocessor is included.
    oof_pred_log = cross_val_predict(
        regressor_pipeline, X_train, y_train_log, cv=5, n_jobs=-1,
    )

    y_train_orig = np.expm1(y_train_log)
    oof_pred_orig = np.expm1(oof_pred_log)

    # ── 2. Generate deal labels ────────────────────────────────────────────
    print("[2/5] Generating deal labels …")
    ratio = y_train_orig / oof_pred_orig.clip(min=1)
    labels_train = pd.Series(ratio).apply(_label_deal).values

    # Test labels (using regressor predictions)
    y_test_pred_log = regressor_pipeline.predict(X_test)
    y_test_orig = np.expm1(y_test_log)
    y_test_pred_orig = np.expm1(y_test_pred_log)
    ratio_test = y_test_orig / y_test_pred_orig.clip(min=1)
    labels_test = pd.Series(ratio_test).apply(_label_deal).values

    print(f"  Train label distribution:")
    for lbl in ["Underpriced", "Fair", "Overpriced"]:
        c = (labels_train == lbl).sum()
        print(f"    {lbl:12s}: {c:5d}  ({c/len(labels_train)*100:.1f}%)")

    # ── 3. Prepare classification features ─────────────────────────────────
    # Features = car features + listed price (= actual selling price here)
    # In deployment, listed_price comes from user input.
    X_train_clf = X_train.copy()
    X_train_clf["listed_price"] = y_train_orig.values

    X_test_clf = X_test.copy()
    X_test_clf["listed_price"] = y_test_orig.values

    # Build a new preprocessor that includes listed_price as a numeric column
    from src.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_preprocessor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    num_feats_clf = NUMERIC_FEATURES + ["listed_price"]
    numeric_pipeline_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline_clf = Pipeline([
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    clf_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline_clf,      num_feats_clf),
            ("cat", categorical_pipeline_clf,   CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    clf_preprocessor.fit(X_train_clf)
    X_train_t = clf_preprocessor.transform(X_train_clf)
    X_test_t  = clf_preprocessor.transform(X_test_clf)

    # Encode labels
    le = LabelEncoder()
    le.fit(["Underpriced", "Fair", "Overpriced"])
    y_train_enc = le.transform(labels_train)
    y_test_enc  = le.transform(labels_test)

    # ── 4. Compare classifiers ─────────────────────────────────────────────
    print("\n[3/5] Comparing classifiers (5-fold CV) …")
    classifiers = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric="mlogloss"),
    }

    cv_results = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(
            clf, X_train_t, y_train_enc,
            cv=5, scoring="f1_macro", n_jobs=-1,
        )
        cv_results[name] = scores.mean()
        print(f"  {name:25s}  CV macro-F1: {scores.mean():.4f}")

    best_clf_name = max(cv_results, key=cv_results.get)
    print(f"\n  ➜ Best classifier: {best_clf_name}")

    # ── 5. Tune the best classifier ───────────────────────────────────────
    print(f"\n[4/5] Tuning {best_clf_name} …")
    if best_clf_name == "RandomForest":
        param_dist = {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        base = RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1)
    elif best_clf_name == "XGBoost":
        param_dist = {
            "n_estimators": [200, 400, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        base = XGBClassifier(
            random_state=42, n_jobs=-1, verbosity=0,
            use_label_encoder=False, eval_metric="mlogloss")
    elif best_clf_name == "SVM":
        param_dist = {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.001],
        }
        base = SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42)
    else:
        param_dist = {"C": [0.01, 0.1, 1, 10, 100]}
        base = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42)

    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=20, cv=5,
        scoring="f1_macro",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train_t, y_train_enc)
    best_clf = search.best_estimator_
    print(f"  Best params: {search.best_params_}")

    # ── 6. Evaluate on test set ────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set …")
    y_pred = best_clf.predict(X_test_t)
    y_proba = best_clf.predict_proba(X_test_t)

    acc     = accuracy_score(y_test_enc, y_pred)
    prec    = precision_score(y_test_enc, y_pred, average="macro", zero_division=0)
    rec     = recall_score(y_test_enc, y_pred, average="macro", zero_division=0)
    f1      = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test_enc, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = 0.0

    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall   : {rec:.4f}")
    print(f"  Macro F1       : {f1:.4f}")
    print(f"  ROC-AUC (OvR)  : {roc_auc:.4f}")

    print("\n  Classification Report:")
    print(classification_report(
        y_test_enc, y_pred,
        target_names=le.classes_,
    ))

    # Confusion matrix plot
    cm = confusion_matrix(y_test_enc, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Deal Classifier")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save comparison results for all classifiers (retrain on full train)
    comparison = []
    for name, clf_model in classifiers.items():
        clf_model.fit(X_train_t, y_train_enc)
        yp = clf_model.predict(X_test_t)
        yprob = clf_model.predict_proba(X_test_t)
        try:
            rauc = roc_auc_score(y_test_enc, yprob, multi_class="ovr", average="macro")
        except ValueError:
            rauc = 0.0
        comparison.append({
            "model": name,
            "accuracy":  round(float(accuracy_score(y_test_enc, yp)), 4),
            "precision": round(float(precision_score(y_test_enc, yp, average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_test_enc, yp, average="macro", zero_division=0)), 4),
            "f1":        round(float(f1_score(y_test_enc, yp, average="macro", zero_division=0)), 4),
            "roc_auc":   round(float(rauc), 4),
        })

    comparison.append({
        "model": f"{best_clf_name} (tuned)",
        "accuracy":  round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec), 4),
        "f1":        round(float(f1), 4),
        "roc_auc":   round(float(roc_auc), 4),
    })

    metrics_out = {
        "best_model": best_clf_name,
        "test_accuracy":  round(float(acc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall":    round(float(rec), 4),
        "test_f1":        round(float(f1), 4),
        "test_roc_auc":   round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": le.classes_.tolist(),
        "comparison": comparison,
    }

    with open(MODELS_DIR / "classification_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Save classifier pipeline
    clf_full_pipeline = Pipeline([
        ("preprocessor", clf_preprocessor),
        ("classifier",   best_clf),
    ])
    joblib.dump(clf_full_pipeline, MODELS_DIR / "classifier.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

    print("\n✓ Classification complete. Model saved to models/classifier.joblib")
    return best_clf, metrics_out


if __name__ == "__main__":
    run_classification()
