"""
app.py — Flask backend for the Used Car Price Estimator & Deal Classifier.

Routes:
  GET  /          → Serve the main page
  POST /predict   → Predict price + deal label
  GET  /insights  → Return metrics & chart data for dashboard
  GET  /metadata  → Return form dropdown options
"""

import os
import json
import logging
import traceback
import numpy as np
import pandas as pd
import joblib
try:
    import shap
except ImportError:
    shap = None
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ── Setup ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
STATIC_DIR   = Path(__file__).resolve().parent / "static"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(STATIC_DIR),
)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load models (once at startup) ─────────────────────────────────────────
try:
    reg_pipeline = joblib.load(MODELS_DIR / "regressor.joblib")
    clf_pipeline = joblib.load(MODELS_DIR / "classifier.joblib")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    with open(PROCESSED / "metadata.json") as f:
        metadata = json.load(f)
    with open(MODELS_DIR / "regression_metrics.json") as f:
        reg_metrics = json.load(f)
    with open(MODELS_DIR / "classification_metrics.json") as f:
        clf_metrics = json.load(f)

    # SHAP explainer for the regression model
    regressor = reg_pipeline.named_steps["regressor"]
    preprocessor = reg_pipeline.named_steps["preprocessor"]

    # Feature names
    from src.preprocess import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    except Exception:
        cat_names = []
    feature_names = NUMERIC_FEATURES + cat_names

    # SHAP explainer
    model_type = type(regressor).__name__
    if shap is not None and model_type in ("RandomForestRegressor", "XGBRegressor"):
        shap_explainer = shap.TreeExplainer(regressor)
    else:
        # Will create KernelExplainer on demand
        shap_explainer = None

    MODELS_LOADED = True
    logger.info("All models loaded successfully.")
except Exception as e:
    MODELS_LOADED = False
    logger.error(f"Failed to load models: {e}")
    logger.error(traceback.format_exc())


# ── Helpers ────────────────────────────────────────────────────────────────
REQUIRED_FIELDS = ["brand", "car_age", "km_driven", "fuel", "seller_type",
                   "transmission", "owner", "mileage", "engine",
                   "max_power", "seats"]

DEAL_COLORS = {
    "Underpriced": "green",
    "Fair":        "yellow",
    "Overpriced":  "red",
}


def _validate_input(data: dict) -> list:
    """Return a list of validation error messages, empty if valid."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            errors.append(f"Missing required field: {field}")

    # Numeric range checks
    try:
        if float(data.get("car_age", 0)) < 0:
            errors.append("car_age must be non-negative")
        if float(data.get("km_driven", 0)) < 0:
            errors.append("km_driven must be non-negative")
    except (TypeError, ValueError):
        errors.append("car_age and km_driven must be numeric")

    return errors


def _get_shap_factors(X_transformed, top_n=5):
    """Return top N SHAP feature contributions for a single prediction."""
    try:
        if shap_explainer is not None:
            sv = shap_explainer.shap_values(X_transformed)
        else:
            # Fallback: no SHAP
            return []

        if isinstance(sv, list):
            sv = sv[0]

        sv = sv.flatten()
        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:top_n]

        factors = []
        for i in top_idx:
            fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            factors.append({
                "feature": fname,
                "impact": round(float(sv[i]), 4),
                "abs_impact": round(float(abs_sv[i]), 4),
            })
        return factors
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return []


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/api/index")
@app.route("/api/index.py")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@app.route("/api/index/predict", methods=["POST"])
@app.route("/api/index.py/predict", methods=["POST"])
def predict():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded. Please train first."}), 503

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    # Validate
    errors = _validate_input(data)
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    try:
        # Build feature DataFrame for regression
        row = {
            "brand":        str(data["brand"]),
            "car_age":      float(data["car_age"]),
            "km_driven":    float(data["km_driven"]),
            "fuel":         str(data["fuel"]),
            "seller_type":  str(data["seller_type"]),
            "transmission": str(data["transmission"]),
            "owner":        str(data["owner"]),
            "mileage":      float(data["mileage"]),
            "engine":       float(data["engine"]),
            "max_power":    float(data["max_power"]),
            "seats":        float(data["seats"]),
        }
        X_reg = pd.DataFrame([row])

        # Predict price (log scale → original)
        pred_log = reg_pipeline.predict(X_reg)[0]
        pred_price = float(np.expm1(pred_log))

        # Price range using MAE from training
        mae = reg_metrics["test_MAE"]
        price_low  = max(0, pred_price - mae)
        price_high = pred_price + mae

        # SHAP top factors
        X_transformed = preprocessor.transform(X_reg)
        top_factors = _get_shap_factors(X_transformed)

        # Classification (if listed price provided)
        listed_price = data.get("listed_price")
        deal_label = None
        confidence = None

        if listed_price is not None and str(listed_price).strip() != "":
            listed_price = float(listed_price)
            row_clf = row.copy()
            row_clf["listed_price"] = listed_price
            X_clf = pd.DataFrame([row_clf])

            proba = clf_pipeline.predict_proba(X_clf)[0]
            pred_class = clf_pipeline.predict(X_clf)[0]
            deal_label = label_encoder.inverse_transform([pred_class])[0]
            confidence = round(float(proba.max()), 4)
        else:
            # If no listed price, use predicted price as reference
            listed_price = pred_price
            # Provide a simple rule-based label
            deal_label = "Fair"
            confidence = 1.0

        result = {
            "predicted_price": round(pred_price, 2),
            "predicted_price_lakh": round(pred_price / 100000, 2),
            "price_range": {
                "low":  round(price_low, 2),
                "high": round(price_high, 2),
                "low_lakh":  round(price_low / 100000, 2),
                "high_lakh": round(price_high / 100000, 2),
            },
            "deal_label": deal_label,
            "deal_color": DEAL_COLORS.get(deal_label, "gray"),
            "confidence": confidence,
            "top_factors": top_factors,
            "listed_price": round(float(listed_price), 2),
        }

        logger.info(f"Prediction: ₹{pred_price:,.0f} | Deal: {deal_label}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/insights")
@app.route("/api/index/insights")
@app.route("/api/index.py/insights")
def insights():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded."}), 503

    # Load SHAP global importance
    shap_path = MODELS_DIR / "shap_global_importance.json"
    shap_data = {}
    if shap_path.exists():
        with open(shap_path) as f:
            shap_data = json.load(f)

    return jsonify({
        "regression":     reg_metrics,
        "classification": clf_metrics,
        "shap_global":    shap_data,
    })


@app.route("/metadata")
@app.route("/api/index/metadata")
@app.route("/api/index.py/metadata")
def get_metadata():
    return jsonify(metadata)


@app.route("/static/img/<path:filename>")
def serve_img(filename):
    return send_from_directory(str(STATIC_DIR / "img"), filename)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
