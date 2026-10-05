"""
test_app.py — pytest tests for preprocessing functions and /predict endpoint.

Covers:
  - Unit stripping
  - Imputation
  - Outlier removal
  - /predict with valid input
  - /predict with missing optional listed_price
  - /predict with very old car
  - /predict with unseen brand
  - /predict with missing required fields
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import _strip_unit, impute, remove_outliers_iqr


# ═══════════════════════════════════════════════════════════
# Unit Tests — Preprocessing Helpers
# ═══════════════════════════════════════════════════════════

class TestStripUnit:
    def test_strip_kmpl(self):
        s = pd.Series(["21.1 kmpl", "18.0 kmpl", "25.5 kmpl"])
        result = _strip_unit(s, "kmpl")
        assert list(result) == [21.1, 18.0, 25.5]

    def test_strip_cc(self):
        s = pd.Series(["1197 CC", "2000 CC"])
        result = _strip_unit(s, "CC")
        assert list(result) == [1197.0, 2000.0]

    def test_strip_bhp(self):
        s = pd.Series(["82 bhp", "110.5 bhp"])
        result = _strip_unit(s, "bhp")
        assert list(result) == [82.0, 110.5]

    def test_strip_empty_to_nan(self):
        s = pd.Series(["bhp", "82 bhp"])  # first has only unit text, becomes ""
        result = _strip_unit(s, "bhp")
        assert np.isnan(result.iloc[0])
        assert result.iloc[1] == 82.0


class TestImpute:
    def test_numeric_median(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0]})
        result = impute(df)
        assert result["a"].iloc[1] == 3.0  # median of [1, 3, 5]

    def test_categorical_mode(self):
        df = pd.DataFrame({"b": ["x", "y", "x", None]})
        result = impute(df)
        assert result["b"].iloc[3] == "x"  # mode is "x"


class TestOutlierRemoval:
    def test_removes_outliers(self):
        np.random.seed(42)
        normal = np.random.normal(50, 5, 100).tolist()
        outliers = [500, -200, 1000]
        df = pd.DataFrame({"val": normal + outliers})
        result = remove_outliers_iqr(df, ["val"])
        assert len(result) < len(df)
        assert result["val"].max() < 500

    def test_no_removal_needed(self):
        df = pd.DataFrame({"val": [10, 11, 12, 13, 14]})
        result = remove_outliers_iqr(df, ["val"])
        assert len(result) == 5


# ═══════════════════════════════════════════════════════════
# Integration Tests — /predict endpoint
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def client():
    """Create a Flask test client."""
    from app.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


VALID_PAYLOAD = {
    "brand": "Maruti",
    "car_age": 5,
    "km_driven": 50000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "engine": 1197,
    "max_power": 82,
    "mileage": 21.1,
    "seats": 5,
    "listed_price": 450000,
}


class TestPredictEndpoint:
    def test_valid_prediction(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_price" in data
        assert "deal_label" in data
        assert data["predicted_price"] > 0
        assert data["deal_label"] in ["Underpriced", "Fair", "Overpriced"]

    def test_missing_optional_listed_price(self, client):
        """Listed price is optional — should still return a prediction."""
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "listed_price"}
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_price" in data
        assert data["deal_label"] == "Fair"  # default when no listed price

    def test_very_old_car(self, client):
        """A 30-year-old car should still get a valid prediction."""
        payload = VALID_PAYLOAD.copy()
        payload["car_age"] = 30
        payload["km_driven"] = 200000
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["predicted_price"] > 0

    def test_unseen_brand(self, client):
        """Unseen brand should be handled by OneHotEncoder(handle_unknown='ignore')."""
        payload = VALID_PAYLOAD.copy()
        payload["brand"] = "UnknownBrandXYZ"
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_price" in data

    def test_missing_required_field(self, client):
        """Missing required field should return 400."""
        payload = VALID_PAYLOAD.copy()
        del payload["brand"]
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data
