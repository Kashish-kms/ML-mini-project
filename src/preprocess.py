"""
preprocess.py — Data loading, cleaning, feature engineering, and pipeline
construction for the Used Car Price Estimator.

Steps:
  1. Load raw CSV from data/raw/
  2. Extract brand from car name; compute car_age
  3. Parse mileage / engine / max_power (strip units → numeric)
  4. Impute missing values (median for numeric, mode for categorical)
  5. Remove outliers with the IQR method on selling_price and km_driven
  6. Build a scikit-learn Pipeline + ColumnTransformer (avoid data leakage)
  7. Apply log1p to the target (selling_price)
  8. 80/20 train/test split, save artefacts to data/processed/
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Paths (relative to project root car-price-app/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA     = PROJECT_ROOT / "data" / "raw" / "Car details v3.csv"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CURRENT_YEAR = 2024  # dataset was curated around 2020-2021; fix reference year


# ── helpers ────────────────────────────────────────────────────────────────
def _strip_unit(series, unit=None):
    """
    Convert strings containing numeric values and units into floats.

    Handles examples such as:
    - '17.3 kmpl'
    - '17.3 km/kg'
    - '1248 CC'
    - '74 bhp'
    - '23.4'
    - NaN
    """
    return pd.to_numeric(
        series.astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)", expand=False),
        errors="coerce"
    )


def load_and_clean(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the raw CSV and perform cleaning + feature engineering."""
    df = pd.read_csv(path)

    # 1. Extract brand (first word of name)
    df["brand"] = df["name"].str.split().str[0]

    # 2. Car age
    df["car_age"] = CURRENT_YEAR - df["year"]

    # 3. Parse units ----------------------------------------------------------
    df["mileage"]   = _strip_unit(df["mileage"],   "kmpl")
    # some entries have "km/kg" for CNG — handle that too
    df["mileage"]   = pd.to_numeric(df["mileage"], errors="coerce")

    df["engine"]    = _strip_unit(df["engine"],     "CC")
    df["engine"]    = pd.to_numeric(df["engine"],   errors="coerce")

    df["max_power"] = _strip_unit(df["max_power"],  "bhp")
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

    # 4. Drop columns we no longer need
    df.drop(columns=["name", "year", "torque"], inplace=True, errors="ignore")

    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical."""
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list, factor: float = 1.5) -> pd.DataFrame:
    """Remove rows outside [Q1 - factor*IQR, Q3 + factor*IQR] for given columns."""
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= df[col].between(q1 - factor * iqr, q3 + factor * iqr)
    before = len(df)
    df = df[mask].reset_index(drop=True)
    print(f"  Outlier removal: {before} → {len(df)} rows "
          f"({before - len(df)} removed)")
    return df


# ── Feature / target definitions ──────────────────────────────────────────
NUMERIC_FEATURES = [
    "car_age", "km_driven", "mileage", "engine", "max_power", "seats",
]
CATEGORICAL_FEATURES = [
    "brand", "fuel", "seller_type", "transmission", "owner",
]
TARGET = "selling_price"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    """Return a ColumnTransformer fitted ONLY on training data later."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer",  SimpleImputer(strategy="most_frequent")),
        ("encoder",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      NUMERIC_FEATURES),
            ("cat", categorical_pipeline,   CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


# ── Main entry point ──────────────────────────────────────────────────────
def run_preprocessing():
    """Execute the full preprocessing pipeline and persist artefacts."""
    print("=" * 60)
    print("STEP 1 — Preprocessing")
    print("=" * 60)

    # Load & clean
    print("\n[1/5] Loading raw data …")
    df = load_and_clean()
    print(f"  Shape after cleaning: {df.shape}")

    # Impute
    print("[2/5] Imputing missing values …")
    df = impute(df)

    # Remove outliers on price & km_driven
    print("[3/5] Removing outliers …")
    df = remove_outliers_iqr(df, ["selling_price", "km_driven"])

    # Train/test split
    print("[4/5] Splitting data (80/20) …")
    X = df[ALL_FEATURES]
    y = np.log1p(df[TARGET])  # log1p transform on target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # Fit preprocessor on train only
    print("[5/5] Fitting preprocessor on training set …")
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    # Save everything
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(
        {"X_train": X_train, "X_test": X_test,
         "y_train": y_train, "y_test": y_test},
        PROCESSED / "splits.joblib",
    )
    # Also save the full cleaned dataframe for EDA
    df.to_csv(PROCESSED / "cleaned.csv", index=False)

    # Save category mappings for the frontend dropdowns
    cat_options = {}
    for col in CATEGORICAL_FEATURES:
        cat_options[col] = sorted(df[col].unique().tolist())
    # Also save numeric ranges for validation
    num_ranges = {}
    for col in NUMERIC_FEATURES:
        num_ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
    metadata = {"categorical_options": cat_options, "numeric_ranges": num_ranges}
    with open(PROCESSED / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Preprocessing complete. Artefacts saved to data/processed/ and models/")
    return df, X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing()
