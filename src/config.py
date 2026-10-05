from pathlib import Path

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

APP_DIR = PROJECT_ROOT / "app"
STATIC_DIR = APP_DIR / "static"
IMG_DIR = STATIC_DIR / "img"

# ============================================================
# Dataset
# ============================================================

DATASET_NAME = "Car details v3.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_NAME

TARGET_COLUMN = "selling_price"

# ============================================================
# Machine learning configuration
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Current year used for car_age calculation.
# Keeping this configurable makes the preprocessing easier to update.
CURRENT_YEAR = 2026

# ============================================================
# Original dataset columns
# ============================================================

EXPECTED_COLUMNS = [
    "name",
    "year",
    "selling_price",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

# ============================================================
# Feature configuration
# ============================================================

NUMERIC_FEATURES = [
    "car_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

CATEGORICAL_FEATURES = [
    "brand",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

# ============================================================
# Classification thresholds
# ============================================================

UNDERPRICED_THRESHOLD = 0.90
OVERPRICED_THRESHOLD = 1.10

DEAL_LABELS = [
    "Underpriced",
    "Fair",
    "Overpriced",
]

# ============================================================
# Model output files
# ============================================================

REGRESSOR_PATH = MODELS_DIR / "regressor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"

REGRESSION_METRICS_PATH = MODELS_DIR / "regression_metrics.json"
CLASSIFICATION_METRICS_PATH = MODELS_DIR / "classification_metrics.json"

# ============================================================
# EDA output files
# ============================================================

EDA_CHARTS = {
    "price_distribution": IMG_DIR / "price_distribution.png",
    "log_price_distribution": IMG_DIR / "log_price_distribution.png",
    "price_vs_age": IMG_DIR / "price_vs_age.png",
    "price_vs_km": IMG_DIR / "price_vs_km.png",
    "avg_price_by_brand": IMG_DIR / "avg_price_by_brand.png",
    "avg_price_by_fuel": IMG_DIR / "avg_price_by_fuel.png",
    "avg_price_by_transmission": IMG_DIR / "avg_price_by_transmission.png",
    "correlation_heatmap": IMG_DIR / "correlation_heatmap.png",
}

# ============================================================
# Application configuration
# ============================================================

APP_HOST = "0.0.0.0"
APP_PORT = 5000

MAX_KM_DRIVEN = 2_000_000
MIN_YEAR = 1980
MAX_YEAR = CURRENT_YEAR + 1

MIN_SEATS = 1
MAX_SEATS = 20
