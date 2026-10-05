import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_dataset_exists():
    """
    Verify that the expected CarDekho dataset exists.
    """
    dataset_path = PROJECT_ROOT / "data" / "raw" / "Car details v3.csv"

    assert dataset_path.exists(), (
        f"Dataset not found at: {dataset_path}"
    )


def test_dataset_has_required_columns():
    """
    Verify that the CarDekho dataset contains the required columns.
    """
    dataset_path = PROJECT_ROOT / "data" / "raw" / "Car details v3.csv"

    if not dataset_path.exists():
        pytest.skip("CarDekho dataset is not available.")

    df = pd.read_csv(dataset_path)

    required_columns = {
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
    }

    missing_columns = required_columns - set(df.columns)

    assert not missing_columns, (
        f"Missing required columns: {missing_columns}"
    )


def test_brand_extraction_logic():
    """
    Verify the expected brand extraction behavior.

    Example:
        'Maruti Swift Dzire VDI'
        -> 'Maruti'
    """
    names = pd.Series([
        "Maruti Swift Dzire VDI",
        "Hyundai i20 Sportz",
        "Honda City V MT",
        "Toyota Innova Crysta",
    ])

    brands = names.str.strip().str.split().str[0]

    assert brands.tolist() == [
        "Maruti",
        "Hyundai",
        "Honda",
        "Toyota",
    ]


def test_car_age_calculation():
    """
    Verify car_age = current_year - year.
    """
    current_year = 2026

    years = pd.Series([2026, 2020, 2015, 2010])

    car_age = current_year - years

    assert car_age.tolist() == [0, 6, 11, 16]


def test_numeric_unit_conversion():
    """
    Verify conversion of common CarDekho string values to numeric values.
    """

    mileage = pd.Series([
        "18.9 kmpl",
        "21.4 km/kg",
        "15 kmpl",
        np.nan,
    ])

    converted_mileage = pd.to_numeric(
        mileage.astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce",
    )

    assert converted_mileage.iloc[0] == pytest.approx(18.9)
    assert converted_mileage.iloc[1] == pytest.approx(21.4)
    assert converted_mileage.iloc[2] == pytest.approx(15.0)
    assert pd.isna(converted_mileage.iloc[3])


def test_engine_conversion():
    """
    Verify engine values such as '1248 CC' become numeric.
    """

    engine = pd.Series([
        "1248 CC",
        "1197 CC",
        "998 CC",
        np.nan,
    ])

    converted_engine = pd.to_numeric(
        engine.astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce",
    )

    assert converted_engine.iloc[0] == pytest.approx(1248)
    assert converted_engine.iloc[1] == pytest.approx(1197)
    assert converted_engine.iloc[2] == pytest.approx(998)
    assert pd.isna(converted_engine.iloc[3])


def test_max_power_conversion():
    """
    Verify max_power values such as '74 bhp' become numeric.
    """

    power = pd.Series([
        "74 bhp",
        "88.5 bhp",
        "98.6 bhp",
        np.nan,
    ])

    converted_power = pd.to_numeric(
        power.astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce",
    )

    assert converted_power.iloc[0] == pytest.approx(74)
    assert converted_power.iloc[1] == pytest.approx(88.5)
    assert converted_power.iloc[2] == pytest.approx(98.6)
    assert pd.isna(converted_power.iloc[3])


def test_missing_numeric_values_can_be_imputed():
    """
    Verify median imputation behavior for numeric columns.
    """

    values = pd.Series([10.0, 20.0, np.nan, 30.0])

    median_value = values.median()
    filled = values.fillna(median_value)

    assert not filled.isna().any()
    assert filled.iloc[2] == 20.0


def test_missing_categorical_values_can_be_imputed():
    """
    Verify mode imputation behavior for categorical columns.
    """

    values = pd.Series([
        "Petrol",
        "Diesel",
        "Diesel",
        np.nan,
    ])

    mode_value = values.mode()[0]
    filled = values.fillna(mode_value)

    assert not filled.isna().any()
    assert filled.iloc[3] == "Diesel"


def test_iqr_outlier_detection():
    """
    Verify that IQR-based filtering can identify extreme values.
    """

    values = pd.Series([
        100,
        110,
        105,
        115,
        108,
        112,
        10000,
    ])

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered = values[
        (values >= lower_bound) &
        (values <= upper_bound)
    ]

    assert 10000 not in filtered.values
    assert len(filtered) < len(values)


def test_log1p_target_transformation():
    """
    Verify that log1p transformation is reversible using expm1.
    """

    prices = np.array([
        100000,
        250000,
        500000,
        1000000,
    ])

    log_prices = np.log1p(prices)
    recovered_prices = np.expm1(log_prices)

    np.testing.assert_allclose(
        prices,
        recovered_prices,
        rtol=1e-10,
    )
