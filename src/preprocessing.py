from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cardekho_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessed"

TARGET = "selling_price"
MIN_CATEGORY_COUNT = 10

NUMERIC_FEATURES = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "km_per_year",
    "power_per_engine",
]

CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "seller_type",
    "fuel_type",
    "transmission_type",
    "brand_tier",
]

DROP_COLUMNS = ["Unnamed: 0", "car_name"]

LUXURY_BRANDS = {
    "mercedes-benz", "bmw", "audi", "jaguar", "land rover",
    "porsche", "volvo", "lexus", "mini", "bentley",
    "ferrari", "maserati", "rolls-royce",
}

PREMIUM_BRANDS = {
    "toyota", "honda", "hyundai", "kia", "mg",
    "jeep", "skoda", "volkswagen", "isuzu",
}


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("Dataset is empty.")
    print(f"[load] raw shape: {df.shape}")
    return df


def clean_data(df):
    df = df.copy()

    existing = [c for c in DROP_COLUMNS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"[clean] dropped columns: {existing}")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    if "model" in df.columns:
        df["model"] = df["model"].str.replace(
            r"redi-go", "redigo", regex=True
        )
        df["model"] = df["model"].str.replace(
            r"^dzire (vxi|lxi|zxi)$", "dzire", regex=True
        )
        df["model"] = df["model"].str.replace(
            r"^swift dzire$", "dzire", regex=True
        )

    numeric_cols = [
        "vehicle_age", "km_driven", "mileage",
        "engine", "max_power", "seats", TARGET,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"[clean] dropped {before - len(df)} rows with missing target")

    before = len(df)
    df = df[
        (df[TARGET] > 0)
        & (df["vehicle_age"] >= 0)
        & (df["km_driven"] >= 0)
        & (df["mileage"] > 0)
        & (df["engine"] > 0)
        & (df["max_power"] > 0)
        & (df["seats"].between(2, 10))
    ].copy()
    print(f"[clean] dropped {before - len(df)} rows failing validity rules")

    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] dropped {before - len(df)} exact duplicates")

    df = df.reset_index(drop=True)
    print(f"[clean] final shape: {df.shape}")
    return df


def flag_outliers(df):
    df = df.copy()
    for col in ["km_driven", "selling_price"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f"{col}_is_outlier"] = (
            (df[col] < lower) | (df[col] > upper)
        ).astype(int)

    flags = {
        c: int(df[c].sum())
        for c in df.columns if c.endswith("_is_outlier")
    }
    print(f"[outliers] flagged: {flags}")
    return df


def group_rare_categories(df, min_count=MIN_CATEGORY_COUNT):
    df = df.copy()
    for col in ["brand", "model"]:
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        rare = counts[counts < min_count].index
        if len(rare) > 0:
            df[col] = df[col].where(~df[col].isin(rare), "__rare__")
            print(f"[rare] {col}: grouped {len(rare)} rare values")
    return df


def brand_tier(brand):
    if brand in LUXURY_BRANDS:
        return "luxury"
    if brand in PREMIUM_BRANDS:
        return "premium"
    return "economy"


def engineer_features(df):
    df = df.copy()

    df["km_per_year"] = np.where(
        df["vehicle_age"] > 0,
        df["km_driven"] / df["vehicle_age"],
        df["km_driven"],
    )

    df["power_per_engine"] = (
        df["max_power"] / df["engine"].replace(0, np.nan)
    )

    df["brand_tier"] = df["brand"].map(brand_tier)

    return df


def build_preprocessor(
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
):
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_frame(df):
    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after engineering: {missing}")
    return df[keep_cols].copy()


def save_outputs(df_clean, df_model):
    ensure_dirs()
    df_clean.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False)
    df_model.to_csv(OUTPUT_DIR / "modeling_dataset.csv", index=False)
    print(f"[save] cleaned_dataset.csv and modeling_dataset.csv written")
    print(f"[save] output dir: {OUTPUT_DIR.resolve()}")


def main():
    df = load_raw_data()
    df = clean_data(df)
    df = flag_outliers(df)
    df = group_rare_categories(df)
    df = engineer_features(df)

    df_model = build_model_frame(df)
    save_outputs(df, df_model)

    print(f"\n[info] cleaned columns  : {df.shape[1]}")
    print(f"[info] modeling columns : {df_model.shape[1]}")
    print(f"[info] modeling rows    : {len(df_model)}")
    print("[DONE] Preprocessing finished (no split performed).")


if __name__ == "__main__":
    main()