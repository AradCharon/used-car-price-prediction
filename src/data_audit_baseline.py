from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "cardekho_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET = "selling_price"
RANDOM_STATE = 42
TEST_SIZE = 0.20

REQUIRED_COLUMNS = [
    "brand", "model", "vehicle_age", "km_driven",
    "seller_type", "fuel_type", "transmission_type",
    "mileage", "engine", "max_power", "seats", TARGET,
]

NUMERIC_COLUMNS = [
    "vehicle_age", "km_driven", "mileage",
    "engine", "max_power", "seats", TARGET,
]


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def drop_unnamed_columns(df):
    unnamed = [
        c for c in df.columns
        if str(c).lower().startswith("unnamed:")
    ]
    if unnamed:
        df = df.drop(columns=unnamed)
        print(f"[audit] dropped {len(unnamed)} unnamed index column(s): {unnamed}")
    return df


def validate_required_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def missing_value_report(df):
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
        "dtype": df.dtypes.astype(str),
    })
    report.to_csv(OUTPUT_DIR / "missing_values.csv", encoding="utf-8-sig")
    return report


def duplicate_report(df):
    n = int(df.duplicated().sum())
    df[df.duplicated(keep=False)].to_csv(
        OUTPUT_DIR / "duplicate_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return n


def numeric_summary(df):
    summary = df.select_dtypes(include="number").describe().T
    summary.to_csv(
        OUTPUT_DIR / "numeric_summary.csv",
        encoding="utf-8-sig",
    )
    return summary


def iqr_outlier_report(df, columns):
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lower) | (s > upper)
        rows.append({
            "feature": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_percent": round(mask.mean() * 100, 2),
        })
    report = pd.DataFrame(rows)
    report.to_csv(
        OUTPUT_DIR / "iqr_outlier_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return report


def validity_report(df):
    rules = {
        "negative_vehicle_age": df["vehicle_age"] < 0,
        "negative_km_driven": df["km_driven"] < 0,
        "nonpositive_mileage": df["mileage"] <= 0,
        "nonpositive_engine": df["engine"] <= 0,
        "nonpositive_max_power": df["max_power"] <= 0,
        "seats_outside_2_10": ~df["seats"].between(2, 10),
        "nonpositive_selling_price": df[TARGET] <= 0,
    }
    report = pd.DataFrame([
        {
            "rule": name,
            "flagged_count": int(mask.sum()),
            "flagged_percent": round(mask.mean() * 100, 2),
        }
        for name, mask in rules.items()
    ])
    report.to_csv(
        OUTPUT_DIR / "validity_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return report


def prepare_modeling_frame(df):
    model_df = df[["brand", "model", "vehicle_age", TARGET]].copy()
    initial = len(model_df)

    model_df = model_df.dropna(
        subset=["brand", "model", "vehicle_age", TARGET]
    )
    model_df = model_df[
        (model_df[TARGET] > 0)
        & (model_df["vehicle_age"] >= 0)
        & (model_df["brand"].astype(str).str.strip() != "")
        & (model_df["model"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    print(f"[modeling] rows before filtering: {initial}")
    print(f"[modeling] rows after filtering : {len(model_df)}")
    print(f"[modeling] rows removed         : {initial - len(model_df)}")
    return model_df


def train_test_split_frame(model_df):
    train, test = train_test_split(
        model_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_baseline_lookup(train):
    lookup = {
        "brand_model_age": train.groupby(
            ["brand", "model", "vehicle_age"]
        )[TARGET].median(),
        "brand_model": train.groupby(
            ["brand", "model"]
        )[TARGET].median(),
        "brand": train.groupby("brand")[TARGET].median(),
        "global": train[TARGET].median(),
    }
    return lookup


def predict_baseline(row, lookup):
    key3 = (row["brand"], row["model"], row["vehicle_age"])
    key2 = (row["brand"], row["model"])
    key1 = row["brand"]

    if key3 in lookup["brand_model_age"].index:
        return lookup["brand_model_age"].loc[key3]
    if key2 in lookup["brand_model"].index:
        return lookup["brand_model"].loc[key2]
    if key1 in lookup["brand"].index:
        return lookup["brand"].loc[key1]
    return lookup["global"]


def evaluate_baseline(train, test):
    lookup = build_baseline_lookup(train)
    y_true = test[TARGET]
    y_pred = test.apply(lambda r: predict_baseline(r, lookup), axis=1)

    results = test[["brand", "model", "vehicle_age", TARGET]].copy()
    results["predicted_price"] = y_pred
    results["absolute_error"] = (results[TARGET] - y_pred).abs()
    results.to_csv(
        OUTPUT_DIR / "baseline_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    mae = mean_absolute_error(y_true, y_pred)
    negative_preds = int((y_pred < 0).sum())

    metrics = pd.DataFrame({
        "metric": [
            "dataset_rows",
            "modeling_rows",
            "training_rows",
            "test_rows",
            "random_state",
            "test_size",
            "training_global_median",
            "baseline_mae",
            "baseline_negative_predictions",
        ],
        "value": [
            None,
            len(train) + len(test),
            len(train),
            len(test),
            RANDOM_STATE,
            TEST_SIZE,
            lookup["global"],
            mae,
            negative_preds,
        ],
    })
    return results, metrics, mae, lookup["global"], negative_preds


def main():
    ensure_output_dir()
    print_section("LOAD AND VALIDATE")
    df = load_dataset(DATA_PATH)
    df = drop_unnamed_columns(df)
    validate_required_columns(df, REQUIRED_COLUMNS)
    df = coerce_numeric(df, NUMERIC_COLUMNS)

    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)

    print_section("MISSING VALUES")
    missing = missing_value_report(df)
    print(missing.to_string())

    print_section("DUPLICATES")
    dup_count = duplicate_report(df)
    print(f"Exact duplicate rows: {dup_count}")

    print_section("NUMERIC SUMMARY")
    print(numeric_summary(df).to_string())

    print_section("IQR OUTLIER REPORT")
    outliers = iqr_outlier_report(df, NUMERIC_COLUMNS)
    print(outliers.to_string(index=False))

    print_section("VALIDITY REPORT")
    validity = validity_report(df)
    print(validity.to_string(index=False))

    print_section("MODELING FRAME")
    model_df = prepare_modeling_frame(df)

    print_section("TRAIN / TEST SPLIT")
    train, test = train_test_split_frame(model_df)

    print_section("BASELINE EVALUATION")
    results, metrics, mae, global_median, negative_preds = evaluate_baseline(
        train, test
    )
    metrics.loc[metrics["metric"] == "dataset_rows", "value"] = len(df)
    metrics.to_csv(
        OUTPUT_DIR / "baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Training global median: {global_median:,.2f}")
    print(f"Baseline MAE: {mae:,.2f}")
    print(f"Negative baseline predictions: {negative_preds}")

    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

    print_section("OUTPUT DIRECTORY")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()