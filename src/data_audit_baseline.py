
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("cardekho_dataset.csv")
OUTPUT_DIR = Path("outputs")
TARGET = "selling_price"
RANDOM_STATE = 42
TEST_SIZE = 0.20

REQUIRED_COLUMNS = [
    "brand", "model", "vehicle_age", "km_driven",
    "seller_type", "fuel_type", "transmission_type",
    "mileage", "engine", "max_power", "seats", TARGET
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    unnamed = [
        col for col in df.columns
        if str(col).lower().startswith("unnamed:")
    ]
    df = df.drop(columns=unnamed)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    missing_report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique_values": df.nunique(dropna=True),
        "dtype": df.dtypes.astype(str)
    })

    missing_report.to_csv(
        OUTPUT_DIR / "missing_values.csv",
        encoding="utf-8-sig"
    )

    print("\nMissing values:")
    print(missing_report.to_string())

    duplicate_count = int(df.duplicated().sum())
    print("\nExact duplicate rows:", duplicate_count)

    numeric_columns = [
        "vehicle_age", "km_driven", "mileage",
        "engine", "max_power", "seats", TARGET
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_df = df.select_dtypes(include="number")
    numeric_summary = numeric_df.describe().T

    numeric_summary.to_csv(
        OUTPUT_DIR / "numeric_summary.csv",
        encoding="utf-8-sig"
    )

    print("\nNumeric summary:")
    print(numeric_summary.to_string())

    if TARGET in df.columns:
        price = df[TARGET].dropna()

        if not price.empty:
            q1 = price.quantile(0.25)
            q3 = price.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((price < lower) | (price > upper)).sum()

            print("\nPrice outlier screening:")
            print(f"Q1: {q1:,.2f}")
            print(f"Q3: {q3:,.2f}")
            print(f"Lower bound: {lower:,.2f}")
            print(f"Upper bound: {upper:,.2f}")
            print(f"Flagged rows: {outliers:,}")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    model_df = df[
        ["brand", "model", "vehicle_age", TARGET]
    ].copy()

    initial_rows = len(model_df)

    model_df = model_df.dropna(
        subset=["brand", "model", "vehicle_age", TARGET]
    )

    model_df = model_df[
        (model_df[TARGET] > 0)
        & (model_df["vehicle_age"] >= 0)
        & (model_df["brand"].astype(str).str.strip() != "")
        & (model_df["model"].astype(str).str.strip() != "")
    ].copy()

    print("\n" + "=" * 60)
    print("MODELING DATA")
    print("=" * 60)
    print("Rows before filtering:", initial_rows)
    print("Rows after filtering:", len(model_df))
    print("Rows removed:", initial_rows - len(model_df))

    if len(model_df) < 5:
        raise ValueError("Too few valid rows for evaluation.")

    train, test = train_test_split(
        model_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    median_brand_model_age = train.groupby(
        ["brand", "model", "vehicle_age"]
    )[TARGET].median()

    median_brand_model = train.groupby(
        ["brand", "model"]
    )[TARGET].median()

    median_brand = train.groupby("brand")[TARGET].median()
    global_median = train[TARGET].median()

    def predict_baseline(row):
        key3 = (
            row["brand"],
            row["model"],
            row["vehicle_age"]
        )
        key2 = (row["brand"], row["model"])
        key1 = row["brand"]

        if key3 in median_brand_model_age.index:
            return median_brand_model_age.loc[key3]

        if key2 in median_brand_model.index:
            return median_brand_model.loc[key2]

        if key1 in median_brand.index:
            return median_brand.loc[key1]

        return global_median

    y_true = test[TARGET]
    y_pred = test.apply(predict_baseline, axis=1)
    mae = mean_absolute_error(y_true, y_pred)

    results = test[
        ["brand", "model", "vehicle_age", TARGET]
    ].copy()

    results["predicted_price"] = y_pred
    results["absolute_error"] = (
        results[TARGET] - results["predicted_price"]
    ).abs()

    results.to_csv(
        OUTPUT_DIR / "baseline_test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    metrics = pd.DataFrame({
        "metric": [
            "dataset_rows",
            "modeling_rows",
            "training_rows",
            "test_rows",
            "random_state",
            "test_size",
            "training_global_median",
            "baseline_MAE"
        ],
        "value": [
            len(df),
            len(model_df),
            len(train),
            len(test),
            RANDOM_STATE,
            TEST_SIZE,
            global_median,
            mae
        ]
    })

    metrics.to_csv(
        OUTPUT_DIR / "baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Training global median: {global_median:,.2f}")
    print(f"Baseline MAE: {mae:,.2f}")

    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))

    print("\nOutput directory:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
