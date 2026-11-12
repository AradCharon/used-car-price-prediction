import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data/raw/bama_ml_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
REPORT_DIR = PROJECT_ROOT / "outputs/data_cleaning"

OUTPUT_FILE = OUTPUT_DIR / "bama_ml_dataset_clean.csv"

MAX_PRICE = 29_900_000_000
MAX_ENGINE_VOLUME = 8.0


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Column not found. Available columns: {list(df.columns)}"
    )


def clean_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    initial_rows = len(df)
    initial_columns = len(df.columns)

    print("Initial dataset shape:", df.shape)

    price_col = find_column(
        df, ["price", "Price", "price_toman"]
    )

    engine_col = find_column(
        df, ["engine_liter", "engine_volume",
             "engine", "engine_capacity"]
    )

    df[price_col] = clean_number(df[price_col])
    df[engine_col] = clean_number(df[engine_col])

    before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before - len(df)

    before = len(df)
    df = df[
        df[price_col].notna()
        & (df[price_col] > 0)
    ]
    invalid_prices_removed = before - len(df)

    before = len(df)
    df = df[df[price_col] <= MAX_PRICE]
    high_prices_removed = before - len(df)

    before = len(df)
    df = df[df[engine_col].notna()]
    missing_engine_removed = before - len(df)

    before = len(df)
    df = df[
        (df[engine_col] > 0)
        & (df[engine_col] <= MAX_ENGINE_VOLUME)
    ]
    invalid_engine_removed = before - len(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    summary = pd.DataFrame({
        "Metric": [
            "Initial Rows",
            "Final Rows",
            "Removed Rows",
            "Duplicate Rows Removed",
            "Invalid Prices Removed",
            "High Prices Removed",
            "Missing Engine Values Removed",
            "Invalid Engine Values Removed"
        ],
        "Value": [
            initial_rows,
            len(df),
            initial_rows - len(df),
            duplicates_removed,
            invalid_prices_removed,
            high_prices_removed,
            missing_engine_removed,
            invalid_engine_removed
        ]
    })

    summary.to_csv(
        REPORT_DIR / "cleaning_summary.csv",
        index=False
    )

    comparison = pd.DataFrame({
        "Stage": ["Before Cleaning", "After Cleaning"],
        "Rows": [initial_rows, len(df)],
        "Columns": [initial_columns, len(df.columns)]
    })

    comparison.to_csv(
        REPORT_DIR / "data_before_after_comparison.csv",
        index=False
    )

    missing_values = df.isnull().sum().reset_index()

    missing_values.columns = [
        "Column",
        "Missing Values"
    ]

    missing_values.to_csv(
        REPORT_DIR / "missing_values_after_cleaning.csv",
        index=False
    )

    print("\nCleaning completed successfully!")
    print("Initial rows:", initial_rows)
    print("Final rows:", len(df))
    print("Removed rows:", initial_rows - len(df))
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
