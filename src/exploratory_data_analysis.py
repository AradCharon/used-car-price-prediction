from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "cardekho_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

for d in (TABLE_DIR, FIGURE_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "selling_price"
TOP_N_CATEGORIES = 15

NUMERICAL_COLUMNS = [
    "vehicle_age", "km_driven", "mileage",
    "engine", "max_power", "seats", TARGET,
]

RECORD_COLUMNS = [
    "car_name", "brand", "model", "vehicle_age",
    "km_driven", "mileage", "engine", "max_power",
    "seats", TARGET,
]

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("The dataset is empty.")
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' does not exist.")
    unnamed = [
        c for c in df.columns
        if str(c).lower().startswith("unnamed:")
    ]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df.reset_index(drop=True)


def dataset_overview(df):
    print_section("DATASET OVERVIEW")
    print(f"Observations: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")
    print("\nColumn names:", df.columns.tolist())
    print("\nData information:")
    df.info()

    df.describe(include="all").T.to_csv(TABLE_DIR / "dataset_summary.csv")

    unique_summary = pd.DataFrame({
        "unique_values": df.nunique(),
        "data_type": df.dtypes.astype(str),
    }).sort_values("unique_values", ascending=False)

    unique_summary.to_csv(TABLE_DIR / "unique_values_summary.csv")
    print("\nUnique values summary:")
    print(unique_summary.to_string())


def data_quality_assessment(df):
    print_section("DATA QUALITY ASSESSMENT")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    }).sort_values("missing_percent", ascending=False)
    missing.index.name = "feature"
    missing.to_csv(TABLE_DIR / "missing_values.csv")
    print(missing.to_string())

    dup_mask = df.duplicated(keep=False)
    dup_count = int(df.duplicated().sum())
    print(f"\nDuplicate count: {dup_count:,}")
    print(f"Duplicate percent: {dup_count / len(df) * 100:.2f}%")
    df[dup_mask].to_csv(TABLE_DIR / "duplicate_records.csv", index=False)


def data_validity_analysis(df):
    print_section("DATA VALIDITY ANALYSIS")
    rules = {
        "negative_vehicle_age": df["vehicle_age"] < 0,
        "negative_km_driven": df["km_driven"] < 0,
        "nonpositive_mileage": df["mileage"] <= 0,
        "nonpositive_engine": df["engine"] <= 0,
        "nonpositive_max_power": df["max_power"] <= 0,
        "seats_outside_2_10": ~df["seats"].between(2, 10),
        "nonpositive_selling_price": df[TARGET] <= 0,
    }
    summary = pd.DataFrame([
        {
            "rule": name,
            "flagged_count": int(mask.sum()),
            "flagged_percent": round(mask.mean() * 100, 2),
        }
        for name, mask in rules.items()
    ])
    summary.to_csv(TABLE_DIR / "data_validity_summary.csv", index=False)
    print(summary.to_string(index=False))

    flagged_mask = pd.Series(False, index=df.index)
    for mask in rules.values():
        flagged_mask |= mask
    flagged = df.loc[flagged_mask].copy()
    flagged.to_csv(TABLE_DIR / "potentially_invalid_records.csv", index=False)
    print(f"\nTotal flagged records: {len(flagged):,}")
    if not flagged.empty:
        cols = [c for c in RECORD_COLUMNS if c in flagged.columns]
        print(flagged[cols].head(20).to_string(index=False))


def iqr_outlier_report(df):
    print_section("IQR OUTLIER REPORT")
    rows = []
    for col in NUMERICAL_COLUMNS:
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
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_percent": round(mask.mean() * 100, 2),
        })
    report = pd.DataFrame(rows)
    report.to_csv(TABLE_DIR / "numerical_outlier_summary.csv", index=False)
    print(report.to_string(index=False))
    return report


def target_analysis(df):
    print_section("TARGET VARIABLE ANALYSIS")
    target = df[TARGET].dropna()
    if target.empty:
        raise ValueError("Target variable contains no valid values.")

    stats = target.describe().to_frame().T
    stats["median"] = target.median()
    stats["skewness"] = target.skew()
    stats["kurtosis"] = target.kurtosis()
    stats["zero_count"] = int((target == 0).sum())
    stats["negative_count"] = int((target < 0).sum())
    stats.to_csv(TABLE_DIR / "target_statistics.csv", index=False)
    print(stats.to_string(index=False))

    positive = target[target > 0]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    sns.histplot(target, bins=50, kde=True, ax=axes[0])
    axes[0].set_title("Selling Price Distribution")
    sns.boxplot(x=target, ax=axes[1])
    axes[1].set_title("Selling Price Boxplot")
    sns.histplot(np.log1p(positive), bins=50, kde=True, ax=axes[2])
    axes[2].set_title("Log-Transformed Selling Price")
    axes[2].set_xlabel("Log(1 + Selling Price)")
    save_figure("target_distribution.png")

    q1, q3 = target.quantile(0.25), target.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df[TARGET] < lower) | (df[TARGET] > upper)]
    pd.DataFrame({
        "q1": [q1], "q3": [q3], "iqr": [iqr],
        "lower_bound": [lower], "upper_bound": [upper],
        "outlier_count": [len(outliers)],
        "outlier_percent": [len(outliers) / len(df) * 100],
    }).to_csv(TABLE_DIR / "target_outlier_summary.csv", index=False)
    outliers.to_csv(TABLE_DIR / "potential_price_outliers.csv", index=False)


def numerical_feature_analysis(df):
    print_section("NUMERICAL FEATURE ANALYSIS")
    cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != TARGET
    ]
    if not cols:
        print("No numerical features found.")
        return

    summary = df[cols].describe().T
    summary["skewness"] = df[cols].skew()
    summary["kurtosis"] = df[cols].kurtosis()
    summary.to_csv(TABLE_DIR / "numerical_features_summary.csv")
    print(summary.to_string())

    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=axes[0])
        axes[0].set_title(f"{col} Distribution")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"{col} Boxplot")
        save_figure(f"numerical_{col}.png")


def categorical_feature_analysis(df):
    print_section("CATEGORICAL FEATURE ANALYSIS")
    cols = df.select_dtypes(include=["object", "str", "category"]).columns
    if len(cols) == 0:
        print("No categorical features found.")
        return

    summary = []
    for col in cols:
        vc = df[col].value_counts(dropna=False)
        mode_vals = df[col].mode()
        summary.append({
            "feature": col,
            "unique_values": df[col].nunique(),
            "most_frequent": mode_vals.iloc[0] if not mode_vals.empty else np.nan,
            "most_frequent_count": int(vc.iloc[0]) if not vc.empty else 0,
        })
        vc.rename_axis(col).reset_index(name="count").to_csv(
            TABLE_DIR / f"category_counts_{col}.csv", index=False
        )
        if col in ("car_name", "model"):
            continue
        top = vc.head(TOP_N_CATEGORIES)
        plt.figure(figsize=(11, 6))
        sns.barplot(x=top.values, y=top.index, orient="h")
        plt.title(f"Top Categories: {col}")
        plt.xlabel("Number of Vehicles")
        save_figure(f"categorical_{col}.png")

    pd.DataFrame(summary).to_csv(
        TABLE_DIR / "categorical_features_summary.csv", index=False
    )
    print(pd.DataFrame(summary).to_string(index=False))


def price_relationship_analysis(df):
    print_section("FEATURES VS SELLING PRICE")
    num_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != TARGET
    ]
    for col in num_cols:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df, x=col, y=TARGET, alpha=0.4, s=25)
        plt.title(f"{col} vs Selling Price")
        save_figure(f"price_vs_{col}.png")

    cat_cols = [
        c for c in df.select_dtypes(include=["object", "str", "category"]).columns
        if c in ("brand", "fuel_type", "seller_type", "transmission_type")
    ]
    for col in cat_cols:
        top = df[col].value_counts().head(TOP_N_CATEGORIES).index
        plot_data = df[df[col].isin(top)].copy()
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_data, x=col, y=TARGET, order=top)
        plt.title(f"Selling Price by {col}")
        plt.xticks(rotation=45, ha="right")
        save_figure(f"price_by_{col}.png")


def correlation_analysis(df):
    print_section("CORRELATION ANALYSIS")
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        print("Not enough numerical columns.")
        return
    corr = num_df.corr(method="pearson")
    corr.to_csv(TABLE_DIR / "correlation_matrix.csv")

    target_corr = (
        corr[TARGET].drop(TARGET).sort_values(ascending=False, key=abs)
    )
    target_corr.to_csv(TABLE_DIR / "target_correlations.csv", header=["correlation"])
    print(target_corr.to_string())

    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5)
    plt.title("Numerical Feature Correlation Matrix")
    save_figure("correlation_matrix.png")


def investigate_extreme_records(df):
    print_section("EXTREME RECORD INVESTIGATION")
    cols_to_check = [
        "vehicle_age", "km_driven", "mileage",
        "engine", "max_power", TARGET,
    ]
    cols = [c for c in RECORD_COLUMNS if c in df.columns]
    for col in cols_to_check:
        if col not in df.columns:
            continue
        top = df.nlargest(10, col)[cols]
        print(f"\nTop 10 records by {col}:")
        print(top.to_string(index=False))
        top.to_csv(TABLE_DIR / f"top_10_{col}.csv", index=False)

    print("\nSeat distribution:")
    seats = df["seats"].value_counts(dropna=False).sort_index()
    print(seats.to_string())
    seats.to_csv(TABLE_DIR / "seat_distribution.csv", header=["count"])

    zero_seats = df[df["seats"] == 0]
    print(f"\nRecords with zero seats: {len(zero_seats)}")
    zero_seats.to_csv(TABLE_DIR / "zero_seat_records.csv", index=False)


def generate_report(df):
    print_section("EDA EXECUTION SUMMARY")
    report = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "total_missing": int(df.isna().sum().sum()),
        "exact_duplicates": int(df.duplicated().sum()),
        "target_mean": float(df[TARGET].mean()),
        "target_median": float(df[TARGET].median()),
        "target_std": float(df[TARGET].std()),
        "target_skewness": float(df[TARGET].skew()),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
    }
    lines = ["EXPLORATORY DATA ANALYSIS REPORT", "=" * 45]
    lines += [f"{k}: {v}" for k, v in report.items()]
    lines += ["", "No records were removed or modified during EDA.",
              "IQR outliers are flagged for investigation only."]
    text = "\n".join(lines)
    (REPORT_DIR / "eda_summary.txt").write_text(text, encoding="utf-8")
    pd.Series(report).to_csv(REPORT_DIR / "eda_summary.csv")
    print(text)


def main():
    print_section("USED CAR PRICE PREDICTION - EDA")
    df = load_data()
    dataset_overview(df)
    data_quality_assessment(df)
    data_validity_analysis(df)
    iqr_outlier_report(df)
    investigate_extreme_records(df)
    target_analysis(df)
    numerical_feature_analysis(df)
    categorical_feature_analysis(df)
    price_relationship_analysis(df)
    correlation_analysis(df)
    generate_report(df)
    print_section("EDA COMPLETED")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()