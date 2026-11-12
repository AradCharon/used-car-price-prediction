# -*- coding: utf-8 -*-
"""
Divar Cars - Data Inspection & Audit
Usage:
    pip install pandas numpy reportlab
    python divar_data_audit.py archive.zip
or:
    python divar_data_audit.py final2024.csv

The script creates 13 requested outputs in ./data_audit_outputs/
"""
import sys
import re
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path("data_audit_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    """Load the first CSV found in a ZIP, or load a CSV directly."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV file was found inside the ZIP archive.")
            chosen = csv_files[0]
            with zf.open(chosen) as f:
                try:
                    data = pd.read_csv(f, low_memory=False)
                except UnicodeDecodeError:
                    f.seek(0)
                    data = pd.read_csv(f, encoding="latin1", low_memory=False)
            return data, chosen
    elif path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, low_memory=False), path.name
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1", low_memory=False), path.name
    else:
        raise ValueError("Input must be a .zip or .csv file.")

DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789"
)

def parse_numeric(series: pd.Series) -> pd.Series:
    """Convert Persian/Arabic digits and common formatted numbers to numeric."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        text = str(value).translate(DIGIT_TRANSLATION)
        text = text.replace("٬", "").replace(",", "").replace("٫", ".").strip()
        text = re.sub(r"[^0-9.\\-]", "", text)
        try:
            return float(text) if text else np.nan
        except ValueError:
            return np.nan
    return series.map(convert)

def save_csv(data: pd.DataFrame, filename: str):
    data.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")

def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("archive.zip")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Put archive.zip beside this script "
            "or pass its path as an argument."
        )

    df, source_file = load_dataset(input_path)
    rows, columns = df.shape
    column_names = list(df.columns)
    duplicate_count = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    total_cells = rows * columns

    # Parse selected numeric-looking fields without changing the original data.
    parsed = {}
    for col in ["Price", "Mileage", "Model_Year", "Third_party_insurance"]:
        if col in df.columns:
            parsed[col] = parse_numeric(df[col])

    # 1. Dataset overview
    overview = pd.DataFrame([{
        "dataset_file": source_file,
        "rows": rows,
        "columns": columns,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
        "exact_duplicate_rows": duplicate_count,
        "total_missing_cells": missing_cells,
        "missing_cell_percent": round(missing_cells / total_cells * 100, 4) if total_cells else 0
    }])
    save_csv(overview, "dataset_overview.csv")

    # 2. Column schema, nulls, unique values and sample values
    column_rows = []
    for col in column_names:
        series = df[col]
        counts = series.value_counts(dropna=True)
        column_rows.append({
            "column": col,
            "data_type": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(series.isna().mean() * 100, 4),
            "unique_count": int(series.nunique(dropna=True)),
            "most_frequent_value": str(counts.index[0]) if len(counts) else "",
            "most_frequent_count": int(counts.iloc[0]) if len(counts) else 0,
            "sample_values": " | ".join(map(str, series.dropna().head(5).tolist()))
        })
    column_summary = pd.DataFrame(column_rows)
    save_csv(column_summary, "column_summary.csv")

    # 3. Missing values
    missing_report = pd.DataFrame({
        "column": column_names,
        "missing_count": [int(df[c].isna().sum()) for c in column_names],
        "missing_percent": [round(df[c].isna().mean() * 100, 4) for c in column_names],
        "non_missing_count": [int(df[c].notna().sum()) for c in column_names]
    })
    save_csv(missing_report, "missing_values_report.csv")

    # 4. Duplicate records
    duplicate_report = pd.DataFrame([{
        "exact_duplicate_rows": duplicate_count,
        "duplicate_percent": round(duplicate_count / rows * 100, 4) if rows else 0,
        "rows_after_exact_deduplication": rows - duplicate_count,
        "duplicate_id_count": int(df["id"].duplicated().sum())
            if "id" in df.columns else np.nan
    }])
    save_csv(duplicate_report, "duplicate_records_report.csv")

    # 5. Invalid values: domain checks for common Divar Cars fields
    invalid_rows = []
    def add_invalid(col, rule, mask):
        mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
        invalid_rows.append({
            "column": col,
            "invalid_rule": rule,
            "invalid_count": int(mask.sum()),
            "examples": " | ".join(map(str, df.loc[mask, col].head(5).tolist()))
        })

    if "Price" in parsed:
        p = parsed["Price"]
        add_invalid("Price", "Price is zero or negative", p.notna() & (p <= 0))
        add_invalid("Price", "Price is present but cannot be parsed as numeric",
                    df["Price"].notna() & p.isna())
    if "Mileage" in parsed:
        m = parsed["Mileage"]
        add_invalid("Mileage", "Mileage is negative", m.notna() & (m < 0))
        add_invalid("Mileage", "Mileage is present but cannot be parsed as numeric",
                    df["Mileage"].notna() & m.isna())
    if "Model_Year" in parsed:
        y = parsed["Model_Year"]
        plausible = ((y >= 1300) & (y <= 1500)) | ((y >= 1900) & (y <= 2026))
        add_invalid("Model_Year", "Model year outside plausible Gregorian/Persian year ranges",
                    y.notna() & ~plausible)
    save_csv(pd.DataFrame(invalid_rows), "invalid_values_report.csv")

    # 6. Numerical summary (native numeric columns)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        numeric_summary = df[numeric_columns].describe().T.reset_index()
        numeric_summary.rename(columns={"index": "column"}, inplace=True)
        numeric_summary["missing_count"] = [int(df[c].isna().sum()) for c in numeric_columns]
        numeric_summary["unique_count"] = [int(df[c].nunique(dropna=True)) for c in numeric_columns]
    else:
        numeric_summary = pd.DataFrame(columns=["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"])
    save_csv(numeric_summary, "numerical_summary.csv")

    # 7. Categorical summary
    categorical_columns = [c for c in column_names if c not in numeric_columns]
    categorical_rows = []
    for col in categorical_columns:
        series = df[col]
        counts = series.value_counts(dropna=True)
        categorical_rows.append({
            "column": col,
            "non_null_count": int(series.notna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "top_value": str(counts.index[0]) if len(counts) else "",
            "top_frequency": int(counts.iloc[0]) if len(counts) else 0,
            "top_percent_non_null": round(counts.iloc[0] / series.notna().sum() * 100, 4)
                if len(counts) and series.notna().sum() else np.nan
        })
    save_csv(pd.DataFrame(categorical_rows), "categorical_summary.csv")

    # 8. Outlier detection using IQR (flag only; do not delete records)
    outlier_rows = []
    outlier_candidates = {}
    for col in numeric_columns:
        outlier_candidates[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Price", "Mileage", "Model_Year"]:
        if col in parsed:
            outlier_candidates[col + "_parsed"] = parsed[col]

    for col, series in outlier_candidates.items():
        values = series.dropna()
        if len(values) == 0:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((values < lower) | (values > upper)).sum())
        outlier_rows.append({
            "column": col,
            "method": "IQR (1.5 x IQR)",
            "q1": q1, "q3": q3,
            "lower_bound": lower, "upper_bound": upper,
            "outlier_count": count,
            "outlier_percent": round(count / len(values) * 100, 4)
        })
    save_csv(pd.DataFrame(outlier_rows), "outlier_report.csv")

    # 9. Price analysis
    price_rows = []
    if "Price" in parsed:
        price = parsed["Price"].dropna()
        if len(price):
            metrics = {
                "count": int(price.count()),
                "missing_count": int(df["Price"].isna().sum()),
                "mean": price.mean(),
                "median": price.median(),
                "standard_deviation": price.std(),
                "minimum": price.min(),
                "q1": price.quantile(0.25),
                "q3": price.quantile(0.75),
                "maximum": price.max(),
                "skewness": price.skew(),
                "positive_count": int((price > 0).sum()),
                "zero_or_negative_count": int((price <= 0).sum())
            }
            price_rows = [{"metric": key, "value": value} for key, value in metrics.items()]
    save_csv(pd.DataFrame(price_rows, columns=["metric", "value"]), "price_analysis_report.csv")

    # 10. Correlation matrix (numeric columns + parsed numeric fields)
    correlation_data = df[numeric_columns].copy()
    for col in ["Price", "Mileage", "Model_Year", "Third_party_insurance"]:
        if col in parsed:
            correlation_data[col + "_parsed"] = parsed[col]
    correlation_matrix = correlation_data.corr(numeric_only=True)
    correlation_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv", encoding="utf-8-sig")

    # 11. Dataset comparison: raw, deduplicated, and price-present subset
    deduplicated = df.drop_duplicates()
    price_subset = df.loc[df["Price"].notna()] if "Price" in df.columns else df.iloc[0:0]
    comparison = pd.DataFrame([
        {
            "dataset_variant": "Raw dataset",
            "rows": len(df), "columns": len(df.columns),
            "missing_cells": int(df.isna().sum().sum()),
            "exact_duplicates": int(df.duplicated().sum())
        },
        {
            "dataset_variant": "After exact duplicate removal (comparison only)",
            "rows": len(deduplicated), "columns": len(deduplicated.columns),
            "missing_cells": int(deduplicated.isna().sum().sum()),
            "exact_duplicates": int(deduplicated.duplicated().sum())
        },
        {
            "dataset_variant": "Rows with non-missing Price",
            "rows": len(price_subset), "columns": len(price_subset.columns),
            "missing_cells": int(price_subset.isna().sum().sum()),
            "exact_duplicates": int(price_subset.duplicated().sum())
        }
    ])
    save_csv(comparison, "dataset_comparison.csv")

    # Temporal coverage
    temporal = {}
    if "Time" in df.columns:
        dates = pd.to_datetime(df["Time"], errors="coerce")
        temporal = {
            "valid_date_count": int(dates.notna().sum()),
            "invalid_or_missing_date_count": int(dates.isna().sum()),
            "earliest_date": str(dates.min()) if dates.notna().any() else None,
            "latest_date": str(dates.max()) if dates.notna().any() else None,
            "distinct_dates": int(dates.nunique())
        }

    # Geographical coverage
    geography = {}
    for col in ["City", "Location"]:
        if col in df.columns:
            geography[col] = {
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_10_values": {
                    str(k): int(v) for k, v in df[col].value_counts(dropna=True).head(10).items()
                }
            }

    # Dataset quality and consistency notes
    completeness = 100 * (1 - missing_cells / total_cells) if total_cells else 0
    consistency_notes = [
        "Original values are preserved; Persian/Arabic digits are parsed only for selected numeric analyses.",
        "IQR outliers are flagged and are not automatically deleted.",
        "Missing values may be structurally expected in optional advertisement fields.",
        "Dataset comparison is performed between variants of this dataset; no external comparison dataset was supplied."
    ]

    # 12. HTML quality report
    schema_html = column_summary.to_html(index=False, border=0, classes="report-table")
    price_html = pd.DataFrame(price_rows).to_html(index=False, border=0, classes="report-table") if price_rows else "<p>Price summary unavailable.</p>"
    html_report = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Divar Cars Data Quality Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
h1, h2 {{ color: #23395d; }}
.report-table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
.report-table th, .report-table td {{ border: 1px solid #ddd; padding: 7px; text-align: left; }}
.report-table th {{ background: #edf1f7; }}
.metric {{ display: inline-block; padding: 12px; margin: 5px; background: #f1f3f6; border-radius: 5px; }}
</style></head><body>
<h1>Divar Cars — Data Quality Report</h1>
<p>Source: Kaggle dataset arianghasemi/divar-cars | Input file: {source_file}</p>
<div class="metric"><b>Rows:</b> {rows:,}</div>
<div class="metric"><b>Columns:</b> {columns}</div>
<div class="metric"><b>Exact duplicates:</b> {duplicate_count:,}</div>
<div class="metric"><b>Missing cells:</b> {missing_cells:,} ({100*missing_cells/total_cells if total_cells else 0:.2f}%)</div>
<h2>Column schema and summary</h2>{schema_html}
<h2>Price analysis</h2>{price_html}
<h2>Temporal coverage</h2><pre>{json.dumps(temporal, ensure_ascii=False, indent=2)}</pre>
<h2>Geographical coverage</h2><pre>{json.dumps(geography, ensure_ascii=False, indent=2)}</pre>
<h2>Data quality assessment</h2>
<p>Completeness score: {completeness:.2f}/100. This score measures completeness only, not total dataset quality.</p>
<ul>{''.join('<li>'+note+'</li>' for note in consistency_notes)}</ul>
<h2>Audit scope</h2>
<p>Dataset loading, structure, shape, schema, data types, numerical and categorical summaries,
missing values, duplicates, unique values, invalid-value checks, target price analysis,
distributions (summarized in CSV outputs), IQR outliers, correlation, feature relationships,
temporal and geographical coverage, consistency, quality assessment, dataset comparison,
and report generation.</p>
</body></html>"""
    (OUTPUT_DIR / "data_quality_report.html").write_text(html_report, encoding="utf-8")

    # 13. PDF report (ReportLab)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        styles = getSampleStyleSheet()
        story = [
            Paragraph("Divar Cars — Data Audit Report", styles["Title"]),
            Spacer(1, 10),
            Paragraph(f"Source: Kaggle arianghasemi/divar-cars | Input: {source_file}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("Dataset Overview", styles["Heading2"])
        ]
        overview_table = [
            ["Metric", "Value"],
            ["Rows", f"{rows:,}"],
            ["Columns", str(columns)],
            ["Exact duplicate rows", f"{duplicate_count:,}"],
            ["Missing cells", f"{missing_cells:,} ({100*missing_cells/total_cells if total_cells else 0:.2f}%)"],
            ["Completeness score", f"{completeness:.2f}/100"],
            ["Earliest date", str(temporal.get("earliest_date", "N/A"))],
            ["Latest date", str(temporal.get("latest_date", "N/A"))]
        ]
        table = Table(overview_table, colWidths=[6*cm, 10*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#23395d")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("VALIGN", (0,0), (-1,-1), "TOP")
        ]))
        story += [table, Spacer(1, 12),
                  Paragraph("Price Analysis", styles["Heading2"])]
        if price_rows:
            ptable = [["Metric", "Value"]] + [
                [str(item["metric"]), f'{item["value"]:,.3f}' if isinstance(item["value"], (int, float, np.number)) and pd.notna(item["value"]) else str(item["value"])]
                for item in price_rows
            ]
            t = Table(ptable, colWidths=[6*cm, 10*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#23395d")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey)
            ]))
            story += [t, Spacer(1, 12)]
        story += [
            Paragraph("Temporal Coverage", styles["Heading2"]),
            Paragraph(str(temporal), styles["BodyText"]),
            Spacer(1, 8),
            Paragraph("Geographical Coverage", styles["Heading2"]),
            Paragraph(str(geography), styles["BodyText"]),
            Spacer(1, 8),
            Paragraph("Quality Assessment and Limitations", styles["Heading2"]),
            Paragraph(
                "The completeness score reflects missing cells only. Invalid values, duplicates, "
                "outliers and consistency should be interpreted using domain knowledge. Outliers "
                "are flagged, not removed. The dataset comparison uses variants of this dataset; "
                "no external benchmark dataset was provided. See the accompanying CSV and HTML "
                "files for detailed per-column results.",
                styles["BodyText"]
            )
        ]
        pdf_path = OUTPUT_DIR / "data_audit_report.pdf"
        SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=1.5*cm, bottomMargin=1.5*cm
        ).build(story)
    except ImportError:
        print("WARNING: reportlab is not installed; PDF was not generated. Run: pip install reportlab")

    print("\\nAudit completed.")
    print(f"Rows: {rows:,} | Columns: {columns}")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")
    print("Expected outputs: 13 files (11 CSV + HTML + PDF).")

if __name__ == "__main__":
    main()
