import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root / "data/raw/bama_ml_dataset.csv"
)

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

price_columns = [
    col for col in df.columns
    if "price" in col.lower() or "قیمت" in col
]

if price_columns:

    price_col = price_columns[0]

    prices = pd.to_numeric(
        df[price_col], errors="coerce"
    ).dropna()

    report = pd.DataFrame({
        "Metric": [
            "Mean",
            "Median",
            "Minimum",
            "Maximum",
            "Standard Deviation"
        ],
        "Value": [
            prices.mean(),
            prices.median(),
            prices.min(),
            prices.max(),
            prices.std()
        ]
    })

else:
    report = pd.DataFrame({
        "Status": ["Price column not found"]
    })

report.to_csv(
    output_path / "price_analysis_report.csv",
    index=False
)

print("Price Analysis Completed")
