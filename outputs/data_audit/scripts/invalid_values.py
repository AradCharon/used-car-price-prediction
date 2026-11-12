import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root / "data/raw/bama_ml_dataset.csv"
)

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

report = []

for col in df.columns:

    blank_values = 0

    if df[col].dtype == "object":
        blank_values = df[col].astype(str).str.strip().eq("").sum()

    report.append({
        "Column": col,
        "Blank Values": blank_values
    })

pd.DataFrame(report).to_csv(
    output_path / "invalid_values_report.csv",
    index=False
)

print("Invalid Values Report Completed")
