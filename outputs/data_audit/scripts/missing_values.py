import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root / "data/raw/bama_ml_dataset.csv"
)

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isnull().sum().values,
    "Missing Percentage": (
        df.isnull().mean().values * 100
    ).round(2)
})

missing.to_csv(
    output_path / "missing_values_report.csv",
    index=False
)

print("Missing Values Report Completed")
