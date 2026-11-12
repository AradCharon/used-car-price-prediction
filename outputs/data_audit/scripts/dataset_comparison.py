import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

raw_path = project_root / "data/raw/bama_ml_dataset.csv"

clean_path = project_root / "data/processed/bama_ml_dataset_clean.csv"

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(raw_path)

if clean_path.exists():

    df_clean = pd.read_csv(clean_path)

    comparison = pd.DataFrame({
        "Dataset": ["Raw", "Cleaned"],
        "Rows": [
            len(df_raw),
            len(df_clean)
        ],
        "Columns": [
            len(df_raw.columns),
            len(df_clean.columns)
        ]
    })

else:

    comparison = pd.DataFrame({
        "Status": ["Cleaned dataset not found"]
    })

comparison.to_csv(
    output_path / "dataset_comparison.csv",
    index=False
)

print("Dataset Comparison Completed")
