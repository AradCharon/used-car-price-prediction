import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root / "data/raw/bama_ml_dataset.csv"
)

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

overview = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Duplicate Rows",
        "Total Missing Values"
    ],
    "Value": [
        df.shape[0],
        df.shape[1],
        df.duplicated().sum(),
        df.isnull().sum().sum()
    ]
})

overview.to_csv(
    output_path / "dataset_overview.csv",
    index=False
)

print("Dataset Overview Completed")
