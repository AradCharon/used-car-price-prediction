import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root / "data/raw/bama_ml_dataset.csv"
)

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

numerical = df.select_dtypes(include="number")

correlation = numerical.corr()

correlation.to_csv(
    output_path / "correlation_matrix.csv"
)

print("Correlation Analysis Completed")
