import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    project_root / "data/raw/bama_ml_dataset.csv"
)

output_path = project_root / "outputs/data_audit"
output_path.mkdir(parents=True, exist_ok=True)

categorical = df.select_dtypes(
    include=["object", "category"]
)

report = []

for col in categorical.columns:

    mode = categorical[col].mode()

    report.append({
        "Column": col,
        "Unique Values": categorical[col].nunique(),
        "Most Frequent Value": mode.iloc[0] if not mode.empty else None
    })

pd.DataFrame(report).to_csv(
    output_path / "categorical_summary.csv",
    index=False
)

print("Categorical Summary Completed")
