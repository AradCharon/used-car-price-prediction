# Data Cleaning

This directory contains the data cleaning pipeline
for the Bama Used Car Price Prediction project.

## Dataset

Input:
data/raw/bama_ml_dataset.csv

Output:
data/processed/bama_ml_dataset_clean.csv

## Cleaning Steps

1. Load the raw dataset.
2. Check missing values.
3. Remove duplicate records.
4. Remove invalid prices.
5. Remove unrealistic engine volumes.
6. Validate production years.
7. Save the cleaned dataset.

## Dataset Information

- Initial rows: 39,526
- Initial columns: 13
- Final rows: 27,562
- Removed rows: 11,964

## Requirements

Install dependencies:

pip install -r requirements-cleaning.txt

## Run

python data_cleaning/data_cleaning.py

## Output Files

The cleaning process generates:

- bama_ml_dataset_clean.csv
- cleaning_summary.csv
- data_before_after_comparison.csv
- missing_values_after_cleaning.csv
