import os
import pandas as pd
import numpy as np


FILE_PATH = 'cars_dataset.csv'
OUTPUT_DIR = 'audit_reports_csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH)

df_overview = pd.DataFrame([{
    'Total_Records': len(df),
    'Total_Features': len(df.columns),
    'Unique_Brands': df['name'].nunique() if 'name' in df.columns else 0,
    'Unique_Cities': df['city'].nunique() if 'city' in df.columns else 0,
    'Audit_Date': pd.Timestamp.now().strftime('%Y-%m-%d')
}])
df_overview.to_csv(f'{OUTPUT_DIR}/dataset_overview.csv', index=False, encoding='utf-8-sig')



col_summary = pd.DataFrame({
    'Column_Name': df.columns,
    'Data_Type': [str(df[c].dtype) for c in df.columns],
    'Non_Null_Count': [int(df[c].count()) for c in df.columns],
    'Null_Count': [int(df[c].isnull().sum()) for c in df.columns],
    'Unique_Values': [int(df[c].nunique()) for c in df.columns]
})
col_summary.to_csv(f'{OUTPUT_DIR}/column_summary.csv', index=False, encoding='utf-8-sig')


missing_rep = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': [int(df[c].isnull().sum()) for c in df.columns],
    'Missing_Percentage': [round((df[c].isnull().sum() / len(df)) * 100, 2) for c in df.columns]
})
missing_rep.to_csv(f'{OUTPUT_DIR}/missing_values_report.csv', index=False, encoding='utf-8-sig')


subset_cols = [c for c in ['name', 'model', 'year', 'city', 'km', 'condition', 'color', 'price'] if c in df.columns]
dup_rep = df[df.duplicated(subset=subset_cols, keep=False)]
dup_rep.to_csv(f'{OUTPUT_DIR}/duplicate_records_report.csv', index=False, encoding='utf-8-sig')


cond_invalid_price = (df['price'] < 1_000_000) if 'price' in df.columns else False
cond_invalid_km = ((df['km'] == 0) & (df['year'] < 1390)) if ('km' in df.columns and 'year' in df.columns) else False

invalid_rep = df[cond_invalid_price | cond_invalid_km]
invalid_rep.to_csv(f'{OUTPUT_DIR}/invalid_values_report.csv', index=False, encoding='utf-8-sig')


num_cols = [c for c in ['year', 'km', 'price'] if c in df.columns]
num_summary = df[num_cols].describe().T.reset_index().rename(columns={'index': 'Feature'})
num_summary.to_csv(f'{OUTPUT_DIR}/numerical_summary.csv', index=False, encoding='utf-8-sig')


cat_cols = [c for c in ['name', 'model', 'condition', 'color', 'city'] if c in df.columns]
cat_summary = pd.DataFrame({
    'Feature': cat_cols,
    'Most_Frequent': [df[col].mode()[0] if not df[col].mode().empty else None for col in cat_cols],
    'Most_Frequent_Count': [int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0 for col in cat_cols],
    'Unique_Count': [int(df[col].nunique()) for col in cat_cols]
})
cat_summary.to_csv(f'{OUTPUT_DIR}/dataset_comparison.csv', index=False, encoding='utf-8-sig')