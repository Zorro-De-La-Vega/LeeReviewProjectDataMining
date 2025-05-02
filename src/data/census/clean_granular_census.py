"""
Script: clean_granular_census.py
Purpose: Clean and summarize tract-level ACS census data for Miami-Dade County.
Inputs: data/raw/census/acs5_miamidade_tracts.csv
Outputs: data/processed/census/summary_census.csv

- Computes summary statistics (mean, median, count, std) for each numerical variable across all tracts.
- Ensures GEOID is preserved for spatial joins.
- Modular, reproducible, and ready for integration.

Usage:
    python src/data/census/clean_granular_census.py
"""
import os
import pandas as pd

RAW_PATH = 'data/raw/census/acs5_miamidade_tracts.csv'
PROCESSED_DIR = 'data/processed/census'
SUMMARY_PATH = os.path.join(PROCESSED_DIR, 'summary_census.csv')

os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_and_summarize():
    df = pd.read_csv(RAW_PATH)
    # Clean: remove any duplicate GEOIDs, drop empty columns
    df = df.drop_duplicates(subset=['GEOID'])
    df = df.dropna(axis=1, how='all')
    # Summary: compute mean, median, std, count for all numeric columns (excluding GEOID, tract, county, state, censusgeo)
    exclude_cols = {'GEOID', 'tract', 'county', 'state', 'censusgeo'}
    num_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    summary = df[num_cols].agg(['mean', 'median', 'std', 'count']).T.reset_index()
    summary.columns = ['variable', 'mean', 'median', 'std', 'count']
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"Wrote summary: {SUMMARY_PATH}")

if __name__ == "__main__":
    clean_and_summarize()
