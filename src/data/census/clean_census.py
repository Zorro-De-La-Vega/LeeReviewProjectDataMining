"""
Modular cleaning script for Miami census data.
- Cleans and normalizes census CSV
- Outputs cleaned CSV to data/processed/census/
"""
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import logging
from src.data.merge.neighborhood_standardizer import standardize_neighborhood_series

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

RAW_PATH = os.path.join(project_root(), 'data', 'raw', 'census', 'Census_Median Household Income_Miami_Dade.csv')
PROCESSED_DIR = os.path.join(project_root(), 'data', 'processed', 'census')
PROCESSED_PATH = os.path.join(PROCESSED_DIR, 'cleaned_census.csv')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_census():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        df = pd.read_csv(RAW_PATH)
        df = df.rename(columns=lambda x: x.strip())
        # Standardize ZIP code column
        zip_cols = [c for c in df.columns if c.lower() in ['zip', 'zipcode', 'zip code', 'postal_code']]
        if zip_cols:
            df['zipcode'] = df[zip_cols[0]].astype(str).str.strip().str.upper()
        df = df.drop_duplicates()
        # Example: force numeric for income columns
        income_cols = [c for c in df.columns if 'income' in c.lower()]
        for col in income_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Standardize neighbourhood if present
        if 'neighbourhood' in df.columns:
            df['neighbourhood'] = standardize_neighborhood_series(df['neighbourhood'])
        df = df.dropna()
        df.to_csv(PROCESSED_PATH, index=False)
        logging.info(f"Cleaned census data saved to {PROCESSED_PATH}")

        # --- Summary metrics ---
        summary_dir = PROCESSED_DIR
        summary_path = os.path.join(summary_dir, 'summary_census.csv')
        summary_cols = [c for c in ['neighbourhood', 'zipcode'] if c in df.columns]
        income_cols = [c for c in df.columns if 'income' in c.lower()]
        metrics = {}
        if summary_cols and income_cols:
            metrics['median_income'] = (income_cols[0], 'median')
            metrics['count'] = (income_cols[0], 'count')
            summary = df.groupby(summary_cols).agg(**metrics).reset_index()
            summary.to_csv(summary_path, index=False)
            logging.info(f"Summary census metrics saved to {summary_path}")
        else:
            logging.warning("No neighborhood/zipcode or income columns for summary metrics in census data.")
    except Exception as e:
        logging.error(f"Failed to clean census data: {e}")

if __name__ == "__main__":
    clean_census()
