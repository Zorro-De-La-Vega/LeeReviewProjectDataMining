"""
Modular cleaning script for Miami property sales data.
- Cleans and normalizes property sales CSV
- Outputs cleaned CSV to data/processed/property/
"""
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import logging
from src.data.merge.neighborhood_standardizer import standardize_neighborhood_series

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

RAW_PATH = os.path.join(project_root(), 'data', 'raw', 'property', 'miami-housing-kaggle.csv')
PROCESSED_DIR = os.path.join(project_root(), 'data', 'processed', 'property')
PROCESSED_PATH = os.path.join(PROCESSED_DIR, 'cleaned_property.csv')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_property():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        try:
            df = pd.read_csv(RAW_PATH)
        except Exception as e1:
            logging.warning(f"Default parser failed: {e1}. Retrying with on_bad_lines='skip'.")
            df = pd.read_csv(RAW_PATH, on_bad_lines='skip')
        df = df.rename(columns=lambda x: x.strip())
        # Standardize columns
        col_map = {'LATITUDE': 'latitude', 'LONGITUDE': 'longitude', 'SALE_PRC': 'price', 'Zip Code': 'zipcode', 'ZIP': 'zipcode', 'zip': 'zipcode', 'zipcode': 'zipcode'}
        for old, new in col_map.items():
            if old in df and new not in df:
                df[new] = df[old]
        df = df.drop_duplicates()
        # Ensure numeric
        for col in ['price', 'latitude', 'longitude']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Ensure output always has header
        out_cols = ['parcelno', 'latitude', 'longitude', 'price', 'zipcode']
        for c in out_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[out_cols]
        # Standardize neighbourhood if present
        if 'neighbourhood' in df.columns:
            df['neighbourhood'] = standardize_neighborhood_series(df['neighbourhood'])
        missing_cols = [c for c in ['latitude', 'longitude', 'price', 'zipcode'] if df[c].isna().all()]
        if missing_cols:
            logging.warning(f"Missing columns: {missing_cols}. Columns present: {list(df.columns)}. Saving available data.")
        df.to_csv(PROCESSED_PATH, index=False)
        logging.info(f"Cleaned property data saved to {PROCESSED_PATH}")

        # --- Summary metrics ---
        summary_dir = PROCESSED_DIR
        summary_path = os.path.join(summary_dir, 'summary_property.csv')
        summary_cols = [c for c in ['neighbourhood', 'zipcode'] if c in df.columns]
        metrics = {}
        if summary_cols:
            metrics['avg_price'] = ('price', 'mean') if 'price' in df.columns else None
            metrics['count'] = ('price', 'count') if 'price' in df.columns else None
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                summary = df.groupby(summary_cols).agg(**metrics).reset_index()
                summary.to_csv(summary_path, index=False)
                logging.info(f"Summary property metrics saved to {summary_path}")
            else:
                logging.warning("No suitable columns for summary metrics in property data.")
        else:
            logging.warning("No neighborhood or zipcode columns for summary metrics in property data.")
    except Exception as e:
        logging.error(f"Failed to clean property data: {e}")

if __name__ == "__main__":
    clean_property()
