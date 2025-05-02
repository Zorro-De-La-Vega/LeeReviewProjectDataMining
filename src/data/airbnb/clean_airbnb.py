"""
Modular cleaning script for Airbnb Miami data.
- Reads raw Airbnb CSV in chunks (handles large files)
- Cleans and normalizes columns (price, location, types, etc.)
- Handles missing values and types robustly
- Logs missing columns and cleaning steps for transparency
- Outputs cleaned CSV to data/processed/airbnb/
"""
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Paths relative to the project root
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

RAW_PATH = os.path.join(project_root(), 'data', 'raw', 'airbnb', 'AirBnb-Data-MiamiData.csv')
PROCESSED_DIR = os.path.join(project_root(), 'data', 'processed', 'airbnb')
PROCESSED_PATH = os.path.join(PROCESSED_DIR, 'cleaned_airbnb.csv')

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define columns to keep (customize as needed)
COLUMNS_TO_KEEP = [
    'name', 'property_type', 'room_type', 'city', 'neighbourhood', 'zipcode',
    'latitude', 'longitude', 'price', 'minimum_nights', 'maximum_nights',
    'number_of_reviews', 'review_scores_rating', 'amenities', 'host_id', 'id'
]

# Standardization mapping for Airbnb
COLUMN_MAP = {
    'Zipcode': 'zipcode',
    'Neighbourhood': 'neighbourhood',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Average Daily Rate (USD)': 'price',
}


import logging
from src.data.merge.neighborhood_standardizer import standardize_neighborhood_series

def clean_chunk(chunk):
    # Rename columns if needed, handle missing, convert types, filter columns
    chunk = chunk.rename(columns=lambda x: x.strip())
    # Standardize column names
    for old, new in COLUMN_MAP.items():
        if old in chunk.columns and new not in chunk.columns:
            chunk[new] = chunk[old]
    # Keep only columns that exist in this chunk
    cols = [c for c in COLUMNS_TO_KEEP if c in chunk.columns]
    chunk = chunk[cols]
    # Standardize neighbourhood names if present
    if 'neighbourhood' in chunk:
        chunk['neighbourhood'] = standardize_neighborhood_series(chunk['neighbourhood'])
    # Example cleaning: price to float, fillna, etc.
    if 'price' in chunk:
        chunk['price'] = pd.to_numeric(chunk['price'], errors='coerce')
    if 'review_scores_rating' in chunk:
        chunk['review_scores_rating'] = pd.to_numeric(chunk['review_scores_rating'], errors='coerce')
    chunk = chunk.drop_duplicates()
    # Drop rows with NA for key columns, but only if those columns exist
    required_cols = [c for c in ['latitude', 'longitude', 'price', 'zipcode'] if c in chunk.columns]
    missing_cols = [c for c in ['latitude', 'longitude', 'price', 'zipcode'] if c not in chunk.columns]
    if missing_cols:
        logging.warning(f"Missing columns in chunk: {missing_cols}")
    if required_cols:
        chunk = chunk.dropna(subset=required_cols, how='any')
    return chunk

def main():
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    CHUNKSIZE = 10000
    cleaned_chunks = []
    logging.info(f"Reading raw Airbnb data from {RAW_PATH}")
    for i, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=CHUNKSIZE, low_memory=False)):
        logging.info(f"Cleaning chunk {i+1}")
        cleaned = clean_chunk(chunk)
        cleaned_chunks.append(cleaned)
    if cleaned_chunks:
        df_clean = pd.concat(cleaned_chunks, ignore_index=True)
    else:
        df_clean = pd.DataFrame(columns=COLUMNS_TO_KEEP)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    logging.info(f"Cleaned Airbnb data saved to {PROCESSED_PATH}")

    # --- Summary metrics ---
    summary_dir = PROCESSED_DIR
    summary_path = os.path.join(summary_dir, 'summary_airbnb.csv')
    # Group by neighborhood and zipcode, calculate average price, count, and avg review score
    summary_cols = [c for c in ['neighbourhood', 'zipcode'] if c in df_clean.columns]
    metrics = {}
    if summary_cols:
        metrics['avg_price'] = ('price', 'mean') if 'price' in df_clean.columns else None
        metrics['count'] = ('price', 'count') if 'price' in df_clean.columns else None
        metrics['avg_review_score'] = ('review_scores_rating', 'mean') if 'review_scores_rating' in df_clean.columns else None
        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        if metrics:
            summary = df_clean.groupby(summary_cols).agg(**metrics).reset_index()
            summary.to_csv(summary_path, index=False)
            logging.info(f"Summary Airbnb metrics saved to {summary_path}")
        else:
            logging.warning("No suitable columns for summary metrics in Airbnb data.")
    else:
        logging.warning("No neighborhood or zipcode columns for summary metrics in Airbnb data.")

if __name__ == "__main__":
    main()
