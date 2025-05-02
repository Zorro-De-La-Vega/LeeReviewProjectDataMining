"""
Modular cleaning script for Miami crime datasets.
- Handles multiple raw crime CSVs
- Cleans, normalizes, and merges on ZIP/neighborhood where possible
- Outputs cleaned CSVs to data/processed/crime/
"""
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import logging
from src.data.merge.neighborhood_standardizer import standardize_neighborhood_series

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

RAW_FILES = [
    os.path.join(project_root(), 'data', 'raw', 'crime', 'miami_dade_part1_crimes_ytd_comparison_2024_2025.csv'),
    os.path.join(project_root(), 'data', 'raw', 'crime', 'MiamiDadeMatters_indicator_data_download_20250420.csv')
]
PROCESSED_DIR = os.path.join(project_root(), 'data', 'processed', 'crime')
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_and_merge_crime():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    dfs = []
    for file in RAW_FILES:
        try:
            logging.info(f"Reading {file}")
            try:
                df = pd.read_csv(file)
                logging.info(f"Read {file} with default parser.")
            except Exception as e1:
                logging.warning(f"Default parser failed for {file}: {e1}. Retrying with on_bad_lines='skip' and sep=','.")
                try:
                    df = pd.read_csv(file, on_bad_lines='skip', sep=',')
                    logging.info(f"Read {file} with on_bad_lines='skip'.")
                except Exception as e2:
                    logging.warning(f"on_bad_lines='skip' failed: {e2}. Retrying with sep=';'.")
                    df = pd.read_csv(file, on_bad_lines='skip', sep=';')
                    logging.info(f"Read {file} with sep=';'.")
            df = df.rename(columns=lambda x: x.strip())
            # Standardize ZIP code column
            zip_cols = [c for c in df.columns if c.lower() in ['zip', 'zipcode', 'zip code', 'postal_code']]
            if zip_cols:
                df['zipcode'] = df[zip_cols[0]].astype(str).str.strip().str.upper()
            df = df.drop_duplicates()
            # Standardize neighbourhood if present
            if 'neighbourhood' in df.columns:
                df['neighbourhood'] = standardize_neighborhood_series(df['neighbourhood'])
            dfs.append(df)
        except Exception as e:
            logging.error(f"Failed to read {file}: {e}")
    # Attempt to merge on ZIP or neighborhood if possible
    if len(dfs) > 1:
        merge_keys = [k for k in ['ZIP', 'zipcode', 'Neighborhood', 'neighborhood'] if k in dfs[0].columns and k in dfs[1].columns]
        if merge_keys:
            merged = pd.merge(dfs[0], dfs[1], how='outer', on=merge_keys[0])
            merged.to_csv(os.path.join(PROCESSED_DIR, 'merged_crime.csv'), index=False)
            logging.info(f"Merged crime data saved to {os.path.join(PROCESSED_DIR, 'merged_crime.csv')}")
        else:
            for i, df in enumerate(dfs):
                df.to_csv(os.path.join(PROCESSED_DIR, f'cleaned_crime_{i+1}.csv'), index=False)
                logging.info(f"Cleaned crime data saved to {os.path.join(PROCESSED_DIR, f'cleaned_crime_{i+1}.csv')}")
    elif dfs:
        dfs[0].to_csv(os.path.join(PROCESSED_DIR, 'cleaned_crime.csv'), index=False)
        logging.info(f"Cleaned crime data saved to {os.path.join(PROCESSED_DIR, 'cleaned_crime.csv')}")

    # --- Summary metrics ---
    # Try to summarize merged data if available, else summarize each cleaned df
    import glob
    summary_path = os.path.join(PROCESSED_DIR, 'summary_crime.csv')
    summary_cols = None
    crime_count_col = None
    # Try merged file first
    merged_file = os.path.join(PROCESSED_DIR, 'merged_crime.csv')
    summary_df = None
    if os.path.exists(merged_file):
        df = pd.read_csv(merged_file)
        summary_cols = [c for c in ['neighbourhood', 'zipcode'] if c in df.columns]
        crime_count_col = next((c for c in df.columns if 'count' in c.lower() or 'crime' in c.lower()), None)
        if summary_cols and crime_count_col:
            summary_df = df.groupby(summary_cols)[crime_count_col].sum().reset_index()
    else:
        # Try each cleaned crime file
        for cleaned_path in glob.glob(os.path.join(PROCESSED_DIR, 'cleaned_crime*.csv')):
            df = pd.read_csv(cleaned_path)
            summary_cols = [c for c in ['neighbourhood', 'zipcode'] if c in df.columns]
            crime_count_col = next((c for c in df.columns if 'count' in c.lower() or 'crime' in c.lower()), None)
            if summary_cols and crime_count_col:
                temp = df.groupby(summary_cols)[crime_count_col].sum().reset_index()
                if summary_df is None:
                    summary_df = temp
                else:
                    summary_df = pd.concat([summary_df, temp], ignore_index=True)
        if summary_df is not None:
            summary_df = summary_df.groupby(summary_cols).sum().reset_index()
    if summary_df is not None:
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Summary crime metrics saved to {summary_path}")
    else:
        logging.warning("No suitable columns for summary metrics in crime data.")

if __name__ == "__main__":
    clean_and_merge_crime()
