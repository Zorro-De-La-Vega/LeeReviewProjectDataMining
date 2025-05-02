"""
Merges cleaned Miami datasets (crime, property, rental, census, airbnb) on common keys (ZIP code, neighborhood) where possible.
Outputs merged datasets for efficient analysis and visualization.
"""
import os
import pandas as pd
import logging

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

PROCESSED = os.path.join(project_root(), 'data', 'processed')
FILES = {
    'crime1': os.path.join(PROCESSED, 'crime', 'cleaned_crime_1.csv'),
    'crime2': os.path.join(PROCESSED, 'crime', 'cleaned_crime_2.csv'),
    'property': os.path.join(PROCESSED, 'property', 'cleaned_property.csv'),
    'rental': os.path.join(PROCESSED, 'rental', 'cleaned_rental.csv'),
    'census': os.path.join(PROCESSED, 'census', 'cleaned_census.csv'),
    'airbnb': os.path.join(PROCESSED, 'airbnb', 'cleaned_airbnb.csv'),
}
MERGED_DIR = os.path.join(PROCESSED, 'merged')
os.makedirs(MERGED_DIR, exist_ok=True)


def read_with_keys(path, keys):
    try:
        df = pd.read_csv(path)
        for k in keys:
            if k in df.columns:
                df[k] = df[k].astype(str).str.strip().str.upper()
        return df
    except Exception as e:
        logging.warning(f"Failed to read {path}: {e}")
        return None

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # Try merging on ZIP code
    keys_zip = ['ZIP', 'zipcode', 'Zip Code', 'zip']
    keys_nh = ['neighbourhood', 'Neighborhood', 'neighborhood']
    dfs = {k: read_with_keys(v, keys_zip + keys_nh) for k, v in FILES.items() if os.path.exists(v)}
    # --- ZIP merge ---
    zip_keys = {k: next((col for col in keys_zip if col in (df.columns if df is not None else [])), None) for k, df in dfs.items()}
    to_merge_zip = {k: df for k, df in dfs.items() if zip_keys[k] is not None and df is not None}
    if len(to_merge_zip) >= 2:
        keys = list(to_merge_zip.keys())
        merged = to_merge_zip[keys[0]]
        for k in keys[1:]:
            merged = pd.merge(merged, to_merge_zip[k], how='outer', left_on=zip_keys[keys[0]], right_on=zip_keys[k], suffixes=(None, f'_{k}'))
        merged_path = os.path.join(MERGED_DIR, 'merged_by_zip.csv')
        merged.to_csv(merged_path, index=False)
        logging.info(f"Merged dataset by ZIP code saved to {merged_path}")
    else:
        logging.warning("Not enough datasets with ZIP code to merge. Trying neighbourhood-based merge.")
        # --- Neighborhood merge ---
        nh_keys = {k: next((col for col in keys_nh if col in (df.columns if df is not None else [])), None) for k, df in dfs.items()}
        to_merge_nh = {k: df for k, df in dfs.items() if nh_keys[k] is not None and df is not None}
        if len(to_merge_nh) >= 2:
            keys = list(to_merge_nh.keys())
            merged = to_merge_nh[keys[0]]
            for k in keys[1:]:
                merged = pd.merge(merged, to_merge_nh[k], how='outer', left_on=nh_keys[keys[0]], right_on=nh_keys[k], suffixes=(None, f'_{k}'))
            merged_path = os.path.join(MERGED_DIR, 'merged_by_neighbourhood.csv')
            merged.to_csv(merged_path, index=False)
            logging.info(f"Merged dataset by neighbourhood saved to {merged_path}")
        else:
            logging.warning("Not enough datasets with neighbourhood to merge.")

if __name__ == "__main__":
    main()
