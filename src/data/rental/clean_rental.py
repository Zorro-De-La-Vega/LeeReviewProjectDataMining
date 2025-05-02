"""
Modular cleaning script for Miami rental data.
- Cleans and normalizes rental CSV
- Outputs cleaned CSV to data/processed/rental/
"""
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import logging
from src.data.merge.neighborhood_standardizer import standardize_neighborhood_series

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

RAW_PATH = os.path.join(project_root(), 'data', 'raw', 'rental', 'zumper_miami_rent_summary_apr2025.csv')
PROCESSED_DIR = os.path.join(project_root(), 'data', 'processed', 'rental')
PROCESSED_PATH = os.path.join(PROCESSED_DIR, 'cleaned_rental.csv')
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_rental():
    """
    Parses city-level rental data from the raw Zumper CSV, extracting:
    - Average rent by bedroom count (Table 1)
    - Average rent by property type (Table 2)
    Outputs two structured CSVs to data/processed/rental/:
    - avg_rent_by_bedroom.csv
    - avg_rent_by_property_type.csv
    Note: No neighbourhood or zipcode granularity is available in the current data source.
    """
    import re
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    with open(RAW_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Helper to find table start and end
    def extract_table(start_marker, next_marker=None):
        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if start_marker in line:
                start_idx = i + 1
                continue
            if start_idx is not None and next_marker and next_marker in line:
                end_idx = i
                break
        if start_idx is not None:
            if end_idx is None:
                # Until next blank or comment
                for j in range(start_idx, len(lines)):
                    if lines[j].strip() == '' or lines[j].strip().startswith('#'):
                        end_idx = j
                        break
            if end_idx is None:
                end_idx = len(lines)
            return [l for l in lines[start_idx:end_idx] if l.strip() and not l.strip().startswith('#')]
        return []

    # Table 1: Average Rent by Bedroom Count
    table1_lines = extract_table('Table 1:', 'Table 2:')
    if table1_lines:
        import io
        df_bedroom = pd.read_csv(io.StringIO(''.join(table1_lines)))
        df_bedroom.to_csv(os.path.join(PROCESSED_DIR, 'avg_rent_by_bedroom.csv'), index=False)
        logging.info('Saved avg_rent_by_bedroom.csv')
    else:
        logging.warning('Table 1 (bedroom count) not found in raw file.')

    # Table 2: Average Rent by Property Type
    table2_lines = extract_table('Table 2:')
    if table2_lines:
        import io
        df_property = pd.read_csv(io.StringIO(''.join(table2_lines)))
        df_property.to_csv(os.path.join(PROCESSED_DIR, 'avg_rent_by_property_type.csv'), index=False)
        logging.info('Saved avg_rent_by_property_type.csv')
    else:
        logging.warning('Table 2 (property type) not found in raw file.')

    # Document city-level limitation in a README note
    with open(os.path.join(PROCESSED_DIR, 'README.txt'), 'w', encoding='utf-8') as f:
        f.write(
            'NOTE: Rental data is only available at the city level (Miami) for April 2025.\n'
            'No neighbourhood or zipcode breakdown is present in the current data source.\n'
            'See https://www.zumper.com/rent-research/miami-fl for more details.'
        )
    logging.info('Wrote README.txt documenting city-level limitation.')

if __name__ == "__main__":
    clean_rental()
