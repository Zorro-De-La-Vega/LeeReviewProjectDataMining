"""
Script to analyze the Airbnb data and identify why there are more entries than municipalities
"""

import pandas as pd
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATED_DIR = PROJECT_ROOT / 'data' / 'consolidated'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

# Analyze consolidated dataset
consolidated_file = CONSOLIDATED_DIR / 'airbnb_comprehensive.csv'
if os.path.exists(consolidated_file):
    df = pd.read_csv(consolidated_file)
    print(f"Total entries in consolidated file: {len(df)}")
    
    # Count unique municipalities
    unique_municipalities = [m for m in df['municipality'].unique() 
                           if str(m) != 'nan' and str(m) != 'Unknown']
    print(f"Unique municipalities: {len(unique_municipalities)}")
    
    # Count entries with unknown municipalities
    unknown_entries = len(df[df['municipality'].isin(['Unknown']) | pd.isna(df['municipality'])])
    print(f"Entries with unknown municipality: {unknown_entries}")
    
    # Count entries by municipality status
    neighborhood_counts = df.groupby('neighborhood').size().reset_index(name='count')
    print(f"\nEntries by neighborhood:")
    for idx, row in neighborhood_counts.sort_values('count', ascending=False).head(10).iterrows():
        print(f"  {row['neighborhood']}: {row['count']}")
    
    # Check for neighborhoods that aren't municipalities
    municipalities_list = df['municipality'].dropna().unique().tolist()
    neighborhoods_not_municipalities = []
    for neighborhood in df['neighborhood'].unique():
        if neighborhood not in municipalities_list and neighborhood != 'Unknown':
            neighborhoods_not_municipalities.append(neighborhood)
    
    print(f"\nFound {len(neighborhoods_not_municipalities)} neighborhoods that aren't municipalities:")
    for idx, neighborhood in enumerate(sorted(neighborhoods_not_municipalities)):
        print(f"  {idx+1}. {neighborhood}")
    
    # Compare with the original complete Miami-Dade dataset
    complete_file = PROCESSED_DIR / 'airbnb' / 'complete_miami_dade_airbnb_data.csv'
    if os.path.exists(complete_file):
        complete_df = pd.read_csv(complete_file)
        municipalities = complete_df[complete_df['area_type'] == 'Municipality']
        cdps = complete_df[complete_df['area_type'] == 'Census-Designated Place']
        print(f"\nIn complete dataset:")
        print(f"  Municipalities: {len(municipalities)}")
        print(f"  Census-Designated Places: {len(cdps)}")
        print(f"  Total areas: {len(complete_df)}")
else:
    print(f"Consolidated file not found at {consolidated_file}")
