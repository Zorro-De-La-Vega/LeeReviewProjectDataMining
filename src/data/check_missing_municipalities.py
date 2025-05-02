"""
Script to identify missing municipalities in the consolidated dataset
"""

import pandas as pd
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
CONSOLIDATED_DIR = PROJECT_ROOT / 'data' / 'consolidated'

# Load the complete Miami-Dade dataset
complete_file = PROCESSED_DIR / 'airbnb' / 'complete_miami_dade_airbnb_data.csv'
consolidated_file = CONSOLIDATED_DIR / 'airbnb_comprehensive.csv'

if os.path.exists(complete_file):
    complete_df = pd.read_csv(complete_file)
    print(f"Loaded complete dataset with {len(complete_df)} records")
    
    # Extract all municipalities
    municipalities = complete_df[complete_df['area_type'] == 'Municipality']
    print(f"Found {len(municipalities)} municipalities in complete dataset:")
    for i, row in municipalities.iterrows():
        print(f"{i+1}. {row['area_name']} ({row['designation']}) - Population: {row['population']}, Airbnb count: {row['airbnb_count']}")
else:
    print(f"Complete dataset not found at {complete_file}")

# Check consolidated dataset
if os.path.exists(consolidated_file):
    consolidated_df = pd.read_csv(consolidated_file)
    print(f"\nLoaded consolidated dataset with {len(consolidated_df)} records")
    
    # Extract all municipalities from the consolidated dataset
    if 'municipality' in consolidated_df.columns:
        unique_municipalities = consolidated_df['municipality'].unique()
        print(f"Found {len([m for m in unique_municipalities if str(m) != 'nan' and str(m) != 'Unknown'])} municipalities in consolidated dataset:")
        for i, muni in enumerate([m for m in unique_municipalities if str(m) != 'nan' and str(m) != 'Unknown']):
            print(f"{i+1}. {muni}")
    else:
        print("Municipality column not found in consolidated dataset")
else:
    print(f"Consolidated dataset not found at {consolidated_file}")
