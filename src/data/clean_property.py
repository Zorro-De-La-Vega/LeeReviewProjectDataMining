"""
Script: clean_property.py
Purpose: Clean and prepare property/rent datasets for Tableau/analysis
"""
import pandas as pd
import os

def clean_property(input_path, output_path):
    df = pd.read_csv(input_path)
    # Example cleaning: drop rows with all NaNs, strip whitespace from columns
    df_clean = df.dropna(how='all')
    df_clean.columns = [c.strip().replace(' ', '_').replace('-', '').lower() for c in df_clean.columns]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw = "../../data/raw/property/zumper_miami_rent_summary_apr2025.csv"
    processed = "../../data/processed/zumper_miami_rent_summary_apr2025_clean.csv"
    clean_property(raw, processed)
