"""
Script: clean_crime.py
Purpose: Clean and prepare Miami-Dade crime data for Tableau/analysis
"""
import pandas as pd
import os

def clean_crime(input_path, output_path):
    df = pd.read_csv(input_path)
    # Remove subtotal rows (where UCR Code and Description are blank)
    df_clean = df[df['UCR Code'].notna() & df['UCR Code Description'].notna()]
    # Remove total rows (e.g., 'PART 1 CRIMES TOTAL', 'TOTAL PART 1 - VIOLENT CRIMES', etc.)
    df_clean = df_clean[~df_clean['Crime Type'].str.contains('TOTAL|PART 1 CRIMES TOTAL', na=False)]
    # Clean column names for Tableau compatibility
    df_clean.columns = [c.strip().replace(' ', '_').replace('-', '').lower() for c in df_clean.columns]
    # Export cleaned file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw = "../../data/raw/crime/miami_dade_part1_crimes_ytd_comparison_2024_2025.csv"
    processed = "../../data/processed/miami_dade_part1_crimes_ytd_comparison_2024_2025_clean.csv"
    clean_crime(raw, processed)
