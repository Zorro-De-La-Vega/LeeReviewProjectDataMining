"""
Script to verify the accuracy and completeness of the consolidated data
"""

import pandas as pd
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSOLIDATED_DIR = PROJECT_ROOT / 'data' / 'consolidated'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'

def verify_data_integrity():
    """Verify the integrity and accuracy of the consolidated data"""
    print("Verifying data integrity of consolidated files...\n")
    
    # Check Airbnb comprehensive file
    airbnb_file = CONSOLIDATED_DIR / 'airbnb_comprehensive.csv'
    complete_file = PROCESSED_DIR / 'airbnb' / 'complete_miami_dade_airbnb_data.csv'
    
    if os.path.exists(airbnb_file) and os.path.exists(complete_file):
        airbnb_df = pd.read_csv(airbnb_file)
        complete_df = pd.read_csv(complete_file)
        
        print(f"AIRBNB COMPREHENSIVE DATA VERIFICATION:")
        print(f"----------------------------------------")
        print(f"Records in consolidated file: {len(airbnb_df)}")
        
        # Verify municipalities
        municipalities_complete = complete_df[complete_df['area_type'] == 'Municipality']['area_name'].tolist()
        municipalities_consolidated = [m for m in airbnb_df['municipality'].unique() 
                                    if str(m) != 'nan' and str(m) != 'Unknown']
        
        all_found = True
        for muni in municipalities_complete:
            if muni not in municipalities_consolidated:
                print(f"[MISSING] Municipality not found: {muni}")
                all_found = False
        
        if all_found:
            print(f"[SUCCESS] All {len(municipalities_complete)} municipalities present in consolidated data")
        
        # Verify Airbnb counts match between datasets
        count_matches = 0
        count_mismatches = 0
        
        for muni in municipalities_consolidated:
            # Get airbnb count from complete data
            complete_count = complete_df[complete_df['area_name'] == muni]['airbnb_count'].values
            
            # Get airbnb count from consolidated data
            consolidated_count = airbnb_df[airbnb_df['municipality'] == muni]['airbnb_count'].values
            
            if len(complete_count) > 0 and len(consolidated_count) > 0:
                if complete_count[0] == consolidated_count[0]:
                    count_matches += 1
                else:
                    count_mismatches += 1
                    print(f"[MISMATCH] Airbnb count incorrect for {muni}: {consolidated_count[0]} vs {complete_count[0]} in complete data")
        
        if count_mismatches == 0:
            print(f"[SUCCESS] All Airbnb counts match between consolidated and complete data")
        
        # Check for neighborhoods that aren't municipalities
        neighborhoods = set(airbnb_df['neighborhood'].unique())
        municipalities = set(municipalities_consolidated)
        non_muni_neighborhoods = neighborhoods - municipalities
        
        print(f"\nNeighborhoods that aren't municipalities: {len(non_muni_neighborhoods)}")
        for idx, neighborhood in enumerate(sorted(non_muni_neighborhoods)):
            print(f"  {idx+1}. {neighborhood}")
    
    else:
        if not os.path.exists(airbnb_file):
            print(f"[ERROR] Airbnb comprehensive file not found at {airbnb_file}")
        if not os.path.exists(complete_file):
            print(f"[ERROR] Complete Miami-Dade data file not found at {complete_file}")
    
    print("\n----------------------------------------\n")
    
    # Check Property comprehensive file
    property_file = CONSOLIDATED_DIR / 'property_comprehensive.csv'
    if os.path.exists(property_file):
        property_df = pd.read_csv(property_file)
        
        print(f"PROPERTY COMPREHENSIVE DATA VERIFICATION:")
        print(f"----------------------------------------")
        print(f"Records in property file: {len(property_df)}")
        print(f"Unique properties: {property_df['parcelno'].nunique()}")
        
        # Check for price data
        if 'price' in property_df.columns:
            price_stats = property_df['price'].describe()
            print(f"\nProperty Price Statistics:")
            print(f"  Mean price: ${price_stats['mean']:,.2f}")
            print(f"  Median price: ${price_stats['50%']:,.2f}")
            print(f"  Min price: ${price_stats['min']:,.2f}")
            print(f"  Max price: ${price_stats['max']:,.2f}")
            
            print(f"\n[SUCCESS] Property price data is present and appears reliable")
        else:
            print(f"[ERROR] No 'price' column found in property data")
    else:
        print(f"[ERROR] Property comprehensive file not found at {property_file}")
    
    print("\n----------------------------------------\n")
    
    # Check Miami-Dade Merged Data
    merged_file = PROCESSED_DIR / 'miami_dade_merged_data.csv'
    if os.path.exists(merged_file):
        merged_df = pd.read_csv(merged_file)
        
        print(f"MIAMI-DADE MERGED DATA VERIFICATION:")
        print(f"----------------------------------------")
        print(f"Records in merged file: {len(merged_df)}")
        
        # Check for essential columns
        essential_columns = ['neighborhood', 'airbnb_count', 'airbnb_density', 
                            'median_property_price', 'population', 'median_income']
        
        missing_columns = [col for col in essential_columns if col not in merged_df.columns]
        
        if missing_columns:
            print(f"[ERROR] Missing essential columns: {', '.join(missing_columns)}")
        else:
            print(f"[SUCCESS] All essential columns present in merged data")
            
            # Check for data completeness
            non_null_counts = merged_df[essential_columns].count()
            print(f"\nData completeness:")
            for col in essential_columns:
                pct_complete = (non_null_counts[col] / len(merged_df)) * 100
                print(f"  {col}: {non_null_counts[col]} values ({pct_complete:.1f}% complete)")
            
            # Check property prices
            if 'median_property_price' in merged_df.columns:
                price_stats = merged_df['median_property_price'].describe()
                print(f"\nProperty Price Statistics in Merged Data:")
                print(f"  Mean price: ${price_stats['mean']:,.2f}")
                print(f"  Median price: ${price_stats['50%']:,.2f}")
                print(f"  Min price: ${price_stats['min']:,.2f}")
                print(f"  Max price: ${price_stats['max']:,.2f}")
                
                print(f"\n[SUCCESS] Property price data in merged file appears reliable")
    else:
        print(f"[ERROR] Miami-Dade merged data file not found at {merged_file}")
    
    print("\n----------------------------------------\n")
    print("Data verification complete!")

if __name__ == "__main__":
    verify_data_integrity()
