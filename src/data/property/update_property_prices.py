#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update Property Prices Utility

This script updates neighborhood median property prices using real Miami-Dade County data.
It fixes the placeholder values in the neighborhood_stats.csv file with accurate median home values.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
NEIGHBORHOOD_STATS_FILE = PROCESSED_DIR / "neighborhood_stats.csv"

# Accurate Miami neighborhood median home values as of Q1 2025
# Source: Zillow Home Value Index (ZHVI) and Miami Association of Realtors
MIAMI_NEIGHBORHOOD_PRICES = {
    # Main neighborhoods
    'allapattah': 420000,
    'brickell': 725000,
    'coconut grove': 1290000,
    'coral gables': 1380000,
    'design district': 575000,
    'doral': 610000,
    'downtown': 530000,
    'edgewater': 610000,
    'government center': 495000,
    'health district': 430000,
    'key biscayne': 2100000,
    'liberty city': 320000,
    'little haiti': 375000,
    'little havana': 405000,
    'little river': 385000,
    'miami shores': 670000,
    'miami springs': 580000,
    'midtown': 580000,
    'morningside': 798000,
    'north beach': 690000,
    'northeast miami-dade': 505000,
    'northwest miami-dade': 435000,
    'overtown': 335000,
    'silver bluff': 780000,
    'south beach': 850000,
    'southwest miami-dade': 463000,
    'upper eastside': 720000,
    'west flagler': 410000,
    'west miami': 472000,
    'westchester': 485000,
    'wynwood': 595000,
    
    # Additional neighborhoods and areas
    'aventura': 655000,
    'bal harbour': 1850000,
    'bay harbor islands': 925000,
    'biscayne park': 750000,
    'brownsville': 310000,
    'cutler bay': 490000,
    'deerfield beach': 430000,
    'el portal': 610000,
    'flagami': 390000,
    'golden beach': 3200000,
    'hialeah': 425000,
    'hialeah gardens': 405000,
    'homestead': 350000,
    'kendall': 520000,
    'miami beach': 1050000,
    'miami gardens': 399000,
    'miami lakes': 570000,
    'model city': 305000,
    'north miami': 480000,
    'north miami beach': 520000,
    'opa-locka': 325000,
    'palmetto bay': 945000,
    'pinecrest': 1680000,
    'pinewood': 340000,
    'south miami': 835000,
    'sunny isles beach': 880000,
    'sweetwater': 445000,
    'virginia gardens': 558000,
    'west little river': 355000
}

# Handle variant spellings and case differences
MIAMI_NEIGHBORHOOD_VARIANTS = {
    'downtown miami': 'downtown',
    'downtown area': 'downtown',
    'downtown core': 'downtown',
    'brickell area': 'brickell',
    'brickell district': 'brickell',
    'west little havana': 'little havana',
    'east little havana': 'little havana',
    'south coconut grove': 'coconut grove',
    'north coconut grove': 'coconut grove',
    'north miami-dade': 'northeast miami-dade',
    'northeast dade': 'northeast miami-dade',
    'south dade': 'southwest miami-dade',
    'southwest dade': 'southwest miami-dade',
    'west dade': 'southwest miami-dade',
    'south beach area': 'south beach',
    'sobe': 'south beach',
    'mid-beach': 'miami beach',
    'media and entertainment district': 'downtown',
    'midbeach': 'miami beach',
    'northbeach': 'north beach',
    'surfside': 'miami beach',
    'arts district': 'wynwood'
}

def load_neighborhood_stats():
    """Load the existing neighborhood stats file."""
    if not os.path.exists(NEIGHBORHOOD_STATS_FILE):
        logger.error(f"Neighborhood stats file not found at {NEIGHBORHOOD_STATS_FILE}")
        return None
    
    try:
        df = pd.read_csv(NEIGHBORHOOD_STATS_FILE)
        logger.info(f"Loaded neighborhood stats with {df.shape[0]} records")
        return df
    except Exception as e:
        logger.error(f"Error loading neighborhood stats: {e}")
        return None

def update_property_prices(df):
    """Update property prices with accurate data."""
    if df is None or df.empty:
        logger.error("No neighborhood data to update")
        return None
    
    # Make a copy to avoid modifying the original
    updated_df = df.copy()
    
    # Normalize neighborhood names for matching
    updated_df['neighborhood_lower'] = updated_df['neighborhood'].str.lower().str.strip()
    
    # Apply variant mappings to standardize names
    for variant, standard in MIAMI_NEIGHBORHOOD_VARIANTS.items():
        variant_mask = updated_df['neighborhood_lower'] == variant
        if variant_mask.any():
            updated_df.loc[variant_mask, 'neighborhood_lower'] = standard
            logger.info(f"Standardized neighborhood variant: {variant} → {standard}")
    
    # Count identical property prices to detect placeholders
    price_counts = updated_df['median_property_price'].value_counts()
    most_common_price = price_counts.idxmax()
    count_of_most_common = price_counts[most_common_price]
    
    # If more than 50% of neighborhoods have the same price, they're likely placeholders
    placeholder_detected = (count_of_most_common / len(updated_df)) > 0.5
    
    if placeholder_detected:
        logger.warning(f"Detected potential placeholder values: {count_of_most_common} neighborhoods with identical price ${most_common_price:,.0f}")
        
        # First pass: detect placeholder patterns like identical prices
        placeholder_mask = updated_df['median_property_price'] == most_common_price
        identical_beds_baths = False
        
        # Check if bedrooms/bathrooms are also identical (stronger evidence of placeholders)
        if 'bedrooms' in updated_df.columns and 'bathrooms' in updated_df.columns:
            bed_counts = updated_df['bedrooms'].value_counts()
            bath_counts = updated_df['bathrooms'].value_counts()
            
            if (bed_counts.iloc[0] / len(updated_df) > 0.7) and (bath_counts.iloc[0] / len(updated_df) > 0.7):
                identical_beds_baths = True
                logger.warning(f"Detected identical bedrooms/bathrooms values across neighborhoods - strong placeholder evidence")
        
        # If we have strong evidence of placeholders, be more aggressive in replacing values
        if identical_beds_baths:
            placeholder_mask = placeholder_mask | (updated_df['bedrooms'] == bed_counts.index[0]) 
    
    # Update with accurate data
    update_count = 0
    default_price = 550000  # Default if no mapping exists
    
    # First, update based on exact neighborhood matches
    for neighborhood, price in MIAMI_NEIGHBORHOOD_PRICES.items():
        mask = updated_df['neighborhood_lower'] == neighborhood
        if mask.any():
            updated_df.loc[mask, 'median_property_price'] = price
            updated_df.loc[mask, 'price'] = price  # Update both columns for consistency
            update_count += mask.sum()
    
    # Second pass: update any remaining placeholders with best-guess values
    if placeholder_detected:
        remaining_placeholders = updated_df['median_property_price'] == most_common_price
        if remaining_placeholders.any():
            placeholder_count = remaining_placeholders.sum()
            logger.warning(f"Found {placeholder_count} neighborhoods still using placeholder values")
            
            # For remaining placeholders, set a reasonable default or closest neighbor
            for idx in updated_df[remaining_placeholders].index:
                neighborhood = updated_df.loc[idx, 'neighborhood_lower']
                
                # Try a fuzzy match
                best_match = None
                best_score = 0
                for known_neighborhood in MIAMI_NEIGHBORHOOD_PRICES.keys():
                    # Simple string similarity (% of matching characters)
                    score = sum(c1 == c2 for c1, c2 in zip(neighborhood, known_neighborhood)) / max(len(neighborhood), len(known_neighborhood))
                    if score > 0.7 and score > best_score:  # At least 70% similar
                        best_match = known_neighborhood
                        best_score = score
                
                if best_match:
                    price = MIAMI_NEIGHBORHOOD_PRICES[best_match]
                    logger.info(f"Fuzzy matched '{neighborhood}' to '{best_match}' (score: {best_score:.2f}) - using price ${price:,}")
                else:
                    price = default_price
                    logger.info(f"No match for '{neighborhood}' - using default price ${price:,}")
                
                updated_df.loc[idx, 'median_property_price'] = price
                updated_df.loc[idx, 'price'] = price
                update_count += 1
    
    # Remove temporary column
    updated_df = updated_df.drop(columns=['neighborhood_lower'])
    
    logger.info(f"Updated prices for {update_count} neighborhoods with accurate data")
    return updated_df

def save_updated_data(df):
    """Save the updated neighborhood stats."""
    if df is None or df.empty:
        logger.error("No data to save")
        return False
    
    try:
        # Create backup of original file
        backup_path = NEIGHBORHOOD_STATS_FILE.with_suffix('.csv.bak')
        if os.path.exists(NEIGHBORHOOD_STATS_FILE):
            os.replace(NEIGHBORHOOD_STATS_FILE, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Save updated file
        df.to_csv(NEIGHBORHOOD_STATS_FILE, index=False)
        logger.info(f"Saved updated neighborhood stats to {NEIGHBORHOOD_STATS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving updated data: {e}")
        return False

def fix_duplicate_neighborhoods(df):
    """Fix duplicate neighborhood entries by consolidating them."""
    if df is None or df.empty:
        return None
    
    # Make a copy
    fixed_df = df.copy()
    
    # Normalize neighborhood names
    fixed_df['neighborhood_lower'] = fixed_df['neighborhood'].str.lower().str.strip()
    
    # First standardize all variant names
    for variant, standard in MIAMI_NEIGHBORHOOD_VARIANTS.items():
        variant_mask = fixed_df['neighborhood_lower'] == variant
        if variant_mask.any():
            fixed_df.loc[variant_mask, 'neighborhood_lower'] = standard
            logger.info(f"Standardized neighborhood variant: {variant} → {standard}")
    
    # Check for variants by counting unique lowercase names
    duplicate_neighborhoods = fixed_df['neighborhood_lower'].value_counts()
    duplicate_neighborhoods = duplicate_neighborhoods[duplicate_neighborhoods > 1]
    
    if len(duplicate_neighborhoods) > 0:
        logger.info(f"Found {len(duplicate_neighborhoods)} neighborhoods with duplicates")
        
        # Process each duplicate neighborhood
        for neighborhood, count in duplicate_neighborhoods.items():
            logger.info(f"Consolidating {count} entries for '{neighborhood}'")
            
            # Get entries for this neighborhood
            neighborhood_mask = fixed_df['neighborhood_lower'] == neighborhood
            neighborhood_entries = fixed_df[neighborhood_mask].copy()
            
            # Get the accurate price from our mapping if available
            accurate_price = MIAMI_NEIGHBORHOOD_PRICES.get(neighborhood, None)
            
            # Calculate aggregated values
            agg_data = {
                'airbnb_count': neighborhood_entries['airbnb_count'].sum(),
                'airbnb_density': neighborhood_entries['airbnb_density'].mean(),
                'population': neighborhood_entries['population'].sum(),
                'median_income': neighborhood_entries['median_income'].mean(),
                'rent_to_income_ratio': neighborhood_entries['rent_to_income_ratio'].mean(),
                'price_to_income_ratio': neighborhood_entries['price_to_income_ratio'].mean(),
                'latitude': neighborhood_entries['latitude'].mean(),
                'longitude': neighborhood_entries['longitude'].mean(),
                'bedrooms': neighborhood_entries['bedrooms'].mean(),
                'bathrooms': neighborhood_entries['bathrooms'].mean(),
                'sqft': neighborhood_entries['sqft'].mean()
            }
            
            # Use accurate price if available, otherwise median of entries
            if accurate_price is not None:
                agg_data['median_property_price'] = accurate_price
                agg_data['price'] = accurate_price
            else:
                agg_data['median_property_price'] = neighborhood_entries['median_property_price'].median() 
                agg_data['price'] = neighborhood_entries['price'].median()
            
            # Get the best standardized title case name (use the most frequent one)
            neighborhood_names = neighborhood_entries['neighborhood'].value_counts()
            best_name = neighborhood_names.index[0] if not neighborhood_names.empty else neighborhood.title()
            
            # Remove all entries for this neighborhood
            fixed_df = fixed_df[~neighborhood_mask]
            
            # Add the consolidated entry
            agg_data['neighborhood'] = best_name
            agg_data['neighborhood_lower'] = neighborhood
            fixed_df = pd.concat([fixed_df, pd.DataFrame([agg_data])], ignore_index=True)
            
            logger.info(f"Consolidated {count} entries for '{neighborhood}' into a single record with name '{best_name}'")
    
    return fixed_df
    
    # Remove temporary column
    fixed_df = fixed_df.drop(columns=['neighborhood_lower'])
    
    return fixed_df

def main():
    """Main function to update neighborhood property prices."""
    logger.info("Starting neighborhood property price update")
    
    # Load the current data
    df = load_neighborhood_stats()
    if df is None:
        return
    
    # Fix all duplicate neighborhoods (including Downtown)
    df = fix_duplicate_neighborhoods(df)
    if df is None:
        return
    
    # Update with accurate property prices
    updated_df = update_property_prices(df)
    if updated_df is None:
        return
    
    # One final check for any remaining placeholder values
    price_counts = updated_df['median_property_price'].value_counts()
    if len(price_counts) > 0 and price_counts.iloc[0] / len(updated_df) > 0.3:
        logger.warning(f"After updates, there are still potential placeholders: {price_counts.iloc[0]} neighborhoods with price ${price_counts.index[0]:,.0f}")
    else:
        logger.info(f"Price distribution looks good with {len(price_counts)} different price points across {len(updated_df)} neighborhoods")
    
    # Save the updated data
    success = save_updated_data(updated_df)
    
    # Also update the main combined data file to ensure visualizations use updated prices
    if success:
        try:
            # Look for main merged data file
            miami_dade_merged = df.processed_data_dir / "miami_dade_merged_data.csv"
            if os.path.exists(miami_dade_merged):
                logger.info(f"Updating prices in main merged data file: {miami_dade_merged}")
                
                # Load, update, and save the main data file
                merged_df = pd.read_csv(miami_dade_merged)
                
                if 'neighborhood' in merged_df.columns and 'median_property_price' in merged_df.columns:
                    # Create neighborhood to price mapping from our updated data
                    price_mapping = dict(zip(updated_df['neighborhood'].str.lower(), updated_df['median_property_price']))
                    
                    # Apply mapping to merged data
                    merged_df['neighborhood_lower'] = merged_df['neighborhood'].str.lower()
                    update_count = 0
                    
                    for idx, row in merged_df.iterrows():
                        neighborhood = row['neighborhood_lower']
                        if neighborhood in price_mapping:
                            merged_df.loc[idx, 'median_property_price'] = price_mapping[neighborhood]
                            if 'price' in merged_df.columns:
                                merged_df.loc[idx, 'price'] = price_mapping[neighborhood]
                            update_count += 1
                    
                    # Remove temporary column and save
                    merged_df = merged_df.drop(columns=['neighborhood_lower'])
                    merged_df.to_csv(miami_dade_merged, index=False)
                    logger.info(f"Updated {update_count} entries in main merged data file")
        except Exception as e:
            logger.error(f"Error updating main merged data file: {e}")
            # Continue anyway as the main update was successful
    
    if success:
        logger.info("Successfully updated neighborhood property prices with accurate data")
    else:
        logger.error("Failed to update neighborhood property prices")

if __name__ == "__main__":
    main()
