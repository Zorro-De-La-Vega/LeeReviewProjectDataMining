#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update property prices in the merged data file using the corrected neighborhood stats.

This script copies the accurate property prices from neighborhood_stats.csv 
to miami_dade_merged_data.csv to ensure visualizations show correct data.
"""

import os
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Find project root (assuming script is in src/data/property)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Go up three levels to project root
processed_dir = project_root / "data" / "processed"

def update_merged_data():
    """
    Update the miami_dade_merged_data.csv file with correct property prices from neighborhood_stats.csv
    """
    # Paths to data files
    neighborhood_stats_path = processed_dir / "neighborhood_stats.csv"
    merged_data_path = processed_dir / "miami_dade_merged_data.csv"
    
    # Check if files exist
    if not os.path.exists(neighborhood_stats_path):
        logger.error(f"Neighborhood stats file not found at {neighborhood_stats_path}")
        return False
    
    if not os.path.exists(merged_data_path):
        logger.error(f"Merged data file not found at {merged_data_path}")
        return False
    
    try:
        # Load the files
        logger.info(f"Loading neighborhood stats from {neighborhood_stats_path}")
        neighborhood_stats = pd.read_csv(neighborhood_stats_path)
        
        logger.info(f"Loading merged data from {merged_data_path}")
        merged_data = pd.read_csv(merged_data_path)
        
        # Create a backup of the merged data
        backup_path = str(merged_data_path) + ".bak"
        merged_data.to_csv(backup_path, index=False)
        logger.info(f"Created backup at {backup_path}")
        
        # Check for placeholder values in merged data
        price_counts = merged_data['median_property_price'].value_counts()
        most_common_price = price_counts.idxmax()
        count_of_most_common = price_counts[most_common_price]
        
        if (count_of_most_common / len(merged_data)) > 0.3:  # If >30% have same price, likely placeholders
            logger.warning(f"Detected potential placeholder values: {count_of_most_common} rows with identical price ${most_common_price:,.0f}")
        
        # Create a mapping of neighborhood names to correct property prices
        neighborhood_stats['neighborhood_lower'] = neighborhood_stats['neighborhood'].str.lower()
        price_mapping = dict(zip(neighborhood_stats['neighborhood_lower'], neighborhood_stats['median_property_price']))
        
        # Normalize merged data neighborhood names
        merged_data['neighborhood_lower'] = merged_data['neighborhood'].str.lower()
        
        # Update merged data property prices
        updated_count = 0
        for idx, row in merged_data.iterrows():
            neighborhood = row['neighborhood_lower']
            if neighborhood in price_mapping:
                merged_data.loc[idx, 'median_property_price'] = price_mapping[neighborhood]
                if 'price' in merged_data.columns:  # Also update 'price' column if it exists
                    merged_data.loc[idx, 'price'] = price_mapping[neighborhood]
                updated_count += 1
        
        # Remove temporary column and save updated data
        merged_data = merged_data.drop(columns=['neighborhood_lower'])
        merged_data.to_csv(merged_data_path, index=False)
        
        # Check if we've improved the data
        updated_price_counts = merged_data['median_property_price'].value_counts()
        updated_most_common = updated_price_counts.idxmax()
        updated_count_of_most_common = updated_price_counts[updated_most_common]
        
        logger.info(f"Updated {updated_count} rows in merged data")
        logger.info(f"Before: {count_of_most_common}/{len(merged_data)} rows with same price (${most_common_price:,.0f})")
        logger.info(f"After: {updated_count_of_most_common}/{len(merged_data)} rows with same price (${updated_most_common:,.0f})")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating merged data: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting update of merged data file with correct property prices")
    success = update_merged_data()
    
    if success:
        logger.info("Successfully updated merged data file with correct property prices")
    else:
        logger.error("Failed to update merged data file")
