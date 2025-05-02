#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidate duplicate neighborhoods in the merged data file.

This script standardizes neighborhood names and combines duplicate entries
to ensure each neighborhood appears only once in visualizations.
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

# Neighborhood standardization mapping
NEIGHBORHOOD_MAPPING = {
    'downtown': 'downtown',
    'downtown miami': 'downtown',
    'downtown area': 'downtown',
    'downtown core': 'downtown',
    'brickell': 'brickell',
    'brickell area': 'brickell',
    'brickell district': 'brickell',
    'coral gables': 'coral gables',
    'coconut grove': 'coconut grove',
    'edgewater': 'edgewater',
    'midtown': 'midtown',
    'wynwood': 'wynwood',
    'little havana': 'little havana',
    'south beach': 'south beach',
    'design district': 'design district',
    'allapattah': 'allapattah',
    'overtown': 'overtown',
    'little haiti': 'little haiti',
    'upper eastside': 'upper eastside',
    'north beach': 'north beach'
}

def consolidate_neighborhoods():
    """
    Consolidate duplicate neighborhoods in the merged data file
    """
    # Path to merged data file
    merged_data_path = processed_dir / "miami_dade_merged_data.csv"
    
    # Check if file exists
    if not os.path.exists(merged_data_path):
        logger.error(f"Merged data file not found at {merged_data_path}")
        return False
    
    try:
        # Load the file
        logger.info(f"Loading merged data from {merged_data_path}")
        merged_data = pd.read_csv(merged_data_path)
        
        # Create a backup of the merged data
        backup_path = str(merged_data_path) + ".dupes.bak"
        merged_data.to_csv(backup_path, index=False)
        logger.info(f"Created backup at {backup_path}")
        
        # Check for duplicates
        original_count = len(merged_data)
        merged_data['neighborhood_lower'] = merged_data['neighborhood'].str.lower().str.strip()
        duplicate_counts = merged_data['neighborhood_lower'].value_counts()
        duplicates = duplicate_counts[duplicate_counts > 1]
        
        if len(duplicates) == 0:
            logger.info("No duplicate neighborhoods found")
            return True
        
        logger.info(f"Found {len(duplicates)} neighborhoods with duplicates: {', '.join(duplicates.index.tolist())}")
        
        # Standardize neighborhood names
        for idx, row in merged_data.iterrows():
            neighborhood = row['neighborhood_lower']
            if neighborhood in NEIGHBORHOOD_MAPPING:
                merged_data.loc[idx, 'neighborhood_lower'] = NEIGHBORHOOD_MAPPING[neighborhood]
        
        # Recalculate duplicates after standardization
        duplicate_counts = merged_data['neighborhood_lower'].value_counts()
        duplicates = duplicate_counts[duplicate_counts > 1]
        logger.info(f"After standardization, found {len(duplicates)} neighborhoods to consolidate")
        
        # Create a new DataFrame to hold consolidated data
        consolidated_data = []
        
        # Process each unique neighborhood
        for neighborhood in merged_data['neighborhood_lower'].unique():
            neighborhood_rows = merged_data[merged_data['neighborhood_lower'] == neighborhood]
            
            # If only one row, keep it as is
            if len(neighborhood_rows) == 1:
                consolidated_data.append(neighborhood_rows.iloc[0].to_dict())
                continue
            
            # For duplicates, consolidate the data
            logger.info(f"Consolidating {len(neighborhood_rows)} entries for '{neighborhood}'")
            
            # Choose the best display name (most frequent)
            name_counts = neighborhood_rows['neighborhood'].value_counts()
            display_name = name_counts.index[0]
            
            # Calculate aggregated values
            consolidated_row = {
                'neighborhood': display_name,
                'neighborhood_lower': neighborhood,
                'airbnb_count': neighborhood_rows['airbnb_count'].sum(),
                'airbnb_density': neighborhood_rows['airbnb_density'].mean(),
                'median_property_price': neighborhood_rows['median_property_price'].mean(),
                'population': neighborhood_rows['population'].sum(),
                'median_income': neighborhood_rows['median_income'].mean()
            }
            
            # Add any additional columns that exist in the data
            for col in neighborhood_rows.columns:
                if col not in consolidated_row and col != 'neighborhood' and col != 'neighborhood_lower':
                    # Check column type to handle numeric and non-numeric data differently
                    if pd.api.types.is_numeric_dtype(neighborhood_rows[col]):
                        # For numeric columns, take the mean
                        consolidated_row[col] = neighborhood_rows[col].mean()
                    else:
                        # For non-numeric columns, take the most common value
                        value_counts = neighborhood_rows[col].value_counts()
                        if not value_counts.empty:
                            consolidated_row[col] = value_counts.index[0]
                        else:
                            consolidated_row[col] = None  # Default to None if empty
            
            consolidated_data.append(consolidated_row)
        
        # Create a new DataFrame from the consolidated data
        new_df = pd.DataFrame(consolidated_data)
        
        # Remove the temporary column
        if 'neighborhood_lower' in new_df.columns:
            new_df = new_df.drop(columns=['neighborhood_lower'])
        
        # Save the consolidated data
        new_df.to_csv(merged_data_path, index=False)
        
        logger.info(f"Consolidated from {original_count} to {len(new_df)} unique neighborhoods")
        return True
    
    except Exception as e:
        logger.error(f"Error consolidating neighborhoods: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting consolidation of duplicate neighborhoods")
    success = consolidate_neighborhoods()
    
    if success:
        logger.info("Successfully consolidated duplicate neighborhoods")
    else:
        logger.error("Failed to consolidate neighborhoods")
