"""
Script to fix the Airbnb count discrepancies between consolidated and original data
"""

import pandas as pd
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
CONSOLIDATED_DIR = PROJECT_ROOT / 'data' / 'consolidated'

def fix_airbnb_count_discrepancies():
    """Fix the Airbnb count discrepancies between consolidated and original data"""
    
    # Load the datasets
    consolidated_file = CONSOLIDATED_DIR / 'airbnb_comprehensive.csv'
    complete_file = PROCESSED_DIR / 'airbnb' / 'complete_miami_dade_airbnb_data.csv'
    
    if not os.path.exists(consolidated_file) or not os.path.exists(complete_file):
        logger.error("Required files not found")
        return
    
    consolidated_df = pd.read_csv(consolidated_file)
    complete_df = pd.read_csv(complete_file)
    
    logger.info(f"Loaded consolidated file with {len(consolidated_df)} records")
    logger.info(f"Loaded complete Miami-Dade data with {len(complete_df)} records")
    
    # Create mapping of municipality name to correct Airbnb count
    correct_counts = {}
    for _, row in complete_df.iterrows():
        correct_counts[row['area_name']] = {
            'airbnb_count': row['airbnb_count'],
            'percentage_of_total': row['percentage_of_total']
        }
    
    # Update the consolidated dataframe
    count_updates = 0
    for idx, row in consolidated_df.iterrows():
        municipality = row['municipality']
        if municipality in correct_counts:
            old_count = consolidated_df.at[idx, 'airbnb_count']
            correct_count = correct_counts[municipality]['airbnb_count']
            
            if old_count != correct_count:
                consolidated_df.at[idx, 'airbnb_count'] = correct_count
                consolidated_df.at[idx, 'percentage_of_total'] = correct_counts[municipality]['percentage_of_total']
                count_updates += 1
                logger.info(f"Updated {municipality}: {old_count} → {correct_count}")
    
    logger.info(f"Updated {count_updates} Airbnb count discrepancies")
    
    # Handle neighborhoods that are part of municipalities (like Miami neighborhoods)
    # For neighborhoods, keep their specific counts which were scraped at neighborhood level
    
    # Save the updated consolidated dataframe
    consolidated_df.to_csv(consolidated_file, index=False)
    logger.info(f"Saved corrected data to {consolidated_file}")
    
    # Also update comprehensive neighborhood listings
    comprehensive_file = PROCESSED_DIR / 'airbnb' / 'comprehensive_neighborhood_listings.csv'
    if os.path.exists(comprehensive_file):
        comprehensive_df = pd.read_csv(comprehensive_file)
        
        # Update municipality counts in the comprehensive file
        comp_updates = 0
        for idx, row in comprehensive_df.iterrows():
            municipality = row['municipality'] if 'municipality' in row and pd.notna(row['municipality']) else None
            # Only update if the row is an actual municipality (not a neighborhood)
            if municipality in correct_counts and row['neighborhood'] == municipality:
                old_count = comprehensive_df.at[idx, 'airbnb_count']
                correct_count = correct_counts[municipality]['airbnb_count']
                
                if old_count != correct_count:
                    comprehensive_df.at[idx, 'airbnb_count'] = correct_count
                    comprehensive_df.at[idx, 'percentage_of_total'] = correct_counts[municipality]['percentage_of_total']
                    comp_updates += 1
        
        logger.info(f"Updated {comp_updates} Airbnb count discrepancies in comprehensive file")
        comprehensive_df.to_csv(comprehensive_file, index=False)
        logger.info(f"Saved corrected comprehensive data to {comprehensive_file}")
    
    # Update the data inventory
    update_data_inventory(count_updates)
    
    # Return stats about the corrections
    return count_updates

def update_data_inventory(count_updates):
    """Update the data inventory document with fix information"""
    inventory_path = PROJECT_ROOT / 'docs' / 'DATA_INVENTORY.md'
    if not inventory_path.exists():
        logger.warning("Data inventory not found")
        return
        
    try:
        # Read existing inventory
        with open(inventory_path, 'r') as f:
            inventory_content = f.read()
            
        # Add fix information
        fix_update = f"""
## Data Fix - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Fixed Airbnb count discrepancies in the consolidated datasets:
* Corrected {count_updates} Airbnb count mismatches between consolidated and original data
* All counts now match the authoritative source (complete_miami_dade_airbnb_data.csv)
* This maintains our commitment to using only accurate real-world data
"""
        
        # Update the inventory file
        with open(inventory_path, 'a') as f:
            f.write(fix_update)
            
        logger.info(f"Updated data inventory at {inventory_path}")
        
    except Exception as e:
        logger.error(f"Error updating data inventory: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting to fix Airbnb count discrepancies")
    count_updates = fix_airbnb_count_discrepancies()
    logger.info(f"Fixed {count_updates} Airbnb count discrepancies. Data now accurately reflects original source.")
