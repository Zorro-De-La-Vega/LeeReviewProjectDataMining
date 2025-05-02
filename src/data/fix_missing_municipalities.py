"""
Script to fix missing municipalities in the consolidated dataset
by adding real data from the complete Miami-Dade dataset
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

def fix_missing_municipalities():
    """Add missing municipalities to the consolidated dataset using real data"""
    
    # Load the complete Miami-Dade dataset
    complete_file = PROCESSED_DIR / 'airbnb' / 'complete_miami_dade_airbnb_data.csv'
    consolidated_file = CONSOLIDATED_DIR / 'airbnb_comprehensive.csv'
    
    if not os.path.exists(complete_file):
        logger.error(f"Complete dataset not found at {complete_file}")
        return
        
    if not os.path.exists(consolidated_file):
        logger.error(f"Consolidated dataset not found at {consolidated_file}")
        return
        
    # Load datasets
    complete_df = pd.read_csv(complete_file)
    consolidated_df = pd.read_csv(consolidated_file)
    logger.info(f"Loaded complete dataset with {len(complete_df)} records")
    logger.info(f"Loaded consolidated dataset with {len(consolidated_df)} records")
    
    # Extract all municipalities from the complete dataset
    municipalities = complete_df[complete_df['area_type'] == 'Municipality']
    logger.info(f"Found {len(municipalities)} municipalities in complete dataset")
    
    # Extract municipalities from the consolidated dataset
    if 'municipality' in consolidated_df.columns:
        existing_municipalities = [m for m in consolidated_df['municipality'].unique() 
                               if str(m) != 'nan' and str(m) != 'Unknown']
        logger.info(f"Found {len(existing_municipalities)} municipalities in consolidated dataset")
    else:
        existing_municipalities = []
        logger.warning("Municipality column not found in consolidated dataset")
    
    # Identify missing municipalities
    missing_municipalities = [m for m in municipalities['area_name'].tolist() 
                           if m not in existing_municipalities]
    logger.info(f"Identified {len(missing_municipalities)} missing municipalities")
    
    # Create entries for missing municipalities to add to consolidated file
    missing_entries = []
    for _, row in municipalities.iterrows():
        if row['area_name'] not in existing_municipalities:
            # Create entry using real data from complete dataset
            entry = {
                'neighborhood': row['area_name'],
                'airbnb_count': row['airbnb_count'],
                'percentage_of_total': row['percentage_of_total'],
                'zipcode': 'Unknown',  # Could be enriched from other sources
                'municipality': row['area_name'], 
                'area_type': row['development_type'] if pd.notna(row['development_type']) else row['designation']
            }
            missing_entries.append(entry)
            logger.info(f"Created entry for {row['area_name']} with {row['airbnb_count']} Airbnb listings")
    
    # Add missing entries to consolidated dataframe
    if missing_entries:
        missing_df = pd.DataFrame(missing_entries)
        updated_df = pd.concat([consolidated_df, missing_df], ignore_index=True)
        logger.info(f"Added {len(missing_entries)} municipalities to consolidated dataset")
        
        # Save updated dataset
        updated_df.to_csv(consolidated_file, index=False)
        logger.info(f"Saved updated dataset with {len(updated_df)} records to {consolidated_file}")
        
        # Also update the comprehensive neighborhood listings file
        comprehensive_file = PROCESSED_DIR / 'airbnb' / 'comprehensive_neighborhood_listings.csv'
        if os.path.exists(comprehensive_file):
            comprehensive_df = pd.read_csv(comprehensive_file)
            updated_comprehensive_df = pd.concat([comprehensive_df, missing_df], ignore_index=True)
            updated_comprehensive_df.to_csv(comprehensive_file, index=False)
            logger.info(f"Updated comprehensive neighborhood listings file with missing municipalities")
    else:
        logger.info("No new municipalities to add")
    
    # Output the current list of municipalities in the consolidated dataset
    updated_municipalities = pd.read_csv(consolidated_file)['municipality'].unique()
    valid_municipalities = [m for m in updated_municipalities if str(m) != 'nan' and str(m) != 'Unknown']
    logger.info(f"Consolidated dataset now contains {len(valid_municipalities)} municipalities")
    
    # Update the data inventory with the changes
    update_data_inventory(len(missing_entries))

def update_data_inventory(municipalities_added):
    """Update the data inventory document with the fix information"""
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

Fixed missing municipalities in the consolidated datasets:
* Added {municipalities_added} missing municipalities to airbnb_comprehensive.csv
* All data was sourced from real data in complete_miami_dade_airbnb_data.csv
* No synthetic or generated data was used in this process
"""
        
        # Update the inventory file
        with open(inventory_path, 'a') as f:
            f.write(fix_update)
            
        logger.info(f"Updated data inventory at {inventory_path}")
        
    except Exception as e:
        logger.error(f"Error updating data inventory: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting fix for missing municipalities")
    fix_missing_municipalities()
    logger.info("Fix completed")
