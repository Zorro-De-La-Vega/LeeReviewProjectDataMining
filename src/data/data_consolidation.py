"""
Data Consolidation Script for Miami-Dade Housing Impact Project
This script consolidates redundant data files while preserving all real-world data
"""

import pandas as pd
import os
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
ARCHIVE_DIR = DATA_DIR / 'archive'
CONSOLIDATED_DIR = DATA_DIR / 'consolidated'

# Ensure directories exist
CONSOLIDATED_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)
(ARCHIVE_DIR / 'redundant').mkdir(exist_ok=True)

def consolidate_airbnb_data():
    """Consolidate redundant Airbnb data files"""
    logger.info("Consolidating Airbnb data files")
    
    # Files to consolidate
    airbnb_dir = PROCESSED_DIR / 'airbnb'
    files_to_process = {
        'comprehensive': airbnb_dir / 'comprehensive_neighborhood_listings.csv',
        'by_zipcode': airbnb_dir / 'airbnb_by_zipcode.csv',
        'cleaned': airbnb_dir / 'airbnb_cleaned.csv',
        'summary': airbnb_dir / 'summary_airbnb.csv',
        'by_neighborhood': airbnb_dir / 'summary_airbnb_by_neighbourhood.csv',
    }
    
    # Load main comprehensive file
    if files_to_process['comprehensive'].exists():
        logger.info(f"Loading main comprehensive file: {files_to_process['comprehensive']}")
        comprehensive_df = pd.read_csv(files_to_process['comprehensive'])
    else:
        logger.warning("Comprehensive listings file not found. Creating new one.")
        comprehensive_df = pd.DataFrame()
    
    # Process each supplementary file and merge with comprehensive data
    for name, file_path in files_to_process.items():
        if name == 'comprehensive' or not file_path.exists():
            continue
            
        logger.info(f"Processing {name} from {file_path}")
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Different merging strategies based on file type
            if name == 'by_zipcode':
                # Add zipcode data if not already present
                if 'zipcode' not in comprehensive_df.columns and 'zipcode' in df.columns:
                    # Create a mapping dictionary from neighborhood to zipcode
                    zipcode_mapping = {}
                    for _, row in df.iterrows():
                        if 'neighborhood' in df.columns and 'zipcode' in df.columns:
                            zipcode_mapping[row['neighborhood']] = row['zipcode']
                    
                    # Apply mapping if neighborhoods match
                    if 'neighborhood' in comprehensive_df.columns and zipcode_mapping:
                        comprehensive_df['zipcode'] = comprehensive_df['neighborhood'].map(zipcode_mapping)
                        logger.info("Added zipcode data to comprehensive listings")
            
            elif name == 'cleaned':
                # If comprehensive is empty, use cleaned as base
                if comprehensive_df.empty and not df.empty:
                    comprehensive_df = df.copy()
                    logger.info("Used cleaned data as base for comprehensive file")
                else:
                    # Identify new records in cleaned not in comprehensive
                    if 'id' in df.columns and 'id' in comprehensive_df.columns:
                        new_listings = df[~df['id'].isin(comprehensive_df['id'])]
                        if not new_listings.empty:
                            comprehensive_df = pd.concat([comprehensive_df, new_listings], ignore_index=True)
                            logger.info(f"Added {len(new_listings)} new listings from cleaned data")
            
            elif name == 'summary' or name == 'by_neighborhood':
                # Add summary statistics if not already in comprehensive
                if 'neighborhood' in df.columns:
                    # Identify columns to add (those not already in comprehensive)
                    cols_to_add = [col for col in df.columns 
                                  if col not in comprehensive_df.columns and col != 'neighborhood']
                    
                    if cols_to_add:
                        # Create a mapping of statistics by neighborhood
                        stats_mapping = {}
                        for _, row in df.iterrows():
                            if pd.notna(row['neighborhood']):
                                stats_mapping[row['neighborhood']] = {
                                    col: row[col] for col in cols_to_add if col in row
                                }
                        
                        # Add new columns to comprehensive
                        for col in cols_to_add:
                            comprehensive_df[col] = None
                        
                        # Apply mapping where neighborhoods match
                        if 'neighborhood' in comprehensive_df.columns:
                            for idx, row in comprehensive_df.iterrows():
                                if pd.notna(row['neighborhood']) and row['neighborhood'] in stats_mapping:
                                    for col, value in stats_mapping[row['neighborhood']].items():
                                        comprehensive_df.at[idx, col] = value
                            
                            logger.info(f"Added summary statistics: {', '.join(cols_to_add)}")
            
            # Move processed file to archive
            archive_path = ARCHIVE_DIR / 'redundant' / file_path.name
            logger.info(f"Moving {file_path.name} to archive: {archive_path}")
            if file_path.exists():
                os.rename(file_path, archive_path)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Save consolidated data
    output_file = CONSOLIDATED_DIR / 'airbnb_comprehensive.csv'
    if not comprehensive_df.empty:
        comprehensive_df.to_csv(output_file, index=False)
        logger.info(f"Saved consolidated Airbnb data to {output_file}")
        
        # Create a symbolic link in the original location pointing to consolidated file
        symlink_path = airbnb_dir / 'comprehensive_neighborhood_listings.csv'
        if symlink_path.exists():
            os.remove(symlink_path)
        try:
            os.symlink(output_file, symlink_path)
            logger.info(f"Created symbolic link at {symlink_path}")
        except Exception as e:
            logger.error(f"Could not create symbolic link: {str(e)}")
            # Copy the file instead
            comprehensive_df.to_csv(symlink_path, index=False)
            logger.info(f"Created copy at original location {symlink_path}")
    else:
        logger.warning("No consolidated data to save")

def consolidate_property_data():
    """Consolidate property data files"""
    logger.info("Consolidating property data files")
    
    # Files to consolidate
    property_dir = PROCESSED_DIR / 'property'
    cleaned_file = property_dir / 'cleaned_property.csv'
    summary_file = property_dir / 'summary_property.csv'
    
    if not cleaned_file.exists():
        logger.error(f"Primary property file not found: {cleaned_file}")
        return
    
    # Load the cleaned property data
    cleaned_df = pd.read_csv(cleaned_file)
    logger.info(f"Loaded cleaned property data: {len(cleaned_df)} records")
    
    # Add summary statistics if available
    if summary_file.exists():
        try:
            summary_df = pd.read_csv(summary_file)
            logger.info(f"Loaded property summary: {len(summary_df)} records")
            
            # Check if zipcode exists in both datasets
            if 'zipcode' in summary_df.columns and 'parcelno' in cleaned_df.columns:
                # Extract zipcode from parcel number if possible
                if 'zipcode' not in cleaned_df.columns:
                    # Try to extract zipcode from address or create mapping
                    zipcode_mapping = {}
                    for _, row in summary_df.iterrows():
                        zipcode_mapping[row['zipcode']] = row
                    
                    # Add summary columns to cleaned data
                    for col in summary_df.columns:
                        if col != 'zipcode' and col not in cleaned_df.columns:
                            cleaned_df[f'summary_{col}'] = None
                    
                    # Add a new summary section to the consolidated file
                    summary_section = summary_df.copy()
                    summary_section.columns = ['zipcode'] + [f'summary_{col}' for col in summary_df.columns if col != 'zipcode']
                    
                    # Save the summary section separately
                    summary_section.to_csv(CONSOLIDATED_DIR / 'property_summary_by_zipcode.csv', index=False)
                    logger.info(f"Saved property summary section to separate file")
            
            # Move summary file to archive
            archive_path = ARCHIVE_DIR / 'redundant' / summary_file.name
            logger.info(f"Moving {summary_file.name} to archive: {archive_path}")
            if summary_file.exists():
                os.rename(summary_file, archive_path)
                
        except Exception as e:
            logger.error(f"Error processing property summary: {str(e)}")
    
    # Save consolidated property data
    output_file = CONSOLIDATED_DIR / 'property_comprehensive.csv'
    cleaned_df.to_csv(output_file, index=False)
    logger.info(f"Saved consolidated property data to {output_file}")
    
    # Create a symbolic link in the original location pointing to consolidated file
    symlink_path = property_dir / 'cleaned_property.csv'
    if symlink_path.exists():
        os.remove(symlink_path)
    try:
        os.symlink(output_file, symlink_path)
        logger.info(f"Created symbolic link at {symlink_path}")
    except Exception as e:
        logger.error(f"Could not create symbolic link: {str(e)}")
        # Copy the file instead
        cleaned_df.to_csv(symlink_path, index=False)
        logger.info(f"Created copy at original location {symlink_path}")

def consolidate_merged_data():
    """Consolidate merged data files"""
    logger.info("Consolidating merged data files")
    
    # Files to consolidate
    merged_dir = PROCESSED_DIR / 'merged'
    by_zip_file = merged_dir / 'merged_by_zip.csv'
    zip_summary_file = merged_dir / 'zipcode_summary.csv'
    
    consolidated_df = None
    
    # Load and process each file
    for file_path in [by_zip_file, zip_summary_file]:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {file_path.name}: {len(df)} records")
            
            if consolidated_df is None:
                consolidated_df = df.copy()
            else:
                # Identify columns to add (those not already in consolidated)
                common_cols = set(consolidated_df.columns).intersection(set(df.columns))
                new_cols = [col for col in df.columns if col not in consolidated_df.columns]
                
                if 'zipcode' in common_cols:
                    # Merge on zipcode
                    merged = pd.merge(consolidated_df, df[['zipcode'] + new_cols], 
                                      on='zipcode', how='left')
                    consolidated_df = merged
                    logger.info(f"Added {len(new_cols)} columns from {file_path.name}")
                else:
                    logger.warning(f"No common key for merging in {file_path.name}")
            
            # Move processed file to archive
            archive_path = ARCHIVE_DIR / 'redundant' / file_path.name
            logger.info(f"Moving {file_path.name} to archive: {archive_path}")
            if file_path.exists():
                os.rename(file_path, archive_path)
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Save consolidated data
    if consolidated_df is not None:
        output_file = CONSOLIDATED_DIR / 'zipcode_analysis.csv'
        consolidated_df.to_csv(output_file, index=False)
        logger.info(f"Saved consolidated zipcode data to {output_file}")
        
        # Create a symbolic link in the original location pointing to consolidated file
        symlink_path = merged_dir / 'zipcode_analysis.csv'
        if symlink_path.exists():
            os.remove(symlink_path)
        try:
            os.symlink(output_file, symlink_path)
            logger.info(f"Created symbolic link at {symlink_path}")
        except Exception as e:
            logger.error(f"Could not create symbolic link: {str(e)}")
            # Copy the file instead
            consolidated_df.to_csv(symlink_path, index=False)
            logger.info(f"Created copy at original location {symlink_path}")
    else:
        logger.warning("No consolidated zipcode data to save")

def update_data_inventory():
    """Update the data inventory document with results of consolidation"""
    logger.info("Updating data inventory document")
    
    inventory_path = PROJECT_ROOT / 'docs' / 'DATA_INVENTORY.md'
    if not inventory_path.exists():
        logger.warning("Data inventory not found")
        return
        
    try:
        # Count files before and after consolidation
        processed_files_before = sum(1 for _ in Path(PROCESSED_DIR).glob('**/*.csv'))
        consolidated_files = sum(1 for _ in Path(CONSOLIDATED_DIR).glob('*.csv'))
        archived_files = sum(1 for _ in Path(ARCHIVE_DIR).glob('**/*.csv'))
        
        # Read existing inventory
        with open(inventory_path, 'r') as f:
            inventory_content = f.read()
            
        # Add consolidation section
        consolidation_update = f"""
## Consolidation Results

Data consolidation performed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

* Files before consolidation: {processed_files_before}
* Files consolidated: {consolidated_files}
* Files archived: {archived_files}

### Consolidated Files

| Original Files | Consolidated Into |
|----------------|------------------|
| airbnb_by_zipcode.csv, airbnb_cleaned.csv, summary_airbnb.csv, summary_airbnb_by_neighbourhood.csv | airbnb_comprehensive.csv |
| cleaned_property.csv, summary_property.csv | property_comprehensive.csv |
| merged_by_zip.csv, zipcode_summary.csv | zipcode_analysis.csv |

All redundant files have been archived to `data/archive/redundant/` for reference.
Synthetic data has been archived to `data/archive/synthetic/`.
"""
        
        # Update the inventory file
        with open(inventory_path, 'a') as f:
            f.write(consolidation_update)
            
        logger.info(f"Updated data inventory at {inventory_path}")
        
    except Exception as e:
        logger.error(f"Error updating data inventory: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting data consolidation process")
    
    # Run consolidation functions
    consolidate_airbnb_data()
    consolidate_property_data()
    consolidate_merged_data()
    
    # Update inventory
    update_data_inventory()
    
    logger.info("Data consolidation completed")
