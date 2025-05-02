"""
Data Loading Utility Module

This module provides standardized functions to load all datasets used in the
Miami Housing Impact Hub application with proper caching for performance.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define path variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


@st.cache_data
def load_airbnb_data(filtered=False, **filter_args):
    """
    Load processed Airbnb data with caching.
    
    Args:
        filtered (bool): Whether to apply filtering
        **filter_args: Filtering arguments
        
    Returns:
        pd.DataFrame: Airbnb data
    """
    try:
        # Try to load from processed directory first
        airbnb_dir = PROCESSED_DATA_DIR / "airbnb"
        airbnb_file = airbnb_dir / "airbnb_listings_processed.csv"
        
        if not airbnb_file.exists():
            # Try alternative filenames
            alternatives = [
                airbnb_dir / "airbnb_by_zipcode.csv",
                airbnb_dir / "listings.csv",
                airbnb_dir / "airbnb_listings.csv"
            ]
            
            for alt_file in alternatives:
                if alt_file.exists():
                    airbnb_file = alt_file
                    break
            else:
                logger.warning("No Airbnb data file found in processed directory")
                return None
        
        # Load the data
        airbnb_data = pd.read_csv(airbnb_file)
        logger.info(f"Loaded Airbnb data with {airbnb_data.shape[0]} records from {airbnb_file}")
        
        # Apply filters if requested
        if filtered and airbnb_data is not None:
            # Example: filter by min_listings if provided
            if 'min_listings' in filter_args and 'listing_count' in airbnb_data.columns:
                airbnb_data = airbnb_data[airbnb_data['listing_count'] >= filter_args['min_listings']]
            
            # Add more filters as needed
            
        return airbnb_data
    
    except Exception as e:
        logger.error(f"Error loading Airbnb data: {e}", exc_info=True)
        return None


@st.cache_data
def load_miami_housing_data(filtered=False, **filter_args):
    """
    Load processed Miami housing data with caching.
    
    Args:
        filtered (bool): Whether to apply filtering
        **filter_args: Filtering arguments
        
    Returns:
        pd.DataFrame: Miami housing data
    """
    try:
        # Try to load from processed directory first
        housing_dir = PROCESSED_DATA_DIR / "property"
        housing_file = housing_dir / "miami_housing_processed.csv"
        
        if not housing_file.exists():
            # Try alternative filenames
            alternatives = [
                housing_dir / "housing_data.csv",
                housing_dir / "miami_property_data.csv",
                PROCESSED_DATA_DIR / "miami_dade_merged_data.csv"
            ]
            
            for alt_file in alternatives:
                if alt_file.exists():
                    housing_file = alt_file
                    break
            else:
                logger.warning("No housing data file found in processed directory")
                return None
        
        # Load the data
        housing_data = pd.read_csv(housing_file)
        logger.info(f"Loaded housing data with {housing_data.shape[0]} records from {housing_file}")
        
        # Apply filters if requested
        if filtered and housing_data is not None:
            # Example: filter by price range if provided
            if 'min_price' in filter_args and 'median_property_price' in housing_data.columns:
                housing_data = housing_data[housing_data['median_property_price'] >= filter_args['min_price']]
            
            if 'max_price' in filter_args and 'median_property_price' in housing_data.columns:
                housing_data = housing_data[housing_data['median_property_price'] <= filter_args['max_price']]
            
            # Add more filters as needed
            
        return housing_data
    
    except Exception as e:
        logger.error(f"Error loading housing data: {e}", exc_info=True)
        return None


@st.cache_data
def load_census_data(filtered=False, **filter_args):
    """
    Load processed census data with caching.
    
    Args:
        filtered (bool): Whether to apply filtering
        **filter_args: Filtering arguments
        
    Returns:
        pd.DataFrame: Census data
    """
    try:
        # Try to load from processed directory first
        census_dir = PROCESSED_DATA_DIR / "census"
        census_file = census_dir / "cleaned_acs5_miamidade_tracts.csv"
        
        if not census_file.exists():
            # Try alternative filenames
            alternatives = [
                census_dir / "acs5_miamidade_processed.csv",
                census_dir / "census_data.csv"
            ]
            
            for alt_file in alternatives:
                if alt_file.exists():
                    census_file = alt_file
                    break
            else:
                logger.warning("No census data file found in processed directory")
                return None
        
        # Load the data
        census_data = pd.read_csv(census_file)
        logger.info(f"Loaded census data with {census_data.shape[0]} records from {census_file}")
        
        # Apply filters if requested
        if filtered and census_data is not None:
            # Example: filter by income range if provided
            if 'min_income' in filter_args and 'median_household_income' in census_data.columns:
                census_data = census_data[census_data['median_household_income'] >= filter_args['min_income']]
            
            if 'max_income' in filter_args and 'median_household_income' in census_data.columns:
                census_data = census_data[census_data['median_household_income'] <= filter_args['max_income']]
            
            # Add more filters as needed
            
        return census_data
    
    except Exception as e:
        logger.error(f"Error loading census data: {e}", exc_info=True)
        return None


@st.cache_data
def load_combined_data(filtered=False, **filter_args):
    """
    Load merged dataset combining Airbnb, housing, and census data.
    
    Args:
        filtered (bool): Whether to apply filtering
        **filter_args: Filtering arguments
        
    Returns:
        pd.DataFrame: Combined data
    """
    try:
        # Try to load from processed directory first
        combined_file = PROCESSED_DATA_DIR / "miami_dade_merged_data.csv"
        
        if not combined_file.exists():
            # Try alternative filenames
            alternatives = [
                PROCESSED_DATA_DIR / "combined" / "airbnb_census_by_zipcode.csv",
                PROCESSED_DATA_DIR / "merged" / "combined_data.csv"
            ]
            
            for alt_file in alternatives:
                if alt_file.exists():
                    combined_file = alt_file
                    break
            else:
                logger.warning("No combined data file found in processed directory")
                return None
        
        # Load the data
        combined_data = pd.read_csv(combined_file)
        logger.info(f"Loaded combined data with {combined_data.shape[0]} records from {combined_file}")
        
        # Apply filters if requested
        if filtered and combined_data is not None:
            # Example: filter by density level if provided
            if 'density_levels' in filter_args and 'airbnb_density_level' in combined_data.columns:
                combined_data = combined_data[combined_data['airbnb_density_level'].isin(filter_args['density_levels'])]
            
            # Add more filters as needed
            
        return combined_data
    
    except Exception as e:
        logger.error(f"Error loading combined data: {e}", exc_info=True)
        return None


@st.cache_data
def load_neighborhood_data():
    """
    Load neighborhood/zipcode reference data.
    
    Returns:
        pd.DataFrame: Neighborhood reference data
    """
    try:
        # Check if we have a neighborhood reference file
        neighborhood_file = PROCESSED_DATA_DIR / "summary" / "neighborhood_reference.csv"
        
        if not neighborhood_file.exists():
            # Try alternative filenames
            alternatives = [
                PROCESSED_DATA_DIR / "summary" / "zipcode_reference.csv",
                PROCESSED_DATA_DIR / "summary" / "miami_neighborhoods.csv"
            ]
            
            for alt_file in alternatives:
                if alt_file.exists():
                    neighborhood_file = alt_file
                    break
            else:
                # Create a basic reference from other datasets
                return _create_neighborhood_reference()
        
        # Load the data
        neighborhood_data = pd.read_csv(neighborhood_file)
        logger.info(f"Loaded neighborhood reference with {neighborhood_data.shape[0]} records from {neighborhood_file}")
        
        return neighborhood_data
    
    except Exception as e:
        logger.error(f"Error loading neighborhood data: {e}", exc_info=True)
        return None


@st.cache_data
def load_zillow_data():
    """
    Load Zillow housing data (e.g., ZORI).
    
    Returns:
        pd.DataFrame: Zillow data
    """
    try:
        # Try to load from processed directory first
        zillow_dir = PROCESSED_DATA_DIR / "rental"
        zillow_file = zillow_dir / "zori_data.csv"
        
        if not zillow_file.exists():
            # Try alternative filenames
            alternatives = [
                zillow_dir / "zillow_rent_index.csv",
                zillow_dir / "zori_miami.csv"
            ]
            
            for alt_file in alternatives:
                if alt_file.exists():
                    zillow_file = alt_file
                    break
            else:
                logger.warning("No Zillow data file found in processed directory")
                return None
        
        # Load the data
        zillow_data = pd.read_csv(zillow_file)
        logger.info(f"Loaded Zillow data with {zillow_data.shape[0]} records from {zillow_file}")
        
        # Convert date column if present
        if 'Date' in zillow_data.columns:
            zillow_data['Date'] = pd.to_datetime(zillow_data['Date'])
        
        return zillow_data
    
    except Exception as e:
        logger.error(f"Error loading Zillow data: {e}", exc_info=True)
        return None


def _create_neighborhood_reference():
    """
    Create a basic neighborhood reference from other datasets if none exists.
    
    Returns:
        pd.DataFrame: Basic neighborhood reference
    """
    # Try to extract from other datasets
    reference_data = None
    
    # First try from combined data
    combined_data = load_combined_data(filtered=False)
    if combined_data is not None:
        # Look for zip code and neighborhood columns
        zip_cols = [col for col in combined_data.columns if 'zip' in col.lower()]
        neighborhood_cols = [col for col in combined_data.columns if any(term in col.lower() for term in ['neighborhood', 'area', 'community'])]
        
        if zip_cols and neighborhood_cols:
            # Create reference from available columns
            reference_data = combined_data[[zip_cols[0], neighborhood_cols[0]]].drop_duplicates()
            logger.info(f"Created neighborhood reference with {reference_data.shape[0]} entries from combined data")
            return reference_data
    
    # Next try from Airbnb data
    airbnb_data = load_airbnb_data(filtered=False)
    if airbnb_data is not None:
        # Look for zip code and neighborhood columns
        zip_cols = [col for col in airbnb_data.columns if 'zip' in col.lower()]
        neighborhood_cols = [col for col in airbnb_data.columns if any(term in col.lower() for term in ['neighborhood', 'area', 'community'])]
        
        if zip_cols and neighborhood_cols:
            # Create reference from available columns
            reference_data = airbnb_data[[zip_cols[0], neighborhood_cols[0]]].drop_duplicates()
            logger.info(f"Created neighborhood reference with {reference_data.shape[0]} entries from Airbnb data")
            return reference_data
    
    # If all else fails, create a minimal reference with Miami-Dade zipcodes
    logger.warning("Creating minimal zipcode reference as fallback")
    miami_zipcodes = [
        33101, 33109, 33112, 33116, 33122, 33124, 33125, 33126, 33127, 33128,
        33129, 33130, 33131, 33132, 33133, 33134, 33135, 33136, 33137, 33138,
        33139, 33140, 33141, 33142, 33143, 33144, 33145, 33146, 33147, 33149,
        33150, 33154, 33155, 33156, 33157, 33158, 33160, 33161, 33162, 33165,
        33166, 33167, 33168, 33169, 33170, 33172, 33173, 33174, 33175, 33176,
        33177, 33178, 33179, 33180, 33181, 33182, 33183, 33184, 33185, 33186,
        33187, 33189, 33190, 33193, 33194, 33196
    ]
    
    reference_data = pd.DataFrame({
        'zipcode': miami_zipcodes,
        'neighborhood': [f"Area {zipcode}" for zipcode in miami_zipcodes]
    })
    
    return reference_data


# Helper function for backward compatibility with the DataLoader class
def get_data(dataset_name):
    """
    Get a specific dataset by name (for compatibility with DataLoader).
    
    Args:
        dataset_name (str): Name of the dataset to retrieve
        
    Returns:
        pd.DataFrame: The requested dataset or None if not available
    """
    dataset_map = {
        'airbnb': load_airbnb_data,
        'airbnb_cleaned': load_airbnb_data,
        'airbnb_listings': load_airbnb_data,
        'listings': load_airbnb_data,
        'listings_processed': load_airbnb_data,
        'housing': load_miami_housing_data,
        'housing_data': load_miami_housing_data,
        'miami_housing': load_miami_housing_data,
        'property': load_miami_housing_data,
        'census': load_census_data,
        'census_data': load_census_data,
        'census_cleaned': load_census_data,
        'combined': load_combined_data,
        'merged': load_combined_data,
        'neighborhoods': load_neighborhood_data,
        'zillow': load_zillow_data,
        'zori': load_zillow_data
    }
    
    if dataset_name in dataset_map:
        return dataset_map[dataset_name]()
    
    # Special case for file-based datasets
    try:
        # Check if there's a CSV file with this name in the processed directory
        dataset_file = PROCESSED_DATA_DIR / f"{dataset_name}.csv"
        if os.path.exists(dataset_file):
            return pd.read_csv(dataset_file)
            
        # Also check in subdirectories
        for subdir in os.listdir(PROCESSED_DATA_DIR):
            subdir_path = PROCESSED_DATA_DIR / subdir
            if os.path.isdir(subdir_path):
                dataset_file = subdir_path / f"{dataset_name}.csv"
                if os.path.exists(dataset_file):
                    return pd.read_csv(dataset_file)
    except Exception as e:
        logger.error(f"Error loading dataset '{dataset_name}' from file: {e}")
    
    logger.warning(f"Dataset '{dataset_name}' not found")
    return None


# Functions for getting summary statistics
def get_summary_stats():
    """
    Get summary statistics for all datasets.
    
    Returns:
        dict: Dictionary of summary statistics
    """
    summary_stats = {}
    
    # Try to load summary files
    summary_dir = PROCESSED_DATA_DIR / "summary"
    
    if not summary_dir.exists():
        logger.warning("Summary directory does not exist")
        return summary_stats
    
    # Look for summary files
    for summary_file in summary_dir.glob("summary_*.csv"):
        try:
            dataset_name = summary_file.stem.replace("summary_", "")
            summary_data = pd.read_csv(summary_file)
            
            if not summary_data.empty:
                # Convert to dictionary format
                if summary_data.shape[0] == 1:
                    # Single row, convert to simple dict
                    summary_stats[dataset_name] = summary_data.iloc[0].to_dict()
                else:
                    # Multiple rows, keep as nested dict
                    summary_stats[dataset_name] = summary_data.to_dict(orient='records')
            
        except Exception as e:
            logger.error(f"Error loading summary stats from {summary_file}: {e}")
    
    return summary_stats


if __name__ == "__main__":
    # Test data loading functions
    print("Testing data loading functions...")
    
    airbnb_data = load_airbnb_data()
    if airbnb_data is not None:
        print(f"Loaded Airbnb data: {airbnb_data.shape}")
    
    housing_data = load_miami_housing_data()
    if housing_data is not None:
        print(f"Loaded housing data: {housing_data.shape}")
    
    census_data = load_census_data()
    if census_data is not None:
        print(f"Loaded census data: {census_data.shape}")
    
    combined_data = load_combined_data()
    if combined_data is not None:
        print(f"Loaded combined data: {combined_data.shape}")
    
    neighborhood_data = load_neighborhood_data()
    if neighborhood_data is not None:
        print(f"Loaded neighborhood data: {neighborhood_data.shape}")
    
    zillow_data = load_zillow_data()
    if zillow_data is not None:
        print(f"Loaded Zillow data: {zillow_data.shape}")
