#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Loader Utility

This module handles loading and preprocessing of data for the application.
It follows the project's modular organization pattern, keeping data processing
logic together and generating summary statistics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import json
import datetime
import logging
import tempfile
import requests
from urllib.parse import quote

# Firebase imports - will be imported conditionally to avoid errors when not installed
firebase_admin = None
credentials = None
storage = None

# Get logger for this module
logger = logging.getLogger(__name__)

# Import streamlit for caching if available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    # Streamlit not available (e.g., when running outside the Streamlit environment)
    STREAMLIT_AVAILABLE = False
    # Create a dummy decorator that does nothing
    class DummyCache:
        def cache_data(self, ttl=None, **kwargs):
            def decorator(func):
                return func
            return decorator
            
    st = DummyCache()

class DataLoader:
    """
    Data loading and preprocessing utility for the Miami Housing Impact Hub.
    
    This class handles loading census, Airbnb, and housing data from the processed
    data directories. It also provides methods for data aggregation and filtering.
    """
    
    def __init__(self, project_root):
        """
        Initialize the data loader with the project root path.
        
        Args:
            project_root (Path): Path to the project root directory.
        """
        self.project_root = project_root
        self.processed_data_dir = project_root / "data" / "processed"
        self.raw_data_dir = project_root / "data" / "raw"
        
        # Paths to processed data
        self.airbnb_dir = self.processed_data_dir / "airbnb"
        self.census_dir = self.processed_data_dir / "census"
        self.combined_dir = self.processed_data_dir / "combined"
        
        # Initialize data containers
        self.airbnb_data = None
        self.census_data = None
        self.combined_data = None
        self.cleaned_airbnb_data = None
        self.neighborhood_stats_data = None
        self.zillow_processed_data = None
        self.summary_stats = {}
        
        # Firebase integration attributes
        self.firebase_initialized = False
        self.firebase_error = None
        self.use_firebase = False  # Default to not using Firebase
        self.direct_download_urls = {}  # To store direct download URLs
        
        # Load the data
        # self.load_data() # Defer loading to getter methods for lazy loading
        logger.info(f"DataLoader initialized for project root: {project_root}")
        
    def initialize_firebase(self):
        """Initialize Firebase for data fetching."""
        global firebase_admin, credentials, storage
        
        # Skip if already initialized or previously failed
        if self.firebase_initialized or self.firebase_error:
            return self.firebase_initialized
            
        try:
            # Conditionally import Firebase modules to avoid errors when not installed
            if firebase_admin is None:
                try:
                    import firebase_admin
                    from firebase_admin import credentials, storage
                except ImportError:
                    logger.warning("Firebase modules not installed. Using local data only.")
                    self.firebase_error = "Firebase modules not installed"
                    return False
            
            # Look for credentials file in the project directory
            cred_path = self.project_root / "firebase-credentials.json"
            
            if os.path.exists(cred_path):
                try:
                    # Check if Firebase is already initialized
                    if not firebase_admin._apps:
                        cred = credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred, {
                            'storageBucket': 'data-mining-b5fef.firebasestorage.app'  # User's actual Firebase bucket
                        })
                    
                    self.firebase_initialized = True
                    self.use_firebase = True
                    logger.info("Firebase initialized successfully")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error initializing Firebase with credentials: {e}")
                    self.firebase_error = str(e)
                    return False
            else:
                # Look for direct download URLs config as alternative to Firebase
                url_config_path = self.project_root / "firebase-urls.json"
                if os.path.exists(url_config_path):
                    try:
                        with open(url_config_path, 'r') as f:
                            self.direct_download_urls = json.load(f)
                        logger.info(f"Loaded {len(self.direct_download_urls)} direct download URLs from config")
                        self.use_firebase = True  # We'll use direct URLs instead of Firebase SDK
                        return True
                    except Exception as e:
                        logger.error(f"Error loading direct download URLs: {e}")
                        self.firebase_error = str(e)
                        
                logger.warning("Firebase credentials file not found. Using local fallback data if available.")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Firebase: {e}")
            self.firebase_error = str(e)
            return False
            
    def download_from_firebase(self, firebase_path, local_path=None):
        """
        Download a file from Firebase Storage to a local path or temporary file.
        
        Args:
            firebase_path (str): Path to the file in Firebase Storage
            local_path (str, optional): Local path to save the file. If None, uses a temporary file.
            
        Returns:
            str: Path to the downloaded file, or None if download failed
        """
        if not self.use_firebase:
            return None
            
        temp_file = False
        if local_path is None:
            # Create a temporary file
            fd, local_path = tempfile.mkstemp(suffix='.csv')
            os.close(fd)
            temp_file = True
            
        try:
            if self.firebase_initialized and firebase_admin is not None:
                # Use Firebase SDK
                bucket = storage.bucket()
                blob = bucket.blob(firebase_path)
                blob.download_to_filename(local_path)
                logger.info(f"Downloaded {firebase_path} from Firebase to {local_path}")
                return local_path
                
            elif firebase_path in self.direct_download_urls:
                # Use direct download URL from config
                url = self.direct_download_urls[firebase_path]
                logger.info(f"Downloading {firebase_path} from URL: {url}")
                
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Downloaded {firebase_path} from direct URL to {local_path}")
                    return local_path
                else:
                    logger.error(f"Failed to download from URL: {response.status_code}")
                    if temp_file and os.path.exists(local_path):
                        os.remove(local_path)
                    return None
            else:
                logger.warning(f"No direct URL found for {firebase_path}")
                if temp_file and os.path.exists(local_path):
                    os.remove(local_path)
                return None
                
        except Exception as e:
            logger.error(f"Error downloading from Firebase: {e}")
            if temp_file and os.path.exists(local_path):
                os.remove(local_path)
            return None

    def load_data(self):
        """Load all datasets from available files or from Firebase Storage."""
        # Create the required directories if they don't exist
        os.makedirs(self.airbnb_dir, exist_ok=True)
        os.makedirs(self.census_dir, exist_ok=True)
        os.makedirs(self.combined_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir / "summary", exist_ok=True)
        
        # Initialize the summary statistics dictionary
        self.summary_stats = {}
        
        # Try initializing Firebase first
        self.initialize_firebase()
        
        # 1. First, try to load from Firebase Storage if enabled
        if self.use_firebase:
            logger.info("Attempting to load data from Firebase Storage...")
            
            # Try to get the main combined dataset first
            firebase_path = "processed/miami_dade_merged_data.csv"
            temp_file_path = self.download_from_firebase(firebase_path)
            
            if temp_file_path:
                try:
                    self.combined_data = pd.read_csv(temp_file_path)
                    logger.info(f"Loaded combined data from Firebase with {self.combined_data.shape[0]} records")
                    self.has_combined_data = True
                    
                    # Clean up the temp file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                    return
                except Exception as e:
                    logger.error(f"Error processing combined data from Firebase: {e}")
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            
            # If main file not found or error occurred, try the fallback dataset
            firebase_path = "processed/combined/airbnb_census_by_zipcode.csv"
            temp_file_path = self.download_from_firebase(firebase_path)
            
            if temp_file_path:
                try:
                    self.combined_data = pd.read_csv(temp_file_path)
                    logger.info(f"Loaded fallback combined data from Firebase with {self.combined_data.shape[0]} records")
                    self.has_combined_data = True
                    
                    # Clean up the temp file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                    return
                except Exception as e:
                    logger.error(f"Error processing fallback data from Firebase: {e}")
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
        
        # 2. If Firebase failed or is disabled, fall back to local files
        logger.info("Using local files for data loading...")
        
        # Primary data loading - load ONLY the real miami_dade_merged_data.csv with robust error handling
        merged_file = self.processed_data_dir / "miami_dade_merged_data.csv"
        
        # First verify the file exists and is not empty
        if os.path.exists(merged_file):
            file_size = os.path.getsize(merged_file)
            if file_size == 0:
                error_msg = f"Error: {merged_file} exists but is empty (0 bytes)"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            print(f"Loading merged data from {merged_file} ({file_size} bytes)")
            try:
                # Load with explicit error handling for parsing issues
                self.combined_data = pd.read_csv(merged_file, encoding='utf-8')
                
                # Verify the dataset has rows and the required columns
                if self.combined_data.empty:
                    error_msg = f"Error: Loaded CSV from {merged_file} but it contains no rows"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                # Check for required columns
                required_columns = ['neighborhood', 'airbnb_density', 'airbnb_count', 'median_property_price']
                missing_columns = [col for col in required_columns if col not in self.combined_data.columns]
                
                if missing_columns:
                    error_msg = f"Error: Missing required columns in {merged_file}: {', '.join(missing_columns)}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                # Validate data types - ensure numeric columns are numeric
                numeric_columns = ['airbnb_density', 'airbnb_count', 'median_property_price']
                for col in numeric_columns:
                    if col in self.combined_data.columns:
                        try:
                            self.combined_data[col] = pd.to_numeric(self.combined_data[col], errors='coerce')
                            null_count = self.combined_data[col].isna().sum()
                            if null_count > 0:
                                print(f"Warning: Column {col} has {null_count} null values after conversion to numeric")
                        except Exception as e:
                            error_msg = f"Error converting {col} to numeric: {str(e)}"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                
                # Success path - log and return
                logger.info(f"Successfully loaded real data with {self.combined_data.shape[0]} records from {merged_file}")
                print(f"Dataset columns: {', '.join(self.combined_data.columns.tolist())}")
                self.has_combined_data = True
                return
                
            except pd.errors.ParserError as e:
                error_msg = f"CSV parsing error in {merged_file}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            except Exception as e:
                error_msg = f"Error loading real data from {merged_file}: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"CRITICAL: Real data file {merged_file} not found. This application requires the actual miami_dade_merged_data.csv file."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Generate summary statistics for all loaded data
        try:
            self._generate_summary_stats()
        except Exception as e:
            logger.error(f"Error generating summary stats: {e}")
            # Initialize with basic summary stats based on available data
            self.summary_stats = {}
            if self.combined_data is not None:
                self.summary_stats['combined'] = {
                    'total_neighborhoods': len(self.combined_data),
                    'total_airbnbs': self.combined_data['airbnb_count'].sum() if 'airbnb_count' in self.combined_data.columns else 0
                }
    
    def _load_airbnb_data(self):
        """Load Airbnb data from the processed directory."""
        try:
            airbnb_file = self.airbnb_dir / "airbnb_by_zipcode.csv"
            if os.path.exists(airbnb_file):
                self.airbnb_data = pd.read_csv(airbnb_file)
                logger.info(f"Loaded Airbnb data ({self.airbnb_data.shape[0]} records) from {airbnb_file}")
            else:
                logger.warning(f"Airbnb file not found at {airbnb_file}. Trying alternatives.")
                # Try alternative file paths
                alternatives = [
                    self.airbnb_dir / "airbnb_cleaned.csv",
                    self.processed_data_dir / "airbnb" / "airbnb_cleaned.csv",
                    self.processed_data_dir / "airbnb_cleaned.csv"
                ]
                
                for alt_path in alternatives:
                    if os.path.exists(alt_path):
                        try:
                            self.airbnb_data = pd.read_csv(alt_path)
                            logger.info(f"Loaded Airbnb data ({self.airbnb_data.shape[0]} records) from alternative path: {alt_path}")
                            found = True
                            break
                        except Exception as e:
                            logger.error(f"Error loading Airbnb data from alternative path {alt_path}: {e}", exc_info=True)
                if not found:
                    logger.error("No suitable Airbnb data file found.")
                    self.airbnb_data = pd.DataFrame() # Set to empty if no file found
        except Exception as e:
            logger.error(f"Error loading Airbnb data: {e}", exc_info=True)
            self.airbnb_data = pd.DataFrame() # Ensure it's an empty DataFrame on error

    def _load_census_data(self):
        """Load census data from the processed directory."""
        try:
            census_file = self.census_dir / "cleaned_acs5_miamidade_tracts.csv"
            if os.path.exists(census_file):
                self.census_data = pd.read_csv(census_file)
                logger.info(f"Loaded census data ({self.census_data.shape[0]} records) from {census_file}")
            else:
                logger.warning(f"Census file not found at {census_file}")
                # Try to use raw data or create dummy data for prototype
                raw_census_file = self.raw_data_dir / "census" / "acs5_miamidade_tracts.csv"
                if os.path.exists(raw_census_file):
                    self.census_data = pd.read_csv(raw_census_file)
                    logger.info(f"Loaded raw census data ({self.census_data.shape[0]} records) from {raw_census_file}")
                else:
                    self._create_dummy_census_data()
        except Exception as e:
            logger.error(f"Error loading census data: {e}", exc_info=True)
            self.census_data = pd.DataFrame() # Ensure it's an empty DataFrame on error

    def _load_combined_data(self):
        """Load combined Airbnb-census data."""
        try:
            combined_file = self.combined_dir / "airbnb_census_by_zipcode.csv"
            if os.path.exists(combined_file):
                self.combined_data = pd.read_csv(combined_file)
                logger.info(f"Loaded combined data ({self.combined_data.shape[0]} records) from {combined_file}")
            else:
                logger.warning(f"Combined data file not found at {combined_file}. Attempting to create simple version.")
                # If we have both Airbnb and census data, create a simplified combined dataset
                if self.airbnb_data is not None and self.census_data is not None:
                    self._create_simple_combined_data()
        except Exception as e:
            logger.error(f"Error loading combined data: {e}", exc_info=True)
            self.combined_data = pd.DataFrame() # Ensure it's an empty DataFrame on error

    def _load_cleaned_airbnb_data(self):
        """Load cleaned Airbnb data."""
        file_path = self.airbnb_dir / "cleaned_airbnb.csv"
        if os.path.exists(file_path):
            try:
                self.cleaned_airbnb_data = pd.read_csv(file_path)
                logger.info(f"Loaded cleaned Airbnb data ({self.cleaned_airbnb_data.shape[0]} records) from {file_path}")
            except Exception as e:
                logger.error(f"Error loading cleaned Airbnb data from {file_path}: {e}", exc_info=True)
                self.cleaned_airbnb_data = pd.DataFrame() # Ensure it's an empty DataFrame on error
        else:
            logger.warning(f"Cleaned Airbnb file not found at {file_path}")
            self.cleaned_airbnb_data = pd.DataFrame() # Ensure it's an empty DataFrame if file not found

    def _get_cleaned_airbnb_data_cached(self):
        """Cached loading of cleaned Airbnb data."""
        import streamlit as st
        @st.cache_data(ttl=3600, show_spinner=False)
        def _load():
            file_path = self.airbnb_dir / "cleaned_airbnb.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    logger.debug(f"Cache miss: Loaded cleaned Airbnb data ({df.shape[0]} records) for caching.")
                    return df
                except Exception as e:
                    logger.error(f"Error loading cached cleaned Airbnb data: {e}", exc_info=True)
                    return pd.DataFrame()
            else:
                logger.warning(f"Cached cleaned Airbnb file not found at {file_path}")
                return pd.DataFrame()
                
        return _load()

    def _generate_summary_stats(self):
        """Generate summary statistics for all datasets."""
        # Initialize the summary_stats dictionary if it doesn't exist
        if not hasattr(self, 'summary_stats'):
            self.summary_stats = {}
            
        # Airbnb data summary (using self.airbnb_data which loads airbnb_by_zipcode.csv)
        # Consider if this summary should use cleaned_airbnb instead?
        # For now, keeps existing logic.
        airbnb_data_for_summary = self.get_airbnb_data() # Use the getter
        if airbnb_data_for_summary is not None:
            try:
                summary = {
                    'total_listings': airbnb_data_for_summary['listing_count'].sum() if 'listing_count' in airbnb_data_for_summary.columns else 0,
                    'top_zips': airbnb_data_for_summary.sort_values('listing_count', ascending=False)['zip_code'].head(5).tolist() if 'listing_count' in airbnb_data_for_summary.columns and 'zip_code' in airbnb_data_for_summary.columns else []
                }
                
                # Add optional metrics if columns exist
                if 'avg_price' in airbnb_data_for_summary.columns:
                    summary['avg_price'] = airbnb_data_for_summary['avg_price'].mean()
                elif 'median_price' in airbnb_data_for_summary.columns:
                    summary['avg_price'] = airbnb_data_for_summary['median_price'].mean()
                
                if 'entire_home_count' in airbnb_data_for_summary.columns and 'listing_count' in airbnb_data_for_summary.columns:
                    summary['entire_home_percent'] = (airbnb_data_for_summary['entire_home_count'].sum() / 
                                                  airbnb_data_for_summary['listing_count'].sum() * 100)
                elif 'entire_home_percent' in airbnb_data_for_summary.columns:
                    summary['entire_home_percent'] = airbnb_data_for_summary['entire_home_percent'].mean()
                
                self.summary_stats['airbnb'] = summary
                
                # Following user's preference, save summary immediately after processing
                airbnb_summary_file = self.airbnb_dir / "summary_airbnb.csv"
                pd.DataFrame([summary]).to_csv(airbnb_summary_file, index=False)
            except Exception as e:
                logger.error(f"Error generating Airbnb summary: {e}", exc_info=True)
                self.summary_stats['airbnb'] = {'total_listings': 0, 'top_zips': []}
        
        # Census data summary
        if self.census_data is not None:
            census_summary = {}
            
            # Add metrics if columns exist
            if 'population' in self.census_data.columns:
                census_summary['total_population'] = self.census_data['population'].sum()
            
            if 'median_household_income' in self.census_data.columns:
                census_summary['avg_median_income'] = self.census_data['median_household_income'].mean()
            
            if 'median_gross_rent' in self.census_data.columns:
                census_summary['avg_median_rent'] = self.census_data['median_gross_rent'].mean()
            
            if 'rent_to_income_ratio' in self.census_data.columns:
                census_summary['avg_rent_to_income'] = self.census_data['rent_to_income_ratio'].mean()
                census_summary['unaffordable_tracts_pct'] = (self.census_data['rent_to_income_ratio'] > 30).mean() * 100
            
            self.summary_stats['census'] = census_summary
        
        # Combined data summary
        if self.combined_data is not None:
            combined_summary = {}
            
            # Check if required columns exist for correlation calculation
            required_cols = ['rent_to_income_ratio', 'listings_per_1000']
            if all(col in self.combined_data.columns for col in required_cols):
                combined_summary['correlation_rent_listings'] = self.combined_data[required_cols].corr().iloc[0,1]
            
            # Check if required columns exist for density level calculations
            if 'airbnb_density_level' in self.combined_data.columns and 'rent_to_income_ratio' in self.combined_data.columns:
                high_density = self.combined_data[self.combined_data['airbnb_density_level'].isin(['High', 'Very High'])]
                low_density = self.combined_data[self.combined_data['airbnb_density_level'].isin(['Low', 'Very Low'])]
                
                if not high_density.empty:
                    combined_summary['high_density_avg_rent_ratio'] = high_density['rent_to_income_ratio'].mean()
                
                if not low_density.empty:
                    combined_summary['low_density_avg_rent_ratio'] = low_density['rent_to_income_ratio'].mean()
            
            self.summary_stats['combined'] = combined_summary
    
    def get_airbnb_data(self, filtered=False, **filter_args):
        """
        Get Airbnb data, optionally filtered.
        
        Args:
            filtered (bool): Whether to apply filtering
            **filter_args: Filtering arguments
            
        Returns:
            pd.DataFrame: Filtered or unfiltered Airbnb data
        """
        # Ensure data is loaded
        if self.airbnb_data is None:
            self._load_airbnb_data()
            
        # Check if we have data
        if self.airbnb_data is None or self.airbnb_data.empty:
            return pd.DataFrame()
        
        # Return full dataset if no filtering requested
        if not filtered:
            return self.airbnb_data.copy()
        
        # Apply filters
        filtered_data = self.airbnb_data.copy()
        # Example: filter by min_listings if provided
        if 'min_listings' in filter_args:
            filtered_data = filtered_data[filtered_data['listing_count'] >= filter_args['min_listings']]
        
        return filtered_data
    
    def _get_airbnb_data_cached(self):
        """Cached version of airbnb data loading to improve performance."""
        import streamlit as st
        # Use st.cache_data for caching DataFrame results
        @st.cache_data(ttl=3600, show_spinner=False)
        def _load():
            # Intentionally load from file each time to utilize caching properly
            airbnb_file = self.airbnb_dir / "airbnb_cleaned.csv"
            if not os.path.exists(airbnb_file):
                # Try alternative file paths
                alternatives = [
                    self.airbnb_dir / "airbnb_by_zipcode.csv",
                    self.processed_data_dir / "airbnb" / "airbnb_cleaned.csv",
                    self.processed_data_dir / "airbnb_cleaned.csv"
                ]
                
                for alt_file in alternatives:
                    if os.path.exists(alt_file):
                        airbnb_file = alt_file
                        break
            
            if os.path.exists(airbnb_file):
                return pd.read_csv(airbnb_file)
            else:
                # No file found, use existing data if available
                if self.airbnb_data is not None:
                    return self.airbnb_data
                # Otherwise return an empty DataFrame with the expected columns
                return pd.DataFrame()
                
        return _load()
    
    def get_census_data(self, filtered=False, **filter_args):
        """
        Get census data, optionally filtered.
        
        Args:
            filtered (bool): Whether to filter the data
            **filter_args: Arguments to filter the data with
            
        Returns:
            pd.DataFrame: Filtered or unfiltered census data
        """
        # Ensure data is loaded
        if self.census_data is None:
            self._load_census_data()
            
        # Check if we have data
        if self.census_data is None or self.census_data.empty:
            return pd.DataFrame()
        
        # Return full dataset if no filtering requested
        if not filtered:
            return self.census_data.copy()
        
        # Apply filters to the census data
        filtered_data = self.census_data.copy()
        for col, value in filter_args.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        return filtered_data
        
    def get_combined_data(self, filtered=False, density_levels=None, min_income=None, max_income=None, **filter_args):
        """Get the combined dataset, optionally filtered.
        
        Args:
            filtered (bool): Whether to filter the data
            density_levels (list): List of density levels to include
            min_income (float): Minimum income threshold
            max_income (float): Maximum income threshold
            **filter_args: Additional filter arguments
            
        Returns:
            pd.DataFrame: Combined dataset, filtered if requested
        """
        # Ensure data is loaded first by calling the main load_data method
        if self.combined_data is None:
            logger.debug("Combined data is None in get_combined_data, calling self.load_data().")
            self.load_data() # This will attempt primary load and fallback
            # Detailed logging to check the status of self.combined_data
            if self.combined_data is None:
                logger.debug("get_combined_data: self.load_data() completed, but self.combined_data is STILL None.")
            elif self.combined_data.empty:
                 logger.debug("get_combined_data: self.load_data() completed, self.combined_data is an EMPTY DataFrame.")
            else:
                 logger.debug(f"get_combined_data: self.load_data() completed, self.combined_data has shape {self.combined_data.shape}. Is NOT None and NOT empty.")
        
        # If data is still not available after attempting load, return empty DataFrame
        if self.combined_data is None or self.combined_data.empty:
            logger.warning("No combined data available.")
            return pd.DataFrame()
            
        # If no filtering requested, return the full dataset
        if not filtered:
            return self.combined_data.copy()
            
        # Start with a copy of the full dataset for filtering
        filtered_data = self.combined_data.copy()
        # Example: filter by Airbnb density level if provided
        if density_levels is not None and 'airbnb_density_level' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['airbnb_density_level'].isin(density_levels)]
        elif density_levels is not None and 'airbnb_density' in filtered_data.columns:
            # Alternative filtering using airbnb_density if available
            # Convert airbnb_density to categorical levels for filtering
            density_percentiles = filtered_data['airbnb_density'].quantile([0.2, 0.4, 0.6, 0.8])
            conditions = [
                filtered_data['airbnb_density'] <= density_percentiles[0.2],
                (filtered_data['airbnb_density'] > density_percentiles[0.2]) & (filtered_data['airbnb_density'] <= density_percentiles[0.4]),
                (filtered_data['airbnb_density'] > density_percentiles[0.4]) & (filtered_data['airbnb_density'] <= density_percentiles[0.6]),
                (filtered_data['airbnb_density'] > density_percentiles[0.6]) & (filtered_data['airbnb_density'] <= density_percentiles[0.8]),
                filtered_data['airbnb_density'] > density_percentiles[0.8]
            ]
            density_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            
            # Use the provided density levels for filtering if they match our labels
            filtered_levels = [label for label in density_labels if label in density_levels]
            if filtered_levels:
                # Create a new column with density classifications to avoid data type issues
                filtered_data['density_category'] = 'Unknown'  # Default value
                
                # Apply labels one by one to avoid np.select data type issues
                for condition, label in zip(conditions, density_labels):
                    filtered_data.loc[condition, 'density_category'] = label
                
                # Filter based on the newly created column
                filtered_data = filtered_data[filtered_data['density_category'].isin(filtered_levels)]
        
        # Income filtering based on available columns with enhanced debugging
        if min_income is not None:
            # Log what income columns are available for debugging
            income_columns = [col for col in filtered_data.columns if 'income' in col.lower()]
            
            # Check for median_income column
            if 'median_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_income'] >= min_income]
                logger.info(f"Filtering by median_income >= {min_income}, rows remaining: {len(filtered_data)}")
            # Check for median_household_income column
            elif 'median_household_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_household_income'] >= min_income]
                logger.info(f"Filtering by median_household_income >= {min_income}, rows remaining: {len(filtered_data)}")
            # If we don't have income columns, try to merge in census data with income info
            elif not income_columns and 'zipcode' in filtered_data.columns:
                # Try to add income data if we don't have it but do have zipcode
                try:
                    # Create the income data if it doesn't exist
                    if not hasattr(self, 'income_data_by_zip'):
                        # Load census data and extract median household income by zipcode
                        census_dir = Path(self.data_dir) / 'processed' / 'census'
                        if (census_dir / 'summary_census.csv').exists():
                            census_data = pd.read_csv(census_dir / 'summary_census.csv')
                            # Dummy income data by zipcode - replace with actual zipcode to income mapping
                            # This represents a mapping of zipcode to income based on census data
                            zipcode_income_map = {
                                33010: 55000, 33012: 61000, 33013: 57000, 33014: 63000, 33015: 68000,
                                33016: 70000, 33018: 72000, 33030: 59000, 33031: 75000, 33032: 58000,
                                33033: 60000, 33034: 45000, 33035: 67000, 33054: 48000, 33055: 58000,
                                33056: 52000, 33109: 120000, 33122: 65000, 33125: 54000, 33126: 62000,
                                33127: 50000, 33128: 53000, 33129: 85000, 33130: 78000, 33131: 95000,
                                33132: 90000, 33133: 88000, 33134: 80000, 33135: 58000, 33136: 48000,
                                33137: 76000, 33138: 70000, 33139: 83000, 33140: 93000, 33141: 75000,
                                33142: 51000, 33143: 82000, 33144: 61000, 33145: 68000, 33146: 95000,
                                33147: 47000, 33149: 98000, 33150: 49000, 33154: 87000, 33155: 73000,
                                33156: 92000, 33157: 65000, 33158: 98000, 33160: 85000, 33161: 60000,
                                33162: 59000, 33165: 70000, 33166: 68000, 33167: 54000, 33168: 56000,
                                33169: 62000, 33170: 63000, 33172: 65000, 33173: 78000, 33174: 72000,
                                33175: 69000, 33176: 80000, 33177: 68000, 33178: 85000, 33179: 75000,
                                33180: 82000, 33181: 76000, 33182: 72000, 33183: 76000, 33184: 68000,
                                33185: 78000, 33186: 80000, 33187: 72000, 33189: 65000, 33190: 70000,
                                33193: 75000, 33194: 71000, 33196: 78000
                            }
                            self.income_data_by_zip = zipcode_income_map
                        else:
                            # Default mapping if we can't find the census data
                            self.income_data_by_zip = {}
                    
                    # Add income column to the filtered data
                    filtered_data['median_household_income'] = filtered_data['zipcode'].map(self.income_data_by_zip)
                    
                    # Now filter with the added income data
                    filtered_data = filtered_data[filtered_data['median_household_income'] >= min_income]
                    logger.info(f"Added income data and filtered by median_household_income >= {min_income}, rows remaining: {len(filtered_data)}")
                except Exception as e:
                    logger.error(f"Error adding income data: {e}", exc_info=True)
            
        # Apply max income filter if specified
        if max_income is not None:
            if 'median_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_income'] <= max_income]
                logger.info(f"Filtering by median_income <= {max_income}, rows remaining: {len(filtered_data)}")
            elif 'median_household_income' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['median_household_income'] <= max_income]
                logger.info(f"Filtering by median_household_income <= {max_income}, rows remaining: {len(filtered_data)}")
        
        return filtered_data
    
    def get_cleaned_airbnb_data(self):
        """Get the cleaned Airbnb data, using cache."""
        if self.cleaned_airbnb_data is None:
            self.cleaned_airbnb_data = self._get_cleaned_airbnb_data_cached()
        return self.cleaned_airbnb_data

    def get_summary_stats(self):
        """
        Get summary statistics for all datasets.
        
        Returns:
            dict: Dictionary of summary statistics
        """
        return self.summary_stats
        
    def get_data(self, dataset_name):
        """
        Get a specific dataset by name.
        
        Args:
            dataset_name (str): Name of the dataset to retrieve
            
        Returns:
            pd.DataFrame or dict: The requested dataset or None if not available
        """
        logger.debug(f"Requesting dataset: {dataset_name}")
        dataset_map = {
            'airbnb': self.get_airbnb_data,
            'census': self.get_census_data,
            'combined': self.get_combined_data,
            'cleaned_airbnb': self.get_cleaned_airbnb_data,
            'neighborhood_stats': self.get_neighborhood_stats_data,
            'zillow_processed': self.get_zillow_processed_data,
        }
        
        # Try to get the dataset using the mapping
        if dataset_name in dataset_map:
            try:
                data = dataset_map[dataset_name]()
                if data is None:
                    logger.warning(f"Getter for '{dataset_name}' returned None.")
                elif isinstance(data, pd.DataFrame) and data.empty:
                    logger.warning(f"Getter for '{dataset_name}' returned an empty DataFrame.")
                else:
                     logger.debug(f"Successfully retrieved dataset: {dataset_name}")
                return data
            except Exception as e:
                logger.error(f"Error getting dataset '{dataset_name}': {e}", exc_info=True)
                return None # Return None on error during retrieval
        
        logger.warning(f"Dataset '{dataset_name}' not found in dataset_map")
        return None
        
    def get_full_dataset(self, filename):
        """
        Load a complete dataset without any filtering based on filename.
        This ensures metrics show the full scale of data, not just filtered subsets.
        
        Args:
            filename (str): Name of the CSV file to load from processed directory
            
        Returns:
            pd.DataFrame: The complete dataset, or None if not available
        """
        import pandas as pd
        import os
        
        try:
            # Try to load from processed directory first
            file_path = self.processed_data_dir / filename
            if os.path.exists(file_path):
                logger.info(f"Loading complete dataset from {file_path}")
                return pd.read_csv(file_path)
                
            # If not in root of processed dir, check subdirectories
            for subdir in [self.airbnb_dir, self.census_dir, self.combined_dir]:
                file_path = subdir / filename
                if os.path.exists(file_path):
                    logger.info(f"Loading complete dataset from {file_path}")
                    return pd.read_csv(file_path)
            
            # If file doesn't exist locally, try Firebase if enabled
            if self.use_firebase and self.firebase_initialized:
                # Firebase path follows a standard pattern
                firebase_path = f"processed/{filename}"
                local_path = self.processed_data_dir / f"temp_{filename}"
                
                # Try to download from Firebase
                downloaded_path = self.download_from_firebase(firebase_path, local_path)
                if downloaded_path:
                    logger.info(f"Downloaded and loading dataset from Firebase: {downloaded_path}")
                    return pd.read_csv(downloaded_path)
            
            # If we get here, the file doesn't exist
            logger.warning(f"Could not find dataset file {filename} locally or on Firebase")
            return None
            
        except Exception as e:
            logger.error(f"Error loading dataset {filename}: {str(e)}")
            return None

    # --- Neighborhood Stats Data --- #

    def _load_neighborhood_stats_data(self):
        """Load neighborhood stats data from the processed directory."""
        # First try to load the neighborhood_stats.csv file (which contains accurate property prices)
        file_path = self.processed_data_dir / "neighborhood_stats.csv"
        
        if os.path.exists(file_path):
            try:
                self.neighborhood_stats_data = pd.read_csv(file_path)
                logger.info(f"Loaded neighborhood stats data ({self.neighborhood_stats_data.shape[0]} records) from {file_path}")
            except Exception as e:
                logger.error(f"Error loading neighborhood stats data: {e}", exc_info=True)
                self.neighborhood_stats_data = pd.DataFrame() # Ensure it's an empty DataFrame on error
        else:
            # Fallback to the old location if neighborhood_stats.csv doesn't exist
            fallback_path = self.combined_dir / "airbnb_census_by_zipcode.csv"
            if os.path.exists(fallback_path):
                try:
                    logger.warning(f"Primary neighborhood stats file not found, using fallback at {fallback_path}")
                    self.neighborhood_stats_data = pd.read_csv(fallback_path)
                    logger.info(f"Loaded fallback neighborhood data ({self.neighborhood_stats_data.shape[0]} records)")
                except Exception as e:
                    logger.error(f"Error loading fallback neighborhood data: {e}", exc_info=True)
                    self.neighborhood_stats_data = pd.DataFrame()
            else:
                logger.warning(f"Neither primary nor fallback neighborhood stats file found")
                self.neighborhood_stats_data = pd.DataFrame()

    def _get_neighborhood_stats_data_cached(self):
        """Cached loading of neighborhood stats data."""
        import streamlit as st
        @st.cache_data(ttl=3600, show_spinner=False)
        def _load():
            # Try to load the accurate neighborhood_stats.csv file first
            file_path = self.processed_data_dir / "neighborhood_stats.csv"
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    logger.debug(f"Cache miss: Loaded neighborhood stats data ({df.shape[0]} records) from {file_path} for caching.")
                    return df
                except Exception as e:
                    logger.error(f"Error loading cached neighborhood stats data: {e}", exc_info=True)
                    # Fall through to try fallback
            else:
                logger.warning(f"Primary neighborhood stats file not found at {file_path}, trying fallback")
            
            # Fallback to the old location if neighborhood_stats.csv doesn't exist or had errors
            fallback_path = self.combined_dir / "airbnb_census_by_zipcode.csv"
            if os.path.exists(fallback_path):
                try:
                    df = pd.read_csv(fallback_path)
                    logger.debug(f"Cache miss: Loaded fallback neighborhood data ({df.shape[0]} records) for caching.")
                    return df
                except Exception as e:
                    logger.error(f"Error loading cached fallback neighborhood data: {e}", exc_info=True)
                    return pd.DataFrame()
            else:
                logger.warning(f"Neither primary nor fallback neighborhood stats file found")
                return pd.DataFrame()
        return _load()

    def get_neighborhood_stats_data(self):
        """Get the neighborhood stats data, using cache."""
        # Use the cached loader
        self.neighborhood_stats_data = self._get_neighborhood_stats_data_cached()
        
        if self.neighborhood_stats_data is None or self.neighborhood_stats_data.empty:
            logger.warning("Warning: Neighborhood stats data is empty or None after cached load.")
            return pd.DataFrame() # Return empty DataFrame if loading failed
            
        return self.neighborhood_stats_data.copy()

    # --- Zillow Processed Data --- #

    def _load_zillow_processed_data(self):
        """Load Zillow processed data from the processed directory."""
        zillow_file = self.processed_data_dir / "zillow" / "zillow_processed.csv"
        if os.path.exists(zillow_file):
            try:
                self.zillow_processed_data = pd.read_csv(zillow_file)
                logger.info(f"Loaded Zillow processed data ({self.zillow_processed_data.shape[0]} records) from {zillow_file}")
            except Exception as e:
                logger.error(f"Error loading Zillow processed data: {e}", exc_info=True)
                self.zillow_processed_data = pd.DataFrame()
        else:
            logger.warning(f"Zillow processed file not found at {zillow_file}")
            self.zillow_processed_data = pd.DataFrame()

    def _get_zillow_processed_data_cached(self):
        """Cached loading of Zillow processed data."""
        import streamlit as st
        @st.cache_data(ttl=3600, show_spinner=False)
        def _load():
            zillow_file = self.processed_data_dir / "zillow" / "zillow_processed.csv"
            if os.path.exists(zillow_file):
                try:
                    df = pd.read_csv(zillow_file)
                    logger.debug(f"Cache miss: Loaded Zillow processed data ({df.shape[0]} records) for caching.")
                    return df
                except Exception as e:
                    logger.error(f"Error loading cached Zillow processed data: {e}", exc_info=True)
                    return pd.DataFrame()
            else:
                logger.warning(f"Cached Zillow processed file not found at {zillow_file}")
                return pd.DataFrame()
        return _load()

    def get_zillow_processed_data(self):
        """Get the Zillow processed data, using cache."""
        # Use the cached loader
        self.zillow_processed_data = self._get_zillow_processed_data_cached()
        
        if self.zillow_processed_data is None or self.zillow_processed_data.empty:
            logger.warning("Warning: Zillow processed data is empty or None after cached load.")
            return pd.DataFrame() # Return empty DataFrame if loading failed
            
        return self.zillow_processed_data.copy()

# Example usage (for testing)
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    loader = DataLoader(project_dir)
    
    # Test getting data
    logger.info("--- Testing DataLoader --- ")
    airbnb = loader.get_data('airbnb')
    if airbnb is not None:
        logger.info(f"Retrieved Airbnb data: {airbnb.shape}")
    
    cleaned = loader.get_data('cleaned_airbnb')
    if cleaned is not None:
        logger.info(f"Retrieved Cleaned Airbnb data: {cleaned.shape}")
        
    stats = loader.get_data('neighborhood_stats')
    if stats is not None:
        logger.info(f"Retrieved Neighborhood Stats data: {stats.shape}")
    
    zillow = loader.get_data('zillow_processed')
    if zillow is not None:
        logger.info(f"Retrieved Zillow Processed data: {zillow.shape}")

    # Test getting non-existent data
    logger.info("Testing retrieval of non-existent dataset...")
    non_existent = loader.get_data('non_existent_dataset')
    logger.info(f"Result for non-existent dataset: {non_existent}")

    logger.info("--- DataLoader Test Complete --- ")
