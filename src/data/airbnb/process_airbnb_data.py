#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Airbnb data for Miami-Dade County, mapping listings to census tracts
and calculating rental density metrics.

This script loads raw Airbnb data, cleans it, and performs spatial analysis
to determine which census tract each listing belongs to. It then calculates
various metrics like rental density, average price, and property type distributions
by census tract.

Output files:
- airbnb_with_tracts.csv: Processed Airbnb data with census tract information
- airbnb_tract_density.csv: Density metrics by census tract
- summary_airbnb.csv: Summary statistics of the Airbnb dataset
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pathlib import Path

# Set up file paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
AIRBNB_RAW_FILE = RAW_DATA_DIR / "airbnb" / "AirBnb-Data-MiamiData.csv"
CENSUS_TRACTS_FILE = RAW_DATA_DIR / "census" / "miami_dade_census_tracts.geojson"

# Create processed directories if they don't exist
PROCESSED_AIRBNB_DIR = PROCESSED_DATA_DIR / "airbnb"
os.makedirs(PROCESSED_AIRBNB_DIR, exist_ok=True)

def load_and_clean_airbnb_data(file_path):
    """
    Load and clean Airbnb data from CSV file.
    
    Args:
        file_path: Path to the raw Airbnb data CSV file
        
    Returns:
        pandas.DataFrame: Cleaned Airbnb data
    """
    print(f"Loading Airbnb data from {file_path}...")
    
    # Load data
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Successfully loaded data with {df.shape[0]} listings and {df.shape[1]} columns.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Initial cleaning
    print("Cleaning Airbnb data...")
    
    # Filter to include only listings with valid coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Convert price column to numeric if it exists
    if 'Average Daily Rate (USD)' in df.columns:
        df['price'] = pd.to_numeric(df['Average Daily Rate (USD)'], errors='coerce')
    
    # Handle bedrooms data
    if 'Bedrooms' in df.columns:
        # Convert 'Studio' to 0 if string type, otherwise leave as is
        if df['Bedrooms'].dtype == 'object':
            df['Bedrooms'] = df['Bedrooms'].replace('Studio', '0')
            df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
        
    # Extract property type information
    if 'Listing Type' in df.columns:
        df['is_entire_home'] = df['Listing Type'].str.contains('entire', case=False).fillna(False)
    
    # Create point geometry for spatial join
    geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    return gdf

def load_census_tracts(file_path):
    """
    Load census tract boundaries from geojson file.
    
    Args:
        file_path: Path to the census tract geojson file
        
    Returns:
        geopandas.GeoDataFrame: Census tract boundaries
    """
    print(f"Loading census tract boundaries from {file_path}...")
    
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            # Fall back to using census tracts from ACS data
            acs_file = RAW_DATA_DIR / "census" / "acs5_miamidade_tracts.csv"
            if os.path.exists(acs_file):
                print(f"Census tract geojson not found. Creating tract geometries from ACS data...")
                # Load census tract data from ACS
                tracts_df = pd.read_csv(acs_file)
                
                # If ACS data has geometry information, create GeoDataFrame
                if 'geometry' in tracts_df.columns:
                    from shapely import wkt
                    tracts_df['geometry'] = tracts_df['geometry'].apply(wkt.loads)
                    tracts = gpd.GeoDataFrame(tracts_df, geometry='geometry', crs="EPSG:4326")
                    return tracts
                else:
                    print("ACS data does not contain geometry information. Cannot proceed with spatial join.")
                    return None
            else:
                print("Neither census tract geojson nor ACS data with geometry found.")
                return None
        
        # Load geojson into GeoDataFrame
        tracts = gpd.read_file(file_path)
        print(f"Successfully loaded {len(tracts)} census tracts.")
        return tracts
    
    except Exception as e:
        print(f"Error loading census tracts: {e}")
        return None

def map_listings_to_tracts(airbnb_gdf, tracts_gdf):
    """
    Perform spatial join to map Airbnb listings to census tracts.
    
    Args:
        airbnb_gdf: GeoDataFrame of Airbnb listings with point geometry
        tracts_gdf: GeoDataFrame of census tract boundaries
        
    Returns:
        geopandas.GeoDataFrame: Airbnb listings with census tract information
    """
    print("Mapping Airbnb listings to census tracts...")
    
    # Ensure both GeoDataFrames have the same CRS
    if airbnb_gdf.crs != tracts_gdf.crs:
        airbnb_gdf = airbnb_gdf.to_crs(tracts_gdf.crs)
    
    # Perform spatial join
    joined = gpd.sjoin(airbnb_gdf, tracts_gdf, how="left", predicate="within")
    
    # Count how many listings were successfully mapped
    mapped_count = joined.dropna(subset=['index_right']).shape[0]
    print(f"Successfully mapped {mapped_count} out of {len(airbnb_gdf)} listings to census tracts.")
    
    return joined

def calculate_tract_metrics(airbnb_with_tracts, tracts_gdf, tract_id_col='GEOID'):
    """
    Calculate metrics by census tract including rental density, average price,
    and property type distribution.
    
    Args:
        airbnb_with_tracts: GeoDataFrame of Airbnb listings with census tract info
        tracts_gdf: GeoDataFrame of census tract boundaries
        tract_id_col: Column name for tract identifier
        
    Returns:
        pandas.DataFrame: Metrics by census tract
    """
    print("Calculating metrics by census tract...")
    
    # Ensure tract_id_col exists in both dataframes
    if tract_id_col not in tracts_gdf.columns:
        # Try to find the tract ID column
        possible_id_cols = [col for col in tracts_gdf.columns if 'tract' in col.lower() or 'id' in col.lower() or 'geoid' in col.lower()]
        if possible_id_cols:
            tract_id_col = possible_id_cols[0]
            print(f"Using '{tract_id_col}' as the tract identifier column.")
        else:
            print("Cannot find tract identifier column.")
            return None
    
    # Calculate area of each tract in square kilometers
    tracts_gdf = tracts_gdf.copy()
    if tracts_gdf.crs.is_geographic:
        # Convert to projected CRS for area calculation
        tracts_gdf = tracts_gdf.to_crs(epsg=3310)  # California Albers Equal Area Projection, but should work for Florida too
    
    tracts_gdf['area_sqkm'] = tracts_gdf.geometry.area / 1_000_000  # Convert from sq meters to sq km
    
    # Count listings by tract
    tract_counts = airbnb_with_tracts.groupby(tract_id_col).size().reset_index(name='listing_count')
    
    # Calculate average price by tract if price column exists
    if 'price' in airbnb_with_tracts.columns:
        tract_prices = airbnb_with_tracts.groupby(tract_id_col)['price'].agg(['mean', 'median', 'std']).reset_index()
        tract_metrics = pd.merge(tract_counts, tract_prices, on=tract_id_col, how='left')
    else:
        tract_metrics = tract_counts
    
    # Calculate property type distribution
    if 'is_entire_home' in airbnb_with_tracts.columns:
        entire_home_counts = airbnb_with_tracts[airbnb_with_tracts['is_entire_home']].groupby(tract_id_col).size().reset_index(name='entire_home_count')
        tract_metrics = pd.merge(tract_metrics, entire_home_counts, on=tract_id_col, how='left')
        tract_metrics['entire_home_percent'] = (tract_metrics['entire_home_count'] / tract_metrics['listing_count'] * 100).round(1)
    
    # Calculate bedroom distribution
    if 'bedrooms' in airbnb_with_tracts.columns:
        bedroom_stats = airbnb_with_tracts.groupby(tract_id_col)['bedrooms'].agg(['mean', 'median']).reset_index()
        bedroom_stats.columns = [tract_id_col, 'avg_bedrooms', 'median_bedrooms']
        tract_metrics = pd.merge(tract_metrics, bedroom_stats, on=tract_id_col, how='left')
    
    # Add area information
    tract_area = tracts_gdf[[tract_id_col, 'area_sqkm']].copy()
    tract_metrics = pd.merge(tract_metrics, tract_area, on=tract_id_col, how='left')
    
    # Calculate rental density (listings per sq km)
    tract_metrics['listing_density'] = (tract_metrics['listing_count'] / tract_metrics['area_sqkm']).round(2)
    
    # Fill NaN values
    tract_metrics = tract_metrics.fillna(0)
    
    return tract_metrics

def generate_summary_statistics(airbnb_df, airbnb_with_tracts, tract_metrics):
    """
    Generate and save summary statistics for the Airbnb dataset.
    
    Args:
        airbnb_df: Original Airbnb DataFrame
        airbnb_with_tracts: Airbnb data with census tract information
        tract_metrics: Census tract metrics DataFrame
        
    Returns:
        pandas.DataFrame: Summary statistics
    """
    print("Generating summary statistics...")
    
    summary = []
    
    # Overall statistics
    total_listings = len(airbnb_df)
    summary.append({
        'metric': 'total_listings',
        'value': total_listings,
        'notes': 'Total number of Airbnb listings in the dataset'
    })
    
    # Listings successfully mapped to tracts
    mapped_listings = airbnb_with_tracts.dropna(subset=['index_right']).shape[0]
    mapping_success_rate = (mapped_listings / total_listings * 100).round(1)
    summary.append({
        'metric': 'mapped_listings',
        'value': mapped_listings,
        'notes': f'Listings successfully mapped to census tracts ({mapping_success_rate}% of total)'
    })
    
    # Tract coverage
    tracts_with_listings = tract_metrics.shape[0]
    summary.append({
        'metric': 'tracts_with_listings',
        'value': tracts_with_listings,
        'notes': 'Number of census tracts with at least one Airbnb listing'
    })
    
    # Density statistics
    avg_density = tract_metrics['listing_density'].mean().round(2)
    median_density = tract_metrics['listing_density'].median().round(2)
    max_density = tract_metrics['listing_density'].max().round(2)
    summary.append({
        'metric': 'avg_listing_density',
        'value': avg_density,
        'notes': 'Average Airbnb listing density (listings per sq km) across tracts with listings'
    })
    summary.append({
        'metric': 'median_listing_density',
        'value': median_density,
        'notes': 'Median Airbnb listing density (listings per sq km)'
    })
    summary.append({
        'metric': 'max_listing_density',
        'value': max_density,
        'notes': 'Maximum Airbnb listing density (listings per sq km)'
    })
    
    # Property type statistics
    if 'is_entire_home' in airbnb_df.columns:
        entire_home_percent = (airbnb_df['is_entire_home'].sum() / total_listings * 100).round(1)
        summary.append({
            'metric': 'entire_home_percent',
            'value': entire_home_percent,
            'notes': 'Percentage of listings that are entire homes/apartments'
        })
    
    # Price statistics
    if 'price' in airbnb_df.columns:
        avg_price = airbnb_df['price'].mean().round(2)
        median_price = airbnb_df['price'].median().round(2)
        summary.append({
            'metric': 'avg_price',
            'value': avg_price,
            'notes': 'Average price per night (USD)'
        })
        summary.append({
            'metric': 'median_price',
            'value': median_price,
            'notes': 'Median price per night (USD)'
        })
    
    # Bedroom statistics
    if 'bedrooms' in airbnb_df.columns:
        avg_bedrooms = airbnb_df['bedrooms'].mean().round(2)
        bedroom_distribution = airbnb_df['bedrooms'].value_counts().sort_index().to_dict()
        summary.append({
            'metric': 'avg_bedrooms',
            'value': avg_bedrooms,
            'notes': 'Average number of bedrooms per listing'
        })
        summary.append({
            'metric': 'bedroom_distribution',
            'value': str(bedroom_distribution),
            'notes': 'Distribution of listings by bedroom count'
        })
    
    # High-density tracts
    high_density_threshold = tract_metrics['listing_density'].quantile(0.9).round(2)
    high_density_tracts = tract_metrics[tract_metrics['listing_density'] >= high_density_threshold].shape[0]
    summary.append({
        'metric': 'high_density_threshold',
        'value': high_density_threshold,
        'notes': '90th percentile threshold for high-density tracts (listings per sq km)'
    })
    summary.append({
        'metric': 'high_density_tracts',
        'value': high_density_tracts,
        'notes': 'Number of high-density tracts (≥90th percentile listing density)'
    })
    
    return pd.DataFrame(summary)

def main():
    """
    Main function to process Airbnb data and save results.
    """
    print("Starting Airbnb data processing...")
    
    # Load and clean Airbnb data
    airbnb_gdf = load_and_clean_airbnb_data(AIRBNB_RAW_FILE)
    if airbnb_gdf is None:
        print("Failed to load Airbnb data. Exiting.")
        return
    
    # Load census tract boundaries
    tracts_gdf = load_census_tracts(CENSUS_TRACTS_FILE)
    if tracts_gdf is None:
        print("Failed to load census tracts. Attempting to continue without spatial mapping...")
        # Save cleaned Airbnb data without tract information
        airbnb_df = airbnb_gdf.drop(columns=['geometry'])
        airbnb_df.to_csv(PROCESSED_AIRBNB_DIR / "airbnb_cleaned.csv", index=False)
        print(f"Saved cleaned Airbnb data to {PROCESSED_AIRBNB_DIR / 'airbnb_cleaned.csv'}")
        return
    
    # Map listings to census tracts
    airbnb_with_tracts = map_listings_to_tracts(airbnb_gdf, tracts_gdf)
    
    # Calculate metrics by census tract
    tract_metrics = calculate_tract_metrics(airbnb_with_tracts, tracts_gdf)
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(airbnb_gdf, airbnb_with_tracts, tract_metrics)
    
    # Save results
    print("Saving processed data...")
    
    # Save Airbnb data with tract information
    airbnb_export = airbnb_with_tracts.drop(columns=['geometry'])
    airbnb_export.to_csv(PROCESSED_AIRBNB_DIR / "airbnb_with_tracts.csv", index=False)
    print(f"Saved Airbnb data with tract information to {PROCESSED_AIRBNB_DIR / 'airbnb_with_tracts.csv'}")
    
    # Save tract metrics
    tract_metrics.to_csv(PROCESSED_AIRBNB_DIR / "airbnb_tract_density.csv", index=False)
    print(f"Saved tract metrics to {PROCESSED_AIRBNB_DIR / 'airbnb_tract_density.csv'}")
    
    # Save summary statistics
    summary_stats.to_csv(PROCESSED_AIRBNB_DIR / "summary_airbnb.csv", index=False)
    print(f"Saved summary statistics to {PROCESSED_AIRBNB_DIR / 'summary_airbnb.csv'}")
    
    print("Airbnb data processing complete!")

if __name__ == "__main__":
    main()
