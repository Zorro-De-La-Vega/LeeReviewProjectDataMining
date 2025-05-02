#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate Airbnb data by ZIP code and connect with census data.

Since we don't have the GeoJSON file for spatial joins, this script uses
ZIP codes to connect Airbnb listings with census data, calculating rental
density and other metrics at the ZIP code level instead of census tract level.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up file paths
PROJECT_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
AIRBNB_RAW_FILE = RAW_DATA_DIR / "airbnb" / "AirBnb-Data-MiamiData.csv"
CENSUS_FILE = RAW_DATA_DIR / "census" / "acs5_miamidade_tracts.csv"

# Create processed directories if they don't exist
PROCESSED_AIRBNB_DIR = PROCESSED_DATA_DIR / "airbnb"
COMBINED_DIR = PROCESSED_DATA_DIR / "combined"
os.makedirs(PROCESSED_AIRBNB_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

def load_and_clean_airbnb_data(file_path):
    """
    Load and clean Airbnb data, focusing on the ZIP code level.
    """
    print(f"Loading Airbnb data from {file_path}...")
    
    # Load data
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Successfully loaded data with {df.shape[0]} listings and {df.shape[1]} columns.")
    
    # Initial cleaning
    print("Cleaning Airbnb data...")
    
    # Filter to include only listings with valid ZIP codes
    df = df.dropna(subset=['Zipcode'])
    
    # Standardize ZIP code format (ensure 5 digits with leading zeros)
    df['Zipcode'] = df['Zipcode'].astype(str).str.zfill(5)
    
    # Convert price column
    if 'Average Daily Rate (USD)' in df.columns:
        df['price'] = pd.to_numeric(df['Average Daily Rate (USD)'], errors='coerce')
    
    # Handle bedrooms data
    if 'Bedrooms' in df.columns:
        # Convert 'Studio' to 0 if string type
        if df['Bedrooms'].dtype == 'object':
            df['Bedrooms'] = df['Bedrooms'].replace('Studio', '0')
        df['Bedrooms'] = pd.to_numeric(df['Bedrooms'], errors='coerce')
    
    # Extract property type information
    if 'Listing Type' in df.columns:
        df['is_entire_home'] = df['Listing Type'].str.contains('entire', case=False).fillna(False)
    
    # Remove listings with invalid prices or bedrooms
    df = df.dropna(subset=['price', 'Bedrooms'])
    
    print(f"Cleaned data has {df.shape[0]} listings")
    return df

def load_census_data(file_path):
    """
    Load census data with ZIP code information.
    """
    print(f"Loading census data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded census data with {df.shape[0]} records")
        
        # Check if ZIP code column exists
        zip_col = next((col for col in df.columns if 'zip' in col.lower()), None)
        
        if zip_col is None:
            print("Warning: No ZIP code column found in census data")
            # Create mapping based on tract IDs to ZIP codes (simplified approach)
            # This is a rough approximation as census tracts don't map 1:1 with ZIP codes
            census_zip_map = {}
            for tract_id in df['tract'].unique():
                # Extract first 5 digits for approximation
                census_zip_map[tract_id] = str(tract_id)[:5]
            
            df['zipcode'] = df['tract'].map(census_zip_map)
        
        return df
    
    except Exception as e:
        print(f"Error loading census data: {e}")
        return None

def aggregate_airbnb_by_zipcode(df):
    """
    Aggregate Airbnb data by ZIP code.
    """
    print("Aggregating Airbnb data by ZIP code...")
    
    # Group by ZIP code
    grouped = df.groupby('Zipcode').agg({
        'Listing Title': 'count',  # Count of listings
        'price': ['mean', 'median', 'std'],  # Price statistics
        'Bedrooms': ['mean', 'median'],  # Bedroom statistics
        'is_entire_home': 'mean',  # Percentage of entire homes
        'Max Guests': ['mean', 'median']  # Average guests per listing
    }).reset_index()
    
    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Rename columns for clarity
    grouped = grouped.rename(columns={
        'Listing Title_count': 'listing_count',
        'price_mean': 'mean_price',
        'price_median': 'median_price',
        'price_std': 'price_std',
        'Bedrooms_mean': 'mean_bedrooms',
        'Bedrooms_median': 'median_bedrooms',
        'is_entire_home_mean': 'entire_home_percent',
        'Max Guests_mean': 'mean_guests',
        'Max Guests_median': 'median_guests'
    })
    
    # Convert entire home percentage to actual percentage
    grouped['entire_home_percent'] = grouped['entire_home_percent'] * 100
    
    print(f"Created aggregation for {grouped.shape[0]} ZIP codes")
    return grouped

def join_airbnb_with_census(airbnb_zipcode, census_data):
    """
    Join Airbnb ZIP code aggregation with census data.
    """
    print("Joining Airbnb and census data by ZIP code...")
    
    if census_data is None:
        print("No census data available for joining")
        return airbnb_zipcode
    
    # Find ZIP code column in census data
    zip_col = next((col for col in census_data.columns if 'zip' in col.lower()), None)
    
    if zip_col is None:
        print("Cannot find ZIP code column in census data")
        return airbnb_zipcode
    
    # Standardize ZIP code format in census data
    census_data[zip_col] = census_data[zip_col].astype(str).str.zfill(5)
    
    # Merge datasets
    merged = pd.merge(
        airbnb_zipcode,
        census_data,
        left_on='Zipcode',
        right_on=zip_col,
        how='left'
    )
    
    print(f"Joined data has {merged.shape[0]} records")
    return merged

def calculate_density_metrics(joined_data):
    """
    Calculate Airbnb density metrics using population and housing units.
    """
    print("Calculating density metrics...")
    
    # Find population and housing unit columns
    pop_col = next((col for col in joined_data.columns if 'population' in col.lower()), None)
    housing_col = next((col for col in joined_data.columns if 'housing_units' in col.lower() or 'households' in col.lower()), None)
    
    if pop_col is not None:
        # Calculate listings per 1,000 residents
        joined_data['listings_per_1000_residents'] = (joined_data['listing_count'] / joined_data[pop_col] * 1000).round(2)
        print("Added listings per 1,000 residents metric")
    
    if housing_col is not None:
        # Calculate listings as percentage of housing units
        joined_data['listings_percent_of_housing'] = (joined_data['listing_count'] / joined_data[housing_col] * 100).round(2)
        print("Added listings as percentage of housing units metric")
    
    return joined_data

def create_airbnb_impact_score(data):
    """
    Create a composite impact score to measure the potential impact
    of Airbnb rentals on the local housing market.
    """
    print("Creating Airbnb impact score...")
    
    # Find necessary columns
    rent_col = next((col for col in data.columns if 'rent' in col.lower() and 'median' in col.lower()), None)
    income_col = next((col for col in data.columns if 'income' in col.lower() and 'median' in col.lower()), None)
    housing_col = next((col for col in data.columns if 'housing_units' in col.lower() or 'households' in col.lower()), None)
    
    # Initialize components for the impact score
    components = []
    
    # Component 1: Listing density (listings per housing unit)
    if housing_col is not None and 'listing_count' in data.columns:
        data['density_score'] = (data['listing_count'] / data[housing_col]).fillna(0)
        data['density_score'] = data['density_score'] / data['density_score'].max() * 100
        components.append('density_score')
    
    # Component 2: Price ratio (Airbnb price to median rent)
    if rent_col is not None and 'median_price' in data.columns:
        # Convert monthly rent to daily for comparison
        data['price_ratio'] = (data['median_price'] / (data[rent_col] / 30)).fillna(0)
        data['price_score'] = data['price_ratio'] / data['price_ratio'].max() * 100
        components.append('price_score')
    
    # Component 3: Affordability pressure (rent-to-income ratio)
    if rent_col is not None and income_col is not None:
        data['rent_to_income'] = (data[rent_col] * 12 / data[income_col] * 100).fillna(0)
        data['affordability_score'] = data['rent_to_income'] / data['rent_to_income'].max() * 100
        components.append('affordability_score')
    
    # Component 4: Entire home percentage
    if 'entire_home_percent' in data.columns:
        data['home_type_score'] = data['entire_home_percent'] / data['entire_home_percent'].max() * 100
        components.append('home_type_score')
    
    # Calculate composite score if we have at least 2 components
    if len(components) >= 2:
        data['airbnb_impact_score'] = data[components].mean(axis=1).round(1)
        
        # Classify impact levels
        data['impact_level'] = pd.cut(
            data['airbnb_impact_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
        
        print(f"Created impact score using {len(components)} components")
    else:
        print("Not enough data to create meaningful impact score")
    
    return data

def analyze_affordability_correlation(data):
    """
    Analyze correlation between Airbnb metrics and housing affordability.
    """
    print("Analyzing correlations with affordability...")
    
    # Find affordability-related columns
    affordability_cols = []
    for col in data.columns:
        if any(term in col.lower() for term in ['rent_to_income', 'price_to_income', 'affordability']):
            if data[col].dtype in [np.float64, np.int64]:
                affordability_cols.append(col)
    
    if not affordability_cols:
        print("No affordability metrics found for correlation analysis")
        return None
    
    # Airbnb metrics to correlate
    airbnb_metrics = ['listing_count', 'mean_price', 'median_price', 'entire_home_percent']
    airbnb_metrics = [m for m in airbnb_metrics if m in data.columns]
    
    if not airbnb_metrics:
        print("No Airbnb metrics found for correlation analysis")
        return None
    
    # Calculate correlations
    correlations = []
    
    for airbnb_metric in airbnb_metrics:
        for affordability_col in affordability_cols:
            mask = (~data[airbnb_metric].isna()) & (~data[affordability_col].isna())
            if mask.sum() < 5:  # Skip if too few data points
                continue
                
            from scipy import stats
            pearson_r, pearson_p = stats.pearsonr(data.loc[mask, airbnb_metric], data.loc[mask, affordability_col])
            spearman_r, spearman_p = stats.spearmanr(data.loc[mask, airbnb_metric], data.loc[mask, affordability_col])
            
            correlations.append({
                'airbnb_metric': airbnb_metric,
                'housing_variable': affordability_col,
                'pearson_r': round(pearson_r, 3),
                'pearson_p': round(pearson_p, 4),
                'spearman_r': round(spearman_r, 3),
                'spearman_p': round(spearman_p, 4),
                'significance': 'Significant' if pearson_p < 0.05 else 'Not significant',
                'direction': 'Positive' if pearson_r > 0 else 'Negative',
                'strength': 'Strong' if abs(pearson_r) > 0.5 else 'Moderate' if abs(pearson_r) > 0.3 else 'Weak',
                'sample_size': mask.sum()
            })
    
    correlation_df = pd.DataFrame(correlations)
    print(f"Calculated {len(correlations)} correlations between Airbnb and affordability metrics")
    return correlation_df

def generate_summary_statistics(airbnb_data, zipcode_data, correlation_results=None):
    """
    Generate summary statistics for the Airbnb dataset.
    """
    print("Generating summary statistics...")
    
    summary = []
    
    # Overall statistics
    total_listings = len(airbnb_data)
    summary.append({
        'metric': 'total_listings',
        'value': total_listings,
        'notes': 'Total number of Airbnb listings in the dataset'
    })
    
    # ZIP code coverage
    zipcodes_with_listings = zipcode_data.shape[0]
    summary.append({
        'metric': 'zipcodes_with_listings',
        'value': zipcodes_with_listings,
        'notes': 'Number of ZIP codes with at least one Airbnb listing'
    })
    
    # Listings statistics
    max_listings = zipcode_data['listing_count'].max()
    avg_listings = zipcode_data['listing_count'].mean().round(1)
    summary.append({
        'metric': 'max_listings_per_zipcode',
        'value': max_listings,
        'notes': 'Maximum number of listings in a single ZIP code'
    })
    summary.append({
        'metric': 'avg_listings_per_zipcode',
        'value': avg_listings,
        'notes': 'Average number of listings per ZIP code'
    })
    
    # Price statistics
    avg_price = airbnb_data['price'].mean().round(2)
    median_price = airbnb_data['price'].median().round(2)
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
    
    # Property type statistics
    entire_home_percent = (airbnb_data['is_entire_home'].mean() * 100).round(1)
    summary.append({
        'metric': 'entire_home_percent',
        'value': entire_home_percent,
        'notes': 'Percentage of listings that are entire homes/apartments'
    })
    
    # Impact score statistics if available
    if 'airbnb_impact_score' in zipcode_data.columns:
        avg_impact = zipcode_data['airbnb_impact_score'].mean().round(1)
        summary.append({
            'metric': 'avg_impact_score',
            'value': avg_impact,
            'notes': 'Average Airbnb impact score (0-100 scale)'
        })
        
        # Count by impact level
        if 'impact_level' in zipcode_data.columns:
            impact_counts = zipcode_data['impact_level'].value_counts()
            for level, count in impact_counts.items():
                summary.append({
                    'metric': f'{level}_impact_zipcodes',
                    'value': count,
                    'notes': f'Number of ZIP codes with {level} Airbnb impact'
                })
    
    # Correlation insights if available
    if correlation_results is not None and not correlation_results.empty:
        sig_correlations = correlation_results[correlation_results['pearson_p'] < 0.05]
        
        summary.append({
            'metric': 'significant_correlations',
            'value': len(sig_correlations),
            'notes': 'Number of significant correlations between Airbnb and affordability metrics'
        })
        
        # Add strongest correlations
        if not sig_correlations.empty:
            strongest_pos = sig_correlations.loc[sig_correlations['pearson_r'].idxmax()]
            summary.append({
                'metric': 'strongest_positive_correlation',
                'value': round(strongest_pos['pearson_r'], 3),
                'notes': f"Strongest positive correlation: {strongest_pos['airbnb_metric']} with {strongest_pos['housing_variable']}"
            })
            
            if any(sig_correlations['pearson_r'] < 0):
                strongest_neg = sig_correlations.loc[sig_correlations['pearson_r'].idxmin()]
                summary.append({
                    'metric': 'strongest_negative_correlation',
                    'value': round(strongest_neg['pearson_r'], 3),
                    'notes': f"Strongest negative correlation: {strongest_neg['airbnb_metric']} with {strongest_neg['housing_variable']}"
                })
    
    return pd.DataFrame(summary)

def main():
    """
    Main function to process Airbnb data by ZIP code.
    """
    print("Starting Airbnb data aggregation by ZIP code...")
    
    # Load and clean Airbnb data
    airbnb_data = load_and_clean_airbnb_data(AIRBNB_RAW_FILE)
    
    # Aggregate by ZIP code
    zipcode_agg = aggregate_airbnb_by_zipcode(airbnb_data)
    
    # Save the ZIP code aggregation
    zipcode_agg.to_csv(PROCESSED_AIRBNB_DIR / "airbnb_by_zipcode.csv", index=False)
    print(f"Saved ZIP code aggregation to {PROCESSED_AIRBNB_DIR / 'airbnb_by_zipcode.csv'}")
    
    # Load census data
    census_data = load_census_data(CENSUS_FILE)
    
    # Join with census data if available
    if census_data is not None:
        combined_data = join_airbnb_with_census(zipcode_agg, census_data)
        
        # Calculate density metrics
        combined_data = calculate_density_metrics(combined_data)
        
        # Create impact score
        combined_data = create_airbnb_impact_score(combined_data)
        
        # Analyze correlations
        correlation_results = analyze_affordability_correlation(combined_data)
        
        # Save combined data
        combined_data.to_csv(COMBINED_DIR / "airbnb_census_by_zipcode.csv", index=False)
        print(f"Saved combined data to {COMBINED_DIR / 'airbnb_census_by_zipcode.csv'}")
        
        # Save correlation results if available
        if correlation_results is not None:
            correlation_results.to_csv(COMBINED_DIR / "airbnb_housing_correlation.csv", index=False)
            print(f"Saved correlation results to {COMBINED_DIR / 'airbnb_housing_correlation.csv'}")
        
        # Generate summary statistics
        summary_stats = generate_summary_statistics(airbnb_data, combined_data, correlation_results)
    else:
        # Generate summary without census data
        summary_stats = generate_summary_statistics(airbnb_data, zipcode_agg)
    
    # Save summary statistics
    summary_stats.to_csv(PROCESSED_AIRBNB_DIR / "summary_airbnb.csv", index=False)
    print(f"Saved summary statistics to {PROCESSED_AIRBNB_DIR / 'summary_airbnb.csv'}")
    
    print("Airbnb data processing complete!")

if __name__ == "__main__":
    main()
