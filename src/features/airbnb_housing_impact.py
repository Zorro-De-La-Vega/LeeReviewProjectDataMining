#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze the impact of short-term rentals on housing affordability in Miami-Dade County.

This script combines the processed Airbnb data with census and rental data
to evaluate how short-term rental density correlates with housing affordability metrics.
It generates visualizations and statistical tests to quantify these relationships.

Output files:
- airbnb_housing_correlation.csv: Correlation metrics between Airbnb density and housing variables
- tract_combined_metrics.csv: Combined dataset with census, rental, and Airbnb metrics by tract
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
from scipy import stats

# Set up file paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
AIRBNB_DIR = PROCESSED_DATA_DIR / "airbnb"
CENSUS_DIR = PROCESSED_DATA_DIR / "census"
RENTAL_DIR = PROCESSED_DATA_DIR / "rental"
VISUALIZATION_DIR = PROJECT_DIR / "visualizations" / "airbnb_impact"

# Create output directories if they don't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR / "combined", exist_ok=True)

# Set visualization style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)
colors = sns.color_palette("Set2", 8)

def load_datasets():
    """
    Load processed datasets for analysis.
    
    Returns:
        tuple: (airbnb_data, airbnb_tract_density, census_data, rental_data)
    """
    print("Loading processed datasets...")
    
    # Load Airbnb data with tract information
    airbnb_file = AIRBNB_DIR / "airbnb_with_tracts.csv"
    if os.path.exists(airbnb_file):
        airbnb_data = pd.read_csv(airbnb_file)
        print(f"Loaded Airbnb data with {airbnb_data.shape[0]} listings")
    else:
        print(f"Warning: {airbnb_file} not found")
        airbnb_data = None
    
    # Load Airbnb tract density metrics
    density_file = AIRBNB_DIR / "airbnb_tract_density.csv"
    if os.path.exists(density_file):
        airbnb_tract_density = pd.read_csv(density_file)
        print(f"Loaded Airbnb density data for {airbnb_tract_density.shape[0]} census tracts")
    else:
        print(f"Warning: {density_file} not found")
        airbnb_tract_density = None
    
    # Load census data
    census_file = CENSUS_DIR / "cleaned_acs5_miamidade_tracts.csv"
    if os.path.exists(census_file):
        census_data = pd.read_csv(census_file)
        print(f"Loaded census data for {census_data.shape[0]} census tracts")
    else:
        # Try alternative file
        census_file = PROJECT_DIR / "data" / "raw" / "census" / "acs5_miamidade_tracts.csv"
        if os.path.exists(census_file):
            census_data = pd.read_csv(census_file)
            print(f"Loaded raw census data for {census_data.shape[0]} census tracts")
        else:
            print(f"Warning: No census data found")
            census_data = None
    
    # Load rental data
    rental_files = list(RENTAL_DIR.glob("*.csv"))
    if rental_files:
        # Check for processed rental data by census tract if it exists
        tract_rental_files = [f for f in rental_files if "tract" in f.name.lower()]
        if tract_rental_files:
            rental_data = pd.read_csv(tract_rental_files[0])
            print(f"Loaded rental data by census tract from {tract_rental_files[0].name}")
        else:
            # Use any rental data available
            rental_data = pd.read_csv(rental_files[0])
            print(f"Loaded rental data from {rental_files[0].name}")
    else:
        print("Warning: No rental data found")
        rental_data = None
    
    return airbnb_data, airbnb_tract_density, census_data, rental_data

def merge_datasets(airbnb_tract_density, census_data, rental_data):
    """
    Merge Airbnb, census, and rental datasets by census tract.
    
    Args:
        airbnb_tract_density: DataFrame with Airbnb metrics by census tract
        census_data: DataFrame with census data by tract
        rental_data: DataFrame with rental data
        
    Returns:
        pandas.DataFrame: Combined dataset with all metrics by census tract
    """
    print("Merging datasets by census tract...")
    
    # Handle case where some datasets are missing
    if airbnb_tract_density is None or census_data is None:
        print("Error: Required datasets are missing. Cannot merge.")
        return None
    
    # Identify the tract ID column in each dataset
    tract_cols = {
        'airbnb': next((col for col in airbnb_tract_density.columns 
                       if 'tract' in col.lower() or 'geoid' in col.lower()), None),
        'census': next((col for col in census_data.columns 
                       if 'tract' in col.lower() or 'geoid' in col.lower()), None)
    }
    
    if tract_cols['airbnb'] is None or tract_cols['census'] is None:
        print("Error: Cannot identify tract ID columns in datasets.")
        # Use position-based guess if column names aren't clear
        tract_cols['airbnb'] = airbnb_tract_density.columns[0]
        tract_cols['census'] = 'tract'
        print(f"Using {tract_cols['airbnb']} for Airbnb and {tract_cols['census']} for census as tract identifiers.")
    
    # Prepare census data
    census_subset = census_data.copy()
    
    # Merge Airbnb and census data
    merged = pd.merge(
        airbnb_tract_density,
        census_subset,
        left_on=tract_cols['airbnb'],
        right_on=tract_cols['census'],
        how='right'
    )
    
    # Fill NaN values for tracts without Airbnb listings
    for col in airbnb_tract_density.columns:
        if col != tract_cols['airbnb'] and col in merged.columns:
            merged[col] = merged[col].fillna(0)
    
    # Add rental data if available
    if rental_data is not None and isinstance(rental_data, pd.DataFrame):
        # Try to find a tract column in rental data
        rental_tract_col = next((col for col in rental_data.columns 
                               if 'tract' in col.lower() or 'geoid' in col.lower()), None)
        
        if rental_tract_col is not None:
            # Merge with rental data
            merged = pd.merge(
                merged,
                rental_data,
                left_on=tract_cols['census'],
                right_on=rental_tract_col,
                how='left'
            )
    
    print(f"Successfully merged datasets. Final dataset has {merged.shape[0]} census tracts and {merged.shape[1]} columns.")
    return merged

def calculate_affordability_metrics(combined_data):
    """
    Calculate housing affordability metrics for each census tract.
    
    Args:
        combined_data: DataFrame with combined census, Airbnb, and rental data
        
    Returns:
        pandas.DataFrame: Dataset with additional affordability metrics
    """
    print("Calculating housing affordability metrics...")
    
    df = combined_data.copy()
    
    # Identify relevant columns
    income_col = next((col for col in df.columns if 'median_income' in col.lower()), None)
    rent_col = next((col for col in df.columns if 'median_rent' in col.lower() or 'median_gross_rent' in col.lower()), None)
    home_value_col = next((col for col in df.columns if 'median_home_value' in col.lower() or 'median_house_value' in col.lower()), None)
    
    if income_col is None or rent_col is None:
        print("Warning: Cannot find income or rent columns. Skipping affordability calculations.")
        return df
    
    # Calculate rent-to-income ratio (standard affordability metric)
    # Monthly rent as a percentage of monthly income
    # 30% is the standard threshold for housing affordability
    df['annual_rent'] = df[rent_col] * 12
    df['rent_to_income_ratio'] = (df['annual_rent'] / df[income_col] * 100).round(1)
    
    # Classify tracts by affordability
    df['affordability_status'] = pd.cut(
        df['rent_to_income_ratio'],
        bins=[0, 20, 30, 40, 100],
        labels=['Very Affordable', 'Affordable', 'Cost-Burdened', 'Severely Cost-Burdened'],
        right=True
    )
    
    # Calculate price-to-income ratio for home ownership affordability
    if home_value_col is not None:
        df['price_to_income_ratio'] = (df[home_value_col] / df[income_col]).round(2)
        
        # Standard threshold is 3-4x annual income for affordable home purchase
        df['homeownership_affordability'] = pd.cut(
            df['price_to_income_ratio'],
            bins=[0, 3, 5, 7, 100],
            labels=['Very Affordable', 'Affordable', 'Expensive', 'Very Expensive'],
            right=True
        )
    
    print("Affordability metrics calculated.")
    return df

def analyze_airbnb_impact(combined_data_with_metrics):
    """
    Analyze the relationship between Airbnb density and housing affordability.
    
    Args:
        combined_data_with_metrics: DataFrame with combined data and affordability metrics
        
    Returns:
        pandas.DataFrame: Correlation analysis results
    """
    print("Analyzing Airbnb impact on housing affordability...")
    
    df = combined_data_with_metrics.copy()
    
    # Identify Airbnb density column
    density_col = next((col for col in df.columns if 'density' in col.lower() and 'listing' in col.lower()), 'listing_density')
    
    # Define housing variables to analyze
    housing_vars = []
    for col in ['rent_to_income_ratio', 'median_rent', 'median_home_value', 'price_to_income_ratio', 
                'median_income', 'vacancy_rate', 'median_gross_rent']:
        matching_cols = [c for c in df.columns if col.lower() in c.lower()]
        housing_vars.extend(matching_cols)
    
    # Remove duplicates and ensure columns exist in the DataFrame
    housing_vars = list(set([col for col in housing_vars if col in df.columns]))
    
    if not housing_vars:
        print("Error: No relevant housing variables found for analysis.")
        return None
    
    # Calculate correlations
    correlations = []
    
    for var in housing_vars:
        # Calculate Pearson correlation
        mask = (~df[density_col].isna()) & (~df[var].isna())
        if mask.sum() < 10:
            # Skip if we don't have enough data points
            continue
            
        pearson_r, pearson_p = stats.pearsonr(df.loc[mask, density_col], df.loc[mask, var])
        
        # Calculate Spearman rank correlation (less sensitive to outliers)
        spearman_r, spearman_p = stats.spearmanr(df.loc[mask, density_col], df.loc[mask, var])
        
        correlations.append({
            'housing_variable': var,
            'pearson_r': round(pearson_r, 3),
            'pearson_p': round(pearson_p, 4),
            'spearman_r': round(spearman_r, 3),
            'spearman_p': round(spearman_p, 4),
            'significance': 'Significant' if pearson_p < 0.05 else 'Not significant',
            'direction': 'Positive' if pearson_r > 0 else 'Negative',
            'strength': abs(pearson_r) > 0.3,
            'sample_size': mask.sum()
        })
    
    # Regression analysis for key variables
    key_vars = ['rent_to_income_ratio', 'median_rent', 'median_home_value']
    key_var_cols = [col for col in housing_vars if any(key_var in col.lower() for key_var in key_vars)]
    
    for var in key_var_cols:
        mask = (~df[density_col].isna()) & (~df[var].isna())
        if mask.sum() < 10:
            continue
            
        # Simple linear regression
        X = sm.add_constant(df.loc[mask, density_col])
        y = df.loc[mask, var]
        
        try:
            model = sm.OLS(y, X).fit()
            
            # Extract coefficient, p-value, and R-squared
            coef = model.params[1]
            p_value = model.pvalues[1]
            r_squared = model.rsquared
            
            # Find the correlation entry to update
            for i, corr in enumerate(correlations):
                if corr['housing_variable'] == var:
                    correlations[i].update({
                        'regression_coefficient': round(coef, 4),
                        'regression_p_value': round(p_value, 4),
                        'r_squared': round(r_squared, 3),
                        'regression_significance': 'Significant' if p_value < 0.05 else 'Not significant'
                    })
                    break
        except:
            print(f"Warning: Regression analysis failed for {var}")
    
    print(f"Analyzed impact on {len(correlations)} housing variables.")
    return pd.DataFrame(correlations)

def classify_neighborhoods(combined_data_with_metrics):
    """
    Classify neighborhoods based on Airbnb density and housing affordability.
    
    Args:
        combined_data_with_metrics: DataFrame with combined data and metrics
        
    Returns:
        pandas.DataFrame: Dataset with neighborhood classifications
    """
    print("Classifying neighborhoods by Airbnb impact...")
    
    df = combined_data_with_metrics.copy()
    
    # Identify necessary columns
    density_col = next((col for col in df.columns if 'density' in col.lower() and 'listing' in col.lower()), 'listing_density')
    
    # Find affordability ratio column
    ratio_col = next((col for col in df.columns if 'rent_to_income' in col.lower()), None)
    if ratio_col is None:
        print("Warning: Rent-to-income ratio not found. Using median rent for classification.")
        ratio_col = next((col for col in df.columns if 'median_rent' in col.lower() or 'median_gross_rent' in col.lower()), None)
    
    if ratio_col is None:
        print("Error: Cannot find affordability metrics for classification.")
        return df
    
    # Create density quintiles
    df['airbnb_density_quintile'] = pd.qcut(
        df[density_col].clip(lower=0), 
        q=5, 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Create affordability quintiles
    df['affordability_quintile'] = pd.qcut(
        df[ratio_col].clip(lower=0), 
        q=5, 
        labels=['Most Affordable', 'More Affordable', 'Average', 'Less Affordable', 'Least Affordable']
    )
    
    # Create composite neighborhood classification
    df['neighborhood_classification'] = df.apply(
        lambda row: f"{row['airbnb_density_quintile']} Airbnb Density, {row['affordability_quintile']}",
        axis=1
    )
    
    # Identify high-impact neighborhoods (high Airbnb density, low affordability)
    high_impact_mask = (df['airbnb_density_quintile'].isin(['High', 'Very High'])) & \
                       (df['affordability_quintile'].isin(['Less Affordable', 'Least Affordable']))
    
    df['high_impact_area'] = high_impact_mask
    
    print(f"Identified {high_impact_mask.sum()} high-impact neighborhoods (high Airbnb density with low affordability).")
    return df

def generate_impact_summary(classified_data, correlation_results):
    """
    Generate a summary of Airbnb impact on housing affordability.
    
    Args:
        classified_data: DataFrame with neighborhood classifications
        correlation_results: DataFrame with correlation analysis results
        
    Returns:
        pandas.DataFrame: Summary statistics and findings
    """
    print("Generating impact summary...")
    
    summary = []
    
    # Count tracts by Airbnb density
    density_counts = classified_data['airbnb_density_quintile'].value_counts()
    for category, count in density_counts.items():
        summary.append({
            'metric': f'tracts_with_{category.lower().replace(" ", "_")}_density',
            'value': count,
            'notes': f'Number of census tracts with {category.lower()} Airbnb density'
        })
    
    # Count high-impact areas
    high_impact_count = classified_data['high_impact_area'].sum()
    summary.append({
        'metric': 'high_impact_tracts',
        'value': high_impact_count,
        'notes': 'Number of census tracts with both high Airbnb density and low affordability'
    })
    summary.append({
        'metric': 'high_impact_percent',
        'value': round(high_impact_count / len(classified_data) * 100, 1),
        'notes': 'Percentage of census tracts classified as high-impact areas'
    })
    
    # Summarize correlations
    if correlation_results is not None and not correlation_results.empty:
        sig_correlations = correlation_results[correlation_results['pearson_p'] < 0.05]
        
        summary.append({
            'metric': 'significant_correlations',
            'value': len(sig_correlations),
            'notes': 'Number of housing variables with statistically significant correlation to Airbnb density'
        })
        
        # Add strongest correlations
        if not sig_correlations.empty:
            strongest_pos = sig_correlations.loc[sig_correlations['pearson_r'].idxmax()]
            summary.append({
                'metric': 'strongest_positive_correlation',
                'value': round(strongest_pos['pearson_r'], 3),
                'notes': f"Strongest positive correlation with {strongest_pos['housing_variable']} (p={strongest_pos['pearson_p']})"
            })
            
            strongest_neg = sig_correlations.loc[sig_correlations['pearson_r'].idxmin()]
            summary.append({
                'metric': 'strongest_negative_correlation',
                'value': round(strongest_neg['pearson_r'], 3),
                'notes': f"Strongest negative correlation with {strongest_neg['housing_variable']} (p={strongest_neg['pearson_p']})"
            })
    
    # Affordability metrics
    if 'rent_to_income_ratio' in classified_data.columns:
        # Average rent-to-income ratio by Airbnb density quintile
        affordability_by_density = classified_data.groupby('airbnb_density_quintile')['rent_to_income_ratio'].mean().round(1)
        
        for density, ratio in affordability_by_density.items():
            summary.append({
                'metric': f'avg_rent_to_income_{density.lower().replace(" ", "_")}',
                'value': ratio,
                'notes': f'Average rent-to-income ratio in tracts with {density.lower()} Airbnb density'
            })
    
    # Difference between highest and lowest density areas
    try:
        highest_density = affordability_by_density['Very High']
        lowest_density = affordability_by_density['Very Low']
        difference = highest_density - lowest_density
        
        summary.append({
            'metric': 'affordability_gap',
            'value': round(difference, 1),
            'notes': 'Percentage point difference in rent-to-income ratio between areas with very high vs. very low Airbnb density'
        })
    except:
        pass
    
    print("Impact summary generated.")
    return pd.DataFrame(summary)

def main():
    """
    Main function to analyze Airbnb impact on housing affordability.
    """
    print("Starting analysis of Airbnb impact on housing affordability...")
    
    # Load datasets
    airbnb_data, airbnb_tract_density, census_data, rental_data = load_datasets()
    
    # Merge datasets
    combined_data = merge_datasets(airbnb_tract_density, census_data, rental_data)
    if combined_data is None:
        print("Error: Failed to merge datasets. Analysis cannot continue.")
        return
    
    # Calculate affordability metrics
    combined_data_with_metrics = calculate_affordability_metrics(combined_data)
    
    # Analyze Airbnb impact
    correlation_results = analyze_airbnb_impact(combined_data_with_metrics)
    
    # Classify neighborhoods
    classified_data = classify_neighborhoods(combined_data_with_metrics)
    
    # Generate impact summary
    impact_summary = generate_impact_summary(classified_data, correlation_results)
    
    # Save results
    print("Saving analysis results...")
    
    # Save combined dataset
    combined_output_path = PROCESSED_DATA_DIR / "combined" / "tract_combined_metrics.csv"
    classified_data.to_csv(combined_output_path, index=False)
    print(f"Saved combined metrics to {combined_output_path}")
    
    # Save correlation results
    if correlation_results is not None:
        correlation_output_path = PROCESSED_DATA_DIR / "combined" / "airbnb_housing_correlation.csv"
        correlation_results.to_csv(correlation_output_path, index=False)
        print(f"Saved correlation results to {correlation_output_path}")
    
    # Save impact summary
    impact_output_path = PROCESSED_DATA_DIR / "combined" / "airbnb_impact_summary.csv"
    impact_summary.to_csv(impact_output_path, index=False)
    print(f"Saved impact summary to {impact_output_path}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
