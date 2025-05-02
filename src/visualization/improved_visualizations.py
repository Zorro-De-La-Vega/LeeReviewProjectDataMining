"""
Script: improved_visualizations.py
Purpose: Generate improved, highly informative visualizations for the Miami-Dade project.
Outputs: visualizations/improved_*.png

Includes:
- Top/bottom neighborhood analysis
- Rent-to-income ratio analysis (affordability metric)
- Crime trend visualization
- Correlation analysis
- Outlier detection
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import math

# Ensure visualizations directory exists
VIS_DIR = 'visualizations'
os.makedirs(VIS_DIR, exist_ok=True)

# Utility functions
def format_currency(x, pos):
    """Format as currency with K/M for thousands/millions."""
    if abs(x) >= 1_000_000:
        return f'${x/1_000_000:.1f}M'
    elif abs(x) >= 1_000:
        return f'${x/1_000:.0f}K'
    else:
        return f'${x:.0f}'

def clean_outliers(df, column, min_val=None, max_val=None):
    """Remove extreme outliers for better visualization."""
    clean_df = df.copy()
    # Filter out negative values and obvious errors (like -666666666)
    clean_df = clean_df[clean_df[column] > 0]
    
    if min_val is not None:
        clean_df = clean_df[clean_df[column] >= min_val]
    if max_val is not None:
        clean_df = clean_df[clean_df[column] <= max_val]
    return clean_df

# 1. Top and Bottom 10 Tracts by Median Income (with better labels)
def plot_income_extremes(census_path):
    df = pd.read_csv(census_path)
    # Remove invalid income values
    df = clean_outliers(df, 'B19013_001E')
    
    # Get top and bottom 10 tracts
    top10 = df.nlargest(10, 'B19013_001E')
    bottom10 = df.nsmallest(10, 'B19013_001E')
    
    # Plot top 10
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='B19013_001E', y='tract', data=top10, palette='viridis', ax=ax)
    ax.set_title('Top 10 Census Tracts by Median Household Income', fontsize=14)
    ax.set_xlabel('Median Household Income', fontsize=12)
    ax.set_ylabel('Census Tract', fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_top10_income_tracts.png'))
    plt.close()
    
    # Plot bottom 10
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='B19013_001E', y='tract', data=bottom10, palette='viridis', ax=ax)
    ax.set_title('Bottom 10 Census Tracts by Median Household Income', fontsize=14)
    ax.set_xlabel('Median Household Income', fontsize=12)
    ax.set_ylabel('Census Tract', fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(format_currency))
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_bottom10_income_tracts.png'))
    plt.close()

# 2. Distribution of Rent-to-Income Ratio (affordability metric)
def plot_affordability(census_path):
    df = pd.read_csv(census_path)
    # Clean data - remove outliers and invalid values
    df = clean_outliers(df, 'B19013_001E')
    df = clean_outliers(df, 'B25064_001E')
    
    # Calculate annual rent and rent-to-income ratio
    df['annual_rent'] = df['B25064_001E'] * 12
    df['rent_to_income_ratio'] = df['annual_rent'] / df['B19013_001E']
    
    # Filter out extreme ratios for better visualization
    df = df[(df['rent_to_income_ratio'] > 0) & (df['rent_to_income_ratio'] < 1)]
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df['rent_to_income_ratio'], bins=30, kde=True, color='purple')
    plt.axvline(x=0.3, color='red', linestyle='--')  # 30% is a common affordability threshold
    plt.text(0.31, plt.gca().get_ylim()[1]*0.9, 'Affordability threshold (30%)', 
             color='red', fontsize=12)
    plt.title('Distribution of Rent-to-Income Ratio by Census Tract\n(Annual Rent as % of Median Household Income)', 
              fontsize=14)
    plt.xlabel('Rent-to-Income Ratio (Annual Rent / Median Household Income)', fontsize=12)
    plt.ylabel('Number of Census Tracts', fontsize=12)
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0%}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_rent_to_income_ratio.png'))
    plt.close()

# 3. Scatter Plot with Rent vs. Income Colored by Affordability
def plot_income_rent_affordability(census_path):
    df = pd.read_csv(census_path)
    # Clean data
    df = clean_outliers(df, 'B19013_001E', min_val=10000, max_val=250000)
    df = clean_outliers(df, 'B25064_001E', min_val=500, max_val=3500)
    
    # Calculate annual rent and rent-to-income ratio
    df['annual_rent'] = df['B25064_001E'] * 12
    df['rent_to_income_ratio'] = df['annual_rent'] / df['B19013_001E']
    
    # Create affordability categories
    df['affordability'] = pd.cut(
        df['rent_to_income_ratio'],
        bins=[0, 0.2, 0.3, 0.4, 1],
        labels=['Affordable', 'Moderately Affordable', 'Expensive', 'Severely Unaffordable']
    )
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='B19013_001E', 
        y='B25064_001E', 
        hue='affordability',
        palette=['green', 'yellow', 'orange', 'red'],
        size='B01003_001E',  # Population as dot size
        sizes=(20, 200),
        alpha=0.7,
        data=df
    )
    
    plt.title('Median Income vs. Median Rent by Census Tract\nColored by Affordability, Sized by Population', fontsize=14)
    plt.xlabel('Median Household Income ($)', fontsize=12)
    plt.ylabel('Median Monthly Gross Rent ($)', fontsize=12)
    
    # Format x-axis as currency
    scatter.xaxis.set_major_formatter(FuncFormatter(format_currency))
    scatter.yaxis.set_major_formatter(FuncFormatter(format_currency))
    
    plt.legend(title="Affordability\n(Annual Rent/Income)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_income_rent_affordability.png'))
    plt.close()

# 4. Home Value Distribution Cleaned of Outliers
def plot_home_value_distribution(census_path):
    df = pd.read_csv(census_path)
    # Clean data
    df = clean_outliers(df, 'B25077_001E', min_val=50000, max_val=2000000)
    
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df['B25077_001E'], bins=30, kde=True, color='darkblue')
    plt.title('Distribution of Median Home Value by Census Tract', fontsize=14)
    plt.xlabel('Median Home Value ($)', fontsize=12)
    plt.ylabel('Number of Census Tracts', fontsize=12)
    
    # Format x-axis as currency
    ax.xaxis.set_major_formatter(FuncFormatter(format_currency))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_home_value_distribution.png'))
    plt.close()

# 5. Correlation Matrix with Better Labels
def plot_improved_correlation(census_path):
    df = pd.read_csv(census_path)
    # Clean data
    for col in ['B19013_001E', 'B25064_001E', 'B25077_001E']:
        df = clean_outliers(df, col)
    
    # Select and rename columns for readability
    columns_map = {
        'B19013_001E': 'Median Income',
        'B01003_001E': 'Population',
        'B25064_001E': 'Median Rent',
        'B25077_001E': 'Median Home Value', 
        'B25002_003E': 'Vacant Units'
    }
    
    corr_df = df[columns_map.keys()].rename(columns=columns_map)
    corr = corr_df.corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt='.2f', 
        cmap=cmap, 
        vmin=-1, 
        vmax=1,
        mask=mask,
        linewidths=.5, 
        cbar_kws={"shrink": .8},
        square=True
    )
    
    plt.title('Correlation Between Key Census Variables', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_correlation_heatmap.png'))
    plt.close()

# 6. Crime Analysis Visualization
def plot_crime_analysis(crime_path):
    # Handle potential CSV parsing issues with commas in fields
    try:
        df = pd.read_csv(crime_path)
    except pd.errors.ParserError:
        # If parsing error occurs, try with different parameters
        df = pd.read_csv(crime_path, quoting=1, on_bad_lines='skip')
    
    # Filter to only the actual crime types (not subtotals)
    crimes = df[~df['Crime Type'].str.contains('TOTAL') & ~df['UCR Code Description'].isna()]
    
    # Prepare data for plotting
    crimes_summary = crimes.groupby('Crime Type')[['JAN - APR 2024', 'JAN - APR 2025']].sum().reset_index()
    crimes_summary = crimes_summary.sort_values('JAN - APR 2025', ascending=False)
    
    # Calculate percent change for each crime type
    crimes_summary['Percent Change'] = ((crimes_summary['JAN - APR 2025'] - crimes_summary['JAN - APR 2024']) / 
                                       crimes_summary['JAN - APR 2024'] * 100)
    
    # Plot crime counts
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Crime Type', y='JAN - APR 2025', data=crimes_summary, palette='Reds')
    
    # Add data labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 20, int(height), 
               ha='center', fontsize=10)
    
    plt.title('2025 Crime Counts by Type (Jan-Apr)', fontsize=16)
    plt.xlabel('Crime Type', fontsize=14)
    plt.ylabel('Number of Crimes', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_crime_counts.png'))
    plt.close()
    
    # Plot percent change
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Crime Type', y='Percent Change', data=crimes_summary, 
                    palette=['green' if x < 0 else 'red' for x in crimes_summary['Percent Change']])
    
    # Add data labels
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        if math.isnan(height):
            continue
        ax.text(p.get_x() + p.get_width()/2., 
               height + 1 if height >= 0 else height - 3, 
               f'{height:.1f}%', 
               ha='center', fontsize=10)
    
    plt.title('Percent Change in Crime (2024 to 2025, Jan-Apr)', fontsize=16)
    plt.xlabel('Crime Type', fontsize=14)
    plt.ylabel('Percent Change', fontsize=14)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_crime_percent_change.png'))
    plt.close()

# 7. Rent by Bedroom with Year-over-Year Change
def plot_improved_rent_by_bedroom(rent_bedroom_path):
    df = pd.read_csv(rent_bedroom_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot average rent bars
    bars = ax1.bar(df['Bedroom Type'], df['Average Rent (USD)'], color='skyblue')
    ax1.set_ylabel('Average Rent (USD)', color='steelblue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_title('Average Rent by Bedroom Type with Annual Change', fontsize=14)
    
    # Add rent values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100, f'${int(height)}', 
                ha='center', va='bottom', color='steelblue', fontweight='bold')
    
    # Create second y-axis for percentage change
    ax2 = ax1.twinx()
    
    # Convert percentage strings to float values
    df['Change Last Year'] = df['Change Last Year'].replace('No Change', '0%')
    df['Change Last Year'] = df['Change Last Year'].str.rstrip('%').astype(float)
    
    # Plot annual percentage change line
    ax2.plot(df['Bedroom Type'], df['Change Last Year'], color='red', marker='o', linewidth=2)
    ax2.set_ylabel('Annual Change (%)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add percentage annotations
    for i, val in enumerate(df['Change Last Year']):
        ax2.annotate(f'{val}%', 
                    (i, val),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    color='red',
                    fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_rent_by_bedroom.png'))
    plt.close()

# 8. Home Value vs. Vacancy Rate
def plot_home_value_vs_vacancy(census_path):
    df = pd.read_csv(census_path)
    # Clean data
    df = clean_outliers(df, 'B25077_001E', min_val=50000, max_val=2000000)
    df = clean_outliers(df, 'B25002_003E', max_val=2000)
    df = df[df['B01003_001E'] > 0]  # Ensure population > 0
    
    # Calculate vacancy rate (vacant units / population)
    df['vacancy_rate'] = df['B25002_003E'] / df['B01003_001E']
    
    # Filter for visualization
    df = df[df['vacancy_rate'] <= 1]  # Ensure rate is reasonable
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='B25077_001E', 
        y='vacancy_rate', 
        size='B01003_001E',
        sizes=(20, 200),
        alpha=0.6,
        palette='viridis',
        data=df
    )
    
    plt.title('Median Home Value vs. Vacancy Rate by Census Tract', fontsize=14)
    plt.xlabel('Median Home Value ($)', fontsize=12)
    plt.ylabel('Vacancy Rate (Vacant Units per Resident)', fontsize=12)
    
    # Format x-axis as currency
    scatter.xaxis.set_major_formatter(FuncFormatter(format_currency))
    # Format y-axis as percentage
    scatter.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1%}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'improved_home_value_vs_vacancy.png'))
    plt.close()

if __name__ == "__main__":
    # File paths
    census_path = 'data/raw/census/acs5_miamidade_tracts.csv'
    rent_bedroom_path = 'data/processed/rental/avg_rent_by_bedroom.csv'
    rent_property_path = 'data/processed/rental/avg_rent_by_property_type.csv'
    crime_path = 'data/raw/crime/miami_dade_part1_crimes_ytd_comparison_2024_2025.csv'
    
    # Generate improved visualizations
    plot_income_extremes(census_path)
    plot_affordability(census_path)
    plot_income_rent_affordability(census_path)
    plot_home_value_distribution(census_path)
    plot_improved_correlation(census_path)
    plot_crime_analysis(crime_path)
    plot_improved_rent_by_bedroom(rent_bedroom_path)
    plot_home_value_vs_vacancy(census_path)
    
    print("Improved visualizations have been generated in the 'visualizations' folder.")
