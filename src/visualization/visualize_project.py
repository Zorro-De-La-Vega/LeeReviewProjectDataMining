"""
Visualization Script: visualize_project.py
Generates 10 key visualizations from cleaned and granular Miami-Dade data.
Outputs: visualizations/*.png

Requires: pip install pandas matplotlib seaborn geopandas
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: for mapping
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

VIS_DIR = 'visualizations'
os.makedirs(VIS_DIR, exist_ok=True)

# 1. Distribution of Median Household Income by Census Tract
def plot_income_distribution(census_path):
    df = pd.read_csv(census_path)
    plt.figure(figsize=(8,5))
    sns.histplot(df['B19013_001E'], bins=30, kde=True)
    plt.title('Distribution of Median Household Income by Tract')
    plt.xlabel('Median Household Income ($)')
    plt.ylabel('Number of Tracts')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'income_distribution.png'))
    plt.close()

# 2. Distribution of Median Gross Rent by Census Tract
def plot_rent_distribution(census_path):
    df = pd.read_csv(census_path)
    plt.figure(figsize=(8,5))
    sns.histplot(df['B25064_001E'], bins=30, kde=True, color='orange')
    plt.title('Distribution of Median Gross Rent by Tract')
    plt.xlabel('Median Gross Rent ($)')
    plt.ylabel('Number of Tracts')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'rent_distribution.png'))
    plt.close()

# 3. Map: Median Household Income by Tract (Choropleth)
def plot_income_map(census_path, shapefile_path=None):
    if not HAS_GEOPANDAS or shapefile_path is None:
        print('Geopandas or shapefile not available, skipping map.')
        return
    df = pd.read_csv(census_path)
    gdf = gpd.read_file(shapefile_path)
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    merged = gdf.merge(df, on='GEOID')
    ax = merged.plot(column='B19013_001E', cmap='viridis', legend=True, figsize=(10,8), edgecolor='k')
    ax.set_title('Median Household Income by Tract')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'income_choropleth.png'))
    plt.close()

# 4. Map: Median Gross Rent by Tract (Choropleth)
def plot_rent_map(census_path, shapefile_path=None):
    if not HAS_GEOPANDAS or shapefile_path is None:
        print('Geopandas or shapefile not available, skipping map.')
        return
    df = pd.read_csv(census_path)
    gdf = gpd.read_file(shapefile_path)
    gdf['GEOID'] = gdf['GEOID'].astype(str)
    merged = gdf.merge(df, on='GEOID')
    ax = merged.plot(column='B25064_001E', cmap='OrRd', legend=True, figsize=(10,8), edgecolor='k')
    ax.set_title('Median Gross Rent by Tract')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'rent_choropleth.png'))
    plt.close()

# 5. Scatterplot: Median Income vs. Median Gross Rent (Tract Level)
def plot_income_vs_rent(census_path):
    df = pd.read_csv(census_path)
    plt.figure(figsize=(7,5))
    sns.scatterplot(x='B19013_001E', y='B25064_001E', data=df)
    plt.title('Median Income vs. Median Gross Rent by Tract')
    plt.xlabel('Median Household Income ($)')
    plt.ylabel('Median Gross Rent ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'income_vs_rent.png'))
    plt.close()

# 6. Bar Chart: Average Rent by Bedroom Count (City Level)
def plot_rent_by_bedroom(rent_bedroom_path):
    df = pd.read_csv(rent_bedroom_path)
    plt.figure(figsize=(7,5))
    sns.barplot(x=df.columns[0], y=df.columns[1], data=df, palette='Blues')
    plt.title('Average Rent by Bedroom Count (City Level)')
    plt.xlabel('Bedroom Count')
    plt.ylabel('Average Rent ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'rent_by_bedroom.png'))
    plt.close()

# 7. Bar Chart: Average Rent by Property Type (City Level)
def plot_rent_by_property_type(rent_property_path):
    df = pd.read_csv(rent_property_path)
    plt.figure(figsize=(7,5))
    sns.barplot(x=df.columns[0], y=df.columns[1], data=df, palette='Greens')
    plt.title('Average Rent by Property Type (City Level)')
    plt.xlabel('Property Type')
    plt.ylabel('Average Rent ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'rent_by_property_type.png'))
    plt.close()

# 8. Crime Counts by Type (Bar Chart)
def plot_crime_counts(crime_path):
    df = pd.read_csv(crime_path)
    # Use the first column as crime type, sum 2025 counts
    summary = df.groupby(df.columns[0])[df.columns[4]].sum().reset_index()
    plt.figure(figsize=(10,6))
    sns.barplot(x=summary[df.columns[0]], y=summary[df.columns[4]], palette='Reds')
    plt.title('Crime Counts by Type (2025, Miami-Dade)')
    plt.xlabel('Crime Type')
    plt.ylabel('Count (2025)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'crime_counts.png'))
    plt.close()

# 9. Time Comparison: Crime Counts 2024 vs. 2025 (Side-by-Side Bars)
def plot_crime_time_comparison(crime_path):
    df = pd.read_csv(crime_path)
    summary = df.groupby(df.columns[0])[[df.columns[3], df.columns[4]]].sum().reset_index()
    summary = summary.rename(columns={df.columns[3]: '2024', df.columns[4]: '2025'})
    summary_melt = summary.melt(id_vars=df.columns[0], value_vars=['2024', '2025'], var_name='Year', value_name='Count')
    plt.figure(figsize=(10,6))
    sns.barplot(x=df.columns[0], y='Count', hue='Year', data=summary_melt, palette='Set2')
    plt.title('Crime Counts: 2024 vs 2025 (Miami-Dade)')
    plt.xlabel('Crime Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'crime_time_comparison.png'))
    plt.close()

# 10. Correlation Heatmap: Census Variables (Tract Level)
def plot_census_corr_heatmap(census_path):
    df = pd.read_csv(census_path)
    exclude_cols = {'GEOID', 'tract', 'county', 'state', 'censusgeo'}
    num_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap: Census Variables (Tract Level)')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'census_corr_heatmap.png'))
    plt.close()

if __name__ == "__main__":
    # File paths
    census_path = 'data/raw/census/acs5_miamidade_tracts.csv'
    rent_bedroom_path = 'data/processed/rental/avg_rent_by_bedroom.csv'
    rent_property_path = 'data/processed/rental/avg_rent_by_property_type.csv'
    crime_path = 'data/raw/crime/miami_dade_part1_crimes_ytd_comparison_2024_2025.csv'
    # Miami-Dade tract shapefile path (optional, for maps)
    tract_shapefile = None  # e.g., 'data/geo/miami_dade_tracts.shp'

    plot_income_distribution(census_path)
    plot_rent_distribution(census_path)
    plot_income_map(census_path, tract_shapefile)
    plot_rent_map(census_path, tract_shapefile)
    plot_income_vs_rent(census_path)
    plot_rent_by_bedroom(rent_bedroom_path)
    plot_rent_by_property_type(rent_property_path)
    plot_crime_counts(crime_path)
    plot_crime_time_comparison(crime_path)
    plot_census_corr_heatmap(census_path)
