#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate visualizations for the Airbnb impact on housing affordability analysis.

This script creates visualizations that illustrate the relationship between
short-term rental density and housing affordability metrics in Miami-Dade County.

Visualizations created:
1. Choropleth map of Airbnb density by census tract
2. Scatter plot of Airbnb density vs. rent-to-income ratio
3. Bar chart comparing affordability in high vs. low Airbnb density areas
4. Correlation heatmap between Airbnb metrics and housing variables
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import matplotlib.ticker as mtick

# Set up file paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
COMBINED_DIR = PROCESSED_DATA_DIR / "combined"
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
VISUALIZATION_DIR = PROJECT_DIR / "visualizations" / "airbnb_impact"

# Create output directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
colors = plt.cm.tab10.colors

def load_data():
    """
    Load the processed combined data for visualization.
    
    Returns:
        tuple: (combined_data, correlation_data, census_tracts_gdf)
    """
    print("Loading data for visualization...")
    
    # Load combined metrics
    combined_file = COMBINED_DIR / "tract_combined_metrics.csv"
    if os.path.exists(combined_file):
        combined_data = pd.read_csv(combined_file)
        print(f"Loaded combined data with {combined_data.shape[0]} census tracts")
    else:
        print(f"Warning: {combined_file} not found")
        combined_data = None
    
    # Load correlation results
    correlation_file = COMBINED_DIR / "airbnb_housing_correlation.csv"
    if os.path.exists(correlation_file):
        correlation_data = pd.read_csv(correlation_file)
        print(f"Loaded correlation data with {correlation_data.shape[0]} variables")
    else:
        print(f"Warning: {correlation_file} not found")
        correlation_data = None
    
    # Load census tract geometries for mapping
    tracts_geojson = RAW_DATA_DIR / "census" / "miami_dade_census_tracts.geojson"
    if os.path.exists(tracts_geojson):
        tracts_gdf = gpd.read_file(tracts_geojson)
        print(f"Loaded census tract boundaries with {tracts_gdf.shape[0]} tracts")
    else:
        # Try alternative sources for geometries
        acs_file = RAW_DATA_DIR / "census" / "acs5_miamidade_tracts.csv"
        if os.path.exists(acs_file):
            acs_df = pd.read_csv(acs_file)
            # Check if geometry column exists
            if 'geometry' in acs_df.columns:
                from shapely import wkt
                acs_df['geometry'] = acs_df['geometry'].apply(wkt.loads)
                tracts_gdf = gpd.GeoDataFrame(acs_df, geometry='geometry', crs="EPSG:4326")
                print(f"Created census tract boundaries from ACS data with {tracts_gdf.shape[0]} tracts")
            else:
                print("Warning: No geometry data found for census tracts")
                tracts_gdf = None
        else:
            print("Warning: No census tract boundary data found")
            tracts_gdf = None
    
    return combined_data, correlation_data, tracts_gdf

def create_airbnb_density_map(combined_data, tracts_gdf):
    """
    Create a choropleth map of Airbnb rental density by census tract.
    
    Args:
        combined_data: DataFrame with combined metrics
        tracts_gdf: GeoDataFrame with census tract boundaries
    """
    print("Creating Airbnb density choropleth map...")
    
    if combined_data is None or tracts_gdf is None:
        print("Error: Missing data for choropleth map")
        return
    
    # Identify necessary columns
    density_col = next((col for col in combined_data.columns if 'density' in col.lower() and 'listing' in col.lower()), None)
    tract_id_col = next((col for col in combined_data.columns if 'tract' in col.lower() or 'geoid' in col.lower()), None)
    tract_gdf_id_col = next((col for col in tracts_gdf.columns if 'tract' in col.lower() or 'geoid' in col.lower()), None)
    
    if density_col is None or tract_id_col is None or tract_gdf_id_col is None:
        print("Error: Cannot identify required columns for mapping")
        # Try to guess column names
        if density_col is None:
            density_col = 'listing_density'
        if tract_id_col is None:
            tract_id_col = combined_data.columns[0]
        if tract_gdf_id_col is None:
            tract_gdf_id_col = tracts_gdf.columns[0]
        print(f"Using inferred columns: density={density_col}, tract_id={tract_id_col}, gdf_id={tract_gdf_id_col}")
    
    # Merge density data with tract boundaries
    merged_gdf = tracts_gdf.copy()
    # Ensure ID columns are string type for matching
    merged_gdf[tract_gdf_id_col] = merged_gdf[tract_gdf_id_col].astype(str)
    combined_data[tract_id_col] = combined_data[tract_id_col].astype(str)
    
    # Merge on tract IDs
    merged_gdf = merged_gdf.merge(
        combined_data[[tract_id_col, density_col]], 
        left_on=tract_gdf_id_col,
        right_on=tract_id_col,
        how='left'
    )
    
    # Fill NaN values with 0 for tracts without listings
    merged_gdf[density_col] = merged_gdf[density_col].fillna(0)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Set up a colormap with quantiles for better visualization
    vmin = 0
    vmax = merged_gdf[density_col].quantile(0.95)  # 95th percentile to avoid extreme outliers
    
    # Plot the choropleth map
    merged_gdf.plot(
        column=density_col,
        ax=ax,
        cmap='YlOrRd',
        edgecolor='gray',
        linewidth=0.2,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        legend_kwds={
            'label': 'Airbnb Listings per sq km',
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.01,
        }
    )
    
    # Add title and labels
    ax.set_title('Airbnb Rental Density by Census Tract in Miami-Dade County', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axis ticks for cleaner map
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add annotations for notable areas if available
    if 'high_impact_area' in combined_data.columns:
        high_impact_tracts = combined_data[combined_data['high_impact_area'] == True][tract_id_col].tolist()
        high_impact_geometries = merged_gdf[merged_gdf[tract_gdf_id_col].isin(high_impact_tracts)]
        
        # Highlight high-impact areas
        high_impact_geometries.plot(
            ax=ax,
            color='none',
            edgecolor='red',
            linewidth=1.5,
            label='High-Impact Areas'
        )
        
        # Add legend for high-impact areas
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='none', edgecolor='red', linewidth=1.5, label='High-Impact Areas')]
        ax.legend(handles=legend_elements, loc='lower right')
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_density_map.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Airbnb density map to {output_path}")

def create_affordability_scatter_plot(combined_data):
    """
    Create a scatter plot showing the relationship between Airbnb density
    and rent-to-income ratio by census tract.
    
    Args:
        combined_data: DataFrame with combined metrics
    """
    print("Creating affordability scatter plot...")
    
    if combined_data is None:
        print("Error: Missing data for scatter plot")
        return
    
    # Identify necessary columns
    density_col = next((col for col in combined_data.columns if 'density' in col.lower() and 'listing' in col.lower()), 'listing_density')
    ratio_col = next((col for col in combined_data.columns if 'rent_to_income' in col.lower()), None)
    if ratio_col is None:
        ratio_col = next((col for col in combined_data.columns if 'median_rent' in col.lower()), None)
    
    if density_col not in combined_data.columns or ratio_col not in combined_data.columns:
        print("Error: Required columns not found for scatter plot")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter out extreme outliers for better visualization
    density_upper = combined_data[density_col].quantile(0.95)
    ratio_upper = combined_data[ratio_col].quantile(0.95)
    plot_data = combined_data[
        (combined_data[density_col] <= density_upper) & 
        (combined_data[ratio_col] <= ratio_upper)
    ].copy()
    
    # Add population column if available for sizing points
    pop_col = next((col for col in plot_data.columns if 'population' in col.lower()), None)
    if pop_col is not None:
        # Normalize population for point sizes
        plot_data['point_size'] = 20 * (plot_data[pop_col] / plot_data[pop_col].max())
        s = plot_data['point_size']
    else:
        s = 50  # Default point size
    
    # Add color based on affordability status if available
    if 'affordability_status' in plot_data.columns:
        # Create categorical color map
        categories = plot_data['affordability_status'].unique()
        cmap = plt.cm.get_cmap('viridis', len(categories))
        
        # Map categories to colors
        cat_to_int = {cat: i for i, cat in enumerate(categories)}
        plot_data['color_index'] = plot_data['affordability_status'].map(cat_to_int)
        c = plot_data['color_index']
        
        scatter = ax.scatter(
            plot_data[density_col], 
            plot_data[ratio_col],
            s=s,
            c=c,
            cmap=cmap,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Add a legend for affordability categories
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=cmap(cat_to_int[cat]), 
                   markersize=10, label=cat) 
            for cat in categories if cat in plot_data['affordability_status'].values
        ]
        ax.legend(handles=legend_elements, title='Affordability Status', 
                  loc='upper left', bbox_to_anchor=(1, 1))
    else:
        # Simple scatter plot without categories
        scatter = ax.scatter(
            plot_data[density_col], 
            plot_data[ratio_col],
            s=s,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
    
    # Add reference line for 30% affordability threshold
    if 'rent_to_income' in ratio_col.lower():
        ax.axhline(y=30, color='red', linestyle='--', linewidth=1.5, 
                  label='30% Affordability Threshold')
        
        # Annotate the threshold line
        ax.text(
            plot_data[density_col].max() * 0.95, 
            30.5, 
            'Housing Cost Burden Threshold (30%)', 
            color='red', 
            fontweight='bold',
            ha='right'
        )
    
    # Calculate and plot trendline
    try:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            plot_data[density_col], 
            plot_data[ratio_col]
        )
        
        x = np.array([plot_data[density_col].min(), plot_data[density_col].max()])
        y = slope * x + intercept
        
        ax.plot(x, y, color='blue', linestyle='-', linewidth=2, 
               label=f'Trend (r={r_value:.2f}, p={p_value:.4f})')
        
        # Add R-squared and p-value annotation
        ax.annotate(
            f'R² = {r_value**2:.3f}\np-value = {p_value:.4f}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
    except:
        print("Warning: Could not calculate trendline")
    
    # Add title and labels
    ax.set_title('Relationship Between Airbnb Density and Housing Affordability', fontsize=16)
    ax.set_xlabel('Airbnb Listings per Square Kilometer', fontsize=14)
    ax.set_ylabel('Rent-to-Income Ratio (%)', fontsize=14)
    
    # Format y-axis as percentage if it's a ratio
    if 'ratio' in ratio_col.lower() or 'percent' in ratio_col.lower():
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Add legend
    if 'rent_to_income' in ratio_col.lower():
        ax.legend(title='', loc='upper left')
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_affordability_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved affordability scatter plot to {output_path}")

def create_density_comparison_bar_chart(combined_data):
    """
    Create a bar chart comparing housing affordability metrics between
    areas with high, medium, and low Airbnb density.
    
    Args:
        combined_data: DataFrame with combined metrics
    """
    print("Creating density comparison bar chart...")
    
    if combined_data is None:
        print("Error: Missing data for bar chart")
        return
    
    # Check if density quintile column exists
    if 'airbnb_density_quintile' not in combined_data.columns:
        print("Error: Density quintile column not found for comparison chart")
        return
    
    # Find affordability metrics to compare
    metrics = []
    for col in combined_data.columns:
        if any(term in col.lower() for term in ['rent_to_income', 'price_to_income', 'affordability']):
            if combined_data[col].dtype in [np.float64, np.int64]:
                metrics.append(col)
    
    if not metrics:
        print("Error: No affordability metrics found for comparison")
        return
    
    # Select primary metric for comparison
    primary_metric = next((m for m in metrics if 'rent_to_income' in m.lower()), metrics[0])
    
    # Group data by density quintile
    grouped = combined_data.groupby('airbnb_density_quintile')[primary_metric].agg(['mean', 'median', 'std']).reset_index()
    
    # Order categories from low to high density
    density_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    grouped['airbnb_density_quintile'] = pd.Categorical(
        grouped['airbnb_density_quintile'], 
        categories=density_order,
        ordered=True
    )
    grouped = grouped.sort_values('airbnb_density_quintile')
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bars = ax.bar(
        grouped['airbnb_density_quintile'],
        grouped['mean'],
        yerr=grouped['std'],
        capsize=10,
        edgecolor='black',
        color=[plt.cm.YlOrRd(i/4) for i in range(len(grouped))]
    )
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            f'{height:.1f}%',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Add title and labels
    metric_name = primary_metric.replace('_', ' ').title()
    ax.set_title(f'{metric_name} by Airbnb Density Level', fontsize=16)
    ax.set_xlabel('Airbnb Density Level', fontsize=14)
    ax.set_ylabel(f'{metric_name} (%)', fontsize=14)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    
    # Add affordability threshold line if using rent-to-income ratio
    if 'rent_to_income' in primary_metric.lower():
        ax.axhline(y=30, color='red', linestyle='--', linewidth=2, 
                  label='30% Affordability Threshold')
        ax.legend()
    
    # Add grid for readability
    ax.grid(True, axis='y', alpha=0.3)
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_density_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved density comparison bar chart to {output_path}")

def create_correlation_heatmap(correlation_data):
    """
    Create a correlation heatmap showing relationships between
    Airbnb metrics and housing variables.
    
    Args:
        correlation_data: DataFrame with correlation analysis results
    """
    print("Creating correlation heatmap...")
    
    if correlation_data is None:
        print("Error: Missing data for correlation heatmap")
        return
    
    # Prepare data for heatmap
    variables = correlation_data['housing_variable'].tolist()
    correlation_values = correlation_data['pearson_r'].tolist()
    pvalues = correlation_data['pearson_p'].tolist()
    
    # Create DataFrame for heatmap
    heatmap_data = pd.DataFrame({
        'Variable': variables,
        'Correlation with Airbnb Density': correlation_values,
        'P-Value': pvalues
    })
    
    # Clean variable names for display
    heatmap_data['Variable'] = heatmap_data['Variable'].str.replace('_', ' ').str.title()
    
    # Sort by absolute correlation strength
    heatmap_data = heatmap_data.iloc[np.argsort(np.abs(heatmap_data['Correlation with Airbnb Density']))[::-1]]
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot correlation values as a horizontal bar chart
    bars = ax.barh(
        heatmap_data['Variable'],
        heatmap_data['Correlation with Airbnb Density'],
        color=[plt.cm.RdBu_r(0.5 * (corr + 1)) for corr in heatmap_data['Correlation with Airbnb Density']],
        edgecolor='black',
        height=0.7
    )
    
    # Add data labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        pvalue = heatmap_data['P-Value'].iloc[i]
        
        # Add significance indicators
        if pvalue < 0.001:
            significance = "***"
        elif pvalue < 0.01:
            significance = "**"
        elif pvalue < 0.05:
            significance = "*"
        else:
            significance = "ns"
            
        label_x = width + 0.02 if width >= 0 else width - 0.02
        ha = 'left' if width >= 0 else 'right'
            
        ax.text(
            label_x,
            bar.get_y() + bar.get_height()/2,
            f'{width:.2f} {significance}',
            ha=ha, 
            va='center',
            fontweight='bold'
        )
    
    # Add title and labels
    ax.set_title('Correlation between Airbnb Density and Housing Variables', fontsize=16)
    ax.set_xlabel('Pearson Correlation Coefficient (r)', fontsize=14)
    
    # Add reference line at zero
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Set x-axis limits for symmetry
    max_abs_corr = max(abs(min(correlation_values)), abs(max(correlation_values)))
    ax.set_xlim(-max_abs_corr - 0.1, max_abs_corr + 0.1)
    
    # Add significance legend
    ax.text(
        0.02, 0.01,
        "Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_correlation_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation chart to {output_path}")

def main():
    """
    Main function to generate all visualizations.
    """
    print("Starting visualization generation...")
    
    # Load data
    combined_data, correlation_data, tracts_gdf = load_data()
    
    # Create visualizations
    create_airbnb_density_map(combined_data, tracts_gdf)
    create_affordability_scatter_plot(combined_data)
    create_density_comparison_bar_chart(combined_data)
    create_correlation_heatmap(correlation_data)
    
    print("Visualization generation complete!")

if __name__ == "__main__":
    main()
