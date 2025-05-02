#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate visualizations for Airbnb impact on housing market by ZIP code.

This script creates visualizations that illustrate the relationship between
short-term rentals and housing market metrics at the ZIP code level for
Miami-Dade County.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.ticker as mtick

# Set up file paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
AIRBNB_DIR = PROCESSED_DATA_DIR / "airbnb"
COMBINED_DIR = PROCESSED_DATA_DIR / "combined"
VISUALIZATION_DIR = PROJECT_DIR / "visualizations" / "airbnb_impact"

# Create output directory if it doesn't exist
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
colors = plt.cm.tab10.colors

def load_data():
    """
    Load the processed Airbnb and combined data for visualization.
    """
    print("Loading data for visualization...")
    
    # Load Airbnb by ZIP code data
    airbnb_file = AIRBNB_DIR / "airbnb_by_zipcode.csv"
    if os.path.exists(airbnb_file):
        airbnb_data = pd.read_csv(airbnb_file)
        print(f"Loaded Airbnb data for {airbnb_data.shape[0]} ZIP codes")
    else:
        print(f"Warning: {airbnb_file} not found")
        airbnb_data = None
    
    # Load combined Airbnb-census data
    combined_file = COMBINED_DIR / "airbnb_census_by_zipcode.csv"
    if os.path.exists(combined_file):
        combined_data = pd.read_csv(combined_file)
        print(f"Loaded combined data for {combined_data.shape[0]} ZIP codes")
    else:
        print(f"Warning: {combined_file} not found")
        combined_data = None
    
    # Load summary statistics
    summary_file = AIRBNB_DIR / "summary_airbnb.csv"
    if os.path.exists(summary_file):
        summary_data = pd.read_csv(summary_file)
        print(f"Loaded summary statistics with {summary_data.shape[0]} metrics")
    else:
        print(f"Warning: {summary_file} not found")
        summary_data = None
    
    return airbnb_data, combined_data, summary_data

def create_listings_by_zipcode_chart(airbnb_data):
    """
    Create a bar chart of Airbnb listings by ZIP code.
    """
    print("Creating listings by ZIP code chart...")
    
    if airbnb_data is None:
        print("Error: No Airbnb data available for chart")
        return
    
    # Sort by listing count
    sorted_data = airbnb_data.sort_values('listing_count', ascending=False)
    
    # Take top 15 ZIP codes for readability
    plot_data = sorted_data.head(15)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bars = ax.bar(
        plot_data['Zipcode'],
        plot_data['listing_count'],
        color=colors[0],
        edgecolor='black',
        alpha=0.7
    )
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f'{int(height)}',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Add title and labels
    ax.set_title('Airbnb Listings by ZIP Code in Miami-Dade County', fontsize=16)
    ax.set_xlabel('ZIP Code', fontsize=14)
    ax.set_ylabel('Number of Listings', fontsize=14)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_listings_by_zipcode.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved listings by ZIP code chart to {output_path}")

def create_price_distribution_chart(airbnb_data):
    """
    Create charts showing price distribution across ZIP codes.
    """
    print("Creating price distribution charts...")
    
    if airbnb_data is None:
        print("Error: No Airbnb data available for chart")
        return
    
    # Sort by median price
    sorted_data = airbnb_data.sort_values('median_price', ascending=False)
    
    # Take top 15 ZIP codes for readability
    plot_data = sorted_data.head(15)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bars = ax.bar(
        plot_data['Zipcode'],
        plot_data['median_price'],
        color=colors[1],
        edgecolor='black',
        alpha=0.7
    )
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f'${int(height)}',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Add title and labels
    ax.set_title('Median Airbnb Price by ZIP Code in Miami-Dade County', fontsize=16)
    ax.set_xlabel('ZIP Code', fontsize=14)
    ax.set_ylabel('Median Price per Night (USD)', fontsize=14)
    
    # Format y-axis as USD
    ax.yaxis.set_major_formatter('${x:,.0f}')
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_median_price_by_zipcode.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved median price chart to {output_path}")

def create_property_type_chart(airbnb_data):
    """
    Create chart showing percentage of entire homes by ZIP code.
    """
    print("Creating property type chart...")
    
    if airbnb_data is None:
        print("Error: No Airbnb data available for chart")
        return
    
    # Sort by entire home percentage
    sorted_data = airbnb_data.sort_values('entire_home_percent', ascending=False)
    
    # Take top 15 ZIP codes for readability
    plot_data = sorted_data.head(15)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars
    bars = ax.bar(
        plot_data['Zipcode'],
        plot_data['entire_home_percent'],
        color=colors[2],
        edgecolor='black',
        alpha=0.7
    )
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.1f}%',
            ha='center', 
            va='bottom',
            fontweight='bold'
        )
    
    # Add title and labels
    ax.set_title('Percentage of Entire Home Listings by ZIP Code', fontsize=16)
    ax.set_xlabel('ZIP Code', fontsize=14)
    ax.set_ylabel('Entire Home Listings (%)', fontsize=14)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_home_type_by_zipcode.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved property type chart to {output_path}")

def create_combined_metrics_chart(combined_data):
    """
    Create a visualization combining Airbnb metrics with census data.
    """
    print("Creating combined metrics chart...")
    
    if combined_data is None:
        print("Error: No combined data available for chart")
        return
    
    # Find income column if available
    income_col = next((col for col in combined_data.columns if 'median_household_income' in col.lower()), None)
    
    if income_col is None:
        print("Warning: No income column found for combined chart")
        return
    
    # Create scatter plot of income vs. Airbnb listings
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        combined_data[income_col],
        combined_data['listing_count'],
        s=combined_data['median_price'] / 2,  # Size points by price
        alpha=0.7,
        c=combined_data['entire_home_percent'],  # Color by home type percentage
        cmap='viridis',
        edgecolor='black'
    )
    
    # Add ZIP code labels to points
    for i, row in combined_data.iterrows():
        ax.annotate(
            row['Zipcode'],
            (row[income_col], row['listing_count']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    # Add title and labels
    ax.set_title('Airbnb Listings vs. Median Household Income by ZIP Code', fontsize=16)
    ax.set_xlabel('Median Household Income (USD)', fontsize=14)
    ax.set_ylabel('Number of Airbnb Listings', fontsize=14)
    
    # Format x-axis as USD
    ax.xaxis.set_major_formatter('${x:,.0f}')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Percentage of Entire Home Listings', fontsize=12)
    
    # Add legend for bubble size
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=8, alpha=0.7, label='$100/night'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=12, alpha=0.7, label='$200/night'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=16, alpha=0.7, label='$300/night')
    ]
    ax.legend(handles=legend_elements, title='Median Price', loc='upper left')
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_income_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved income comparison chart to {output_path}")

def create_summary_metrics_chart(summary_data):
    """
    Create a visual summary of key Airbnb metrics.
    """
    print("Creating summary metrics chart...")
    
    if summary_data is None:
        print("Error: No summary data available for chart")
        return
    
    # Select relevant metrics for visualization
    key_metrics = [
        'total_listings',
        'zipcodes_with_listings',
        'max_listings_per_zipcode',
        'avg_listings_per_zipcode',
        'avg_price',
        'median_price',
        'entire_home_percent'
    ]
    
    # Filter summary data to include only these metrics
    plot_data = summary_data[summary_data['metric'].isin(key_metrics)]
    
    if plot_data.empty:
        print("Error: No relevant metrics found in summary data")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    
    # Left subplot: Count metrics
    count_metrics = ['total_listings', 'zipcodes_with_listings', 
                     'max_listings_per_zipcode', 'avg_listings_per_zipcode']
    count_data = plot_data[plot_data['metric'].isin(count_metrics)]
    
    # Map metric names to readable labels
    metric_labels = {
        'total_listings': 'Total Listings',
        'zipcodes_with_listings': 'ZIP Codes with Listings',
        'max_listings_per_zipcode': 'Max Listings per ZIP',
        'avg_listings_per_zipcode': 'Avg Listings per ZIP',
        'avg_price': 'Average Price',
        'median_price': 'Median Price',
        'entire_home_percent': '% Entire Homes'
    }
    
    count_data['label'] = count_data['metric'].map(metric_labels)
    
    # Create horizontal bar chart for counts
    count_bars = axs[0].barh(
        count_data['label'],
        count_data['value'],
        color=colors[0:len(count_data)],
        edgecolor='black',
        alpha=0.7
    )
    
    # Add data labels
    for bar in count_bars:
        width = bar.get_width()
        axs[0].text(
            width + (width * 0.01),
            bar.get_y() + bar.get_height()/2,
            f'{int(width):,}',
            ha='left', 
            va='center',
            fontweight='bold'
        )
    
    axs[0].set_title('Airbnb Listing Counts', fontsize=14)
    axs[0].grid(axis='x', alpha=0.3)
    
    # Right subplot: Price and percentage metrics
    price_metrics = ['avg_price', 'median_price', 'entire_home_percent']
    price_data = plot_data[plot_data['metric'].isin(price_metrics)]
    price_data['label'] = price_data['metric'].map(metric_labels)
    
    # Create horizontal bar chart for prices/percentages
    price_bars = axs[1].barh(
        price_data['label'],
        price_data['value'],
        color=colors[4:4+len(price_data)],
        edgecolor='black',
        alpha=0.7
    )
    
    # Add data labels
    for bar in price_bars:
        width = bar.get_width()
        if bar.get_y() == 1:  # Median price
            label = f'${int(width):,}'
        elif bar.get_y() == 0:  # Average price
            label = f'${int(width):,}'
        else:  # Percentage
            label = f'{width:.1f}%'
            
        axs[1].text(
            width + (width * 0.01),
            bar.get_y() + bar.get_height()/2,
            label,
            ha='left', 
            va='center',
            fontweight='bold'
        )
    
    axs[1].set_title('Airbnb Pricing & Property Types', fontsize=14)
    axs[1].grid(axis='x', alpha=0.3)
    
    # Add overall title
    fig.suptitle('Key Airbnb Metrics for Miami-Dade County', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    output_path = VISUALIZATION_DIR / "airbnb_key_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved key metrics chart to {output_path}")

def main():
    """
    Main function to generate all visualizations.
    """
    print("Starting visualization generation...")
    
    # Load data
    airbnb_data, combined_data, summary_data = load_data()
    
    # Create visualizations
    create_listings_by_zipcode_chart(airbnb_data)
    create_price_distribution_chart(airbnb_data)
    create_property_type_chart(airbnb_data)
    create_combined_metrics_chart(combined_data)
    create_summary_metrics_chart(summary_data)
    
    print("Visualization generation complete!")

if __name__ == "__main__":
    main()
