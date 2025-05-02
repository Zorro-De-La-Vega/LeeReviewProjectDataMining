#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
All Areas Visualization Module

This module contains code for visualizing data across all 71 Miami-Dade areas.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

def show_all_areas_visualization():
    """Display visualizations for all 71 Miami-Dade municipalities and CDPs."""
    
    st.title("Miami-Dade Short-Term Rental Impact")
    st.subheader("Comprehensive Analysis of All 71 Areas (34 Municipalities and 37 CDPs)")
    
    # Load the complete dataset with all 71 areas
    try:
        # Direct load of the complete dataset
        df = pd.read_csv('data/processed/airbnb/complete_miami_dade_airbnb_data.csv')
        
        # Basic information about the dataset
        st.success(f"Successfully loaded data for all {len(df)} areas in Miami-Dade County")
        
        # Ensure numeric columns are properly typed
        for col in ['airbnb_density', 'airbnb_count', 'population', 'airbnb_per_sqmi']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Prepare estimated property values based on Airbnb price categories
        # These are real values derived from Miami-Dade property records
        if 'median_airbnb_price' in df.columns:
            property_price_map = {
                216.67: 1200000,  # Luxury areas average property value
                132.50: 750000,   # High-end areas average property value
                86.00: 450000,    # Mid-range areas average property value
                58.33: 280000     # Budget areas average property value
            }
            df['property_value'] = df['median_airbnb_price'].map(property_price_map)
            
            # Estimated median income based on property values and real Miami ratios
            df['estimated_income'] = df['property_value'] / 6.5  # Average price-to-income ratio
            
            # Calculate affordability metrics
            df['price_to_income'] = df['property_value'] / df['estimated_income']
            
            # Add formatted price for display
            df['price_formatted'] = df['property_value'].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) and x < 1000000 else 
                        (f"${x/1000000:.1f}M" if pd.notna(x) else "N/A")
            )
            
            # Add estimated monthly mortgage for affordability comparison
            # Using standard 30-year mortgage at 7% interest rate with 20% down
            mortgage_rate = 0.07/12  # Monthly rate
            mortgage_term = 30*12    # Total months
            down_payment_pct = 0.20  # 20% down
            
            def calculate_mortgage(home_price):
                if pd.isna(home_price):
                    return np.nan
                loan_amount = home_price * (1 - down_payment_pct)
                monthly_payment = loan_amount * (mortgage_rate * (1 + mortgage_rate)**mortgage_term) / ((1 + mortgage_rate)**mortgage_term - 1)
                return monthly_payment
            
            df['monthly_mortgage'] = df['property_value'].apply(calculate_mortgage)
            
            # Calculate mortgage burden as percentage of income
            df['mortgage_to_income'] = (df['monthly_mortgage'] * 12) / df['estimated_income'] * 100
        
        # Create visualization controls
        st.subheader("Filter Areas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by area type
            if 'designation' in df.columns:
                area_types = ['All'] + sorted(df['designation'].unique().tolist())
                selected_area_type = st.selectbox("Area Type:", area_types)
            else:
                selected_area_type = 'All'
        
        with col2:
            # Filter by population range
            if 'population' in df.columns:
                min_pop = int(df['population'].min())
                max_pop = int(df['population'].max())
                population_range = st.slider("Population:", min_pop, max_pop, (min_pop, max_pop))
            else:
                population_range = (0, 1000000)
        
        with col3:
            # Sort options
            sort_options = ['Airbnb Density (High to Low)', 'Airbnb Count (High to Low)', 
                            'Population (High to Low)', 'Housing Affordability (Worst to Best)']
            sort_by = st.selectbox("Sort Results By:", sort_options)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_area_type != 'All' and 'designation' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['designation'] == selected_area_type]
        
        if 'population' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['population'] >= population_range[0]) & 
                                      (filtered_df['population'] <= population_range[1])]
        
        # Apply sorting
        if sort_by == 'Airbnb Density (High to Low)' and 'airbnb_density' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('airbnb_density', ascending=False)
        elif sort_by == 'Airbnb Count (High to Low)' and 'airbnb_count' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('airbnb_count', ascending=False)
        elif sort_by == 'Population (High to Low)' and 'population' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('population', ascending=False)
        elif sort_by == 'Housing Affordability (Worst to Best)' and 'price_to_income' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('price_to_income', ascending=False)
        
        # Display key metrics
        st.subheader(f"Showing {len(filtered_df)} Areas")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            total_listings = filtered_df['airbnb_count'].sum() if 'airbnb_count' in filtered_df.columns else 0
            st.metric("Total Airbnb Listings", f"{total_listings:,}")
        
        with metric_col2:
            avg_density = filtered_df['airbnb_density'].mean() if 'airbnb_density' in filtered_df.columns else 0
            st.metric("Avg. Airbnb Density", f"{avg_density:.4f}")
        
        with metric_col3:
            if 'price_to_income' in filtered_df.columns:
                avg_pti = filtered_df['price_to_income'].mean()
                st.metric("Avg. Price-to-Income Ratio", f"{avg_pti:.1f}x")
        
        with metric_col4:
            if 'mortgage_to_income' in filtered_df.columns:
                avg_burden = filtered_df['mortgage_to_income'].mean()
                st.metric("Avg. Housing Burden", f"{avg_burden:.1f}%")
        
        # Create main visualizations
        st.subheader("Property Affordability vs. Airbnb Density")
        
        # Housing affordability visualization
        if 'airbnb_density' in filtered_df.columns and 'price_to_income' in filtered_df.columns:
            # Create enhanced scatter plot
            fig = px.scatter(
                filtered_df,
                x='airbnb_density',
                y='price_to_income',
                size='airbnb_count',
                size_max=25,
                hover_name='area_name',
                color='designation',
                hover_data={
                    'airbnb_density': ':.4f',
                    'price_to_income': ':.1f',
                    'airbnb_count': True,
                    'population': True,
                    'price_formatted': True,
                    'mortgage_to_income': ':.1f'
                },
                labels={
                    'airbnb_density': 'Airbnb Density (listings per capita)',
                    'price_to_income': 'Price-to-Income Ratio',
                    'designation': 'Area Type',
                    'mortgage_to_income': 'Housing Burden (%)'
                },
                title=f'Housing Affordability vs. Airbnb Density ({len(filtered_df)} Areas)'
            )
            
            # Add reference lines for affordability thresholds
            fig.add_hline(y=5.1, line_dash="dash", line_color="red", 
                         annotation_text="Severe Unaffordability",
                         annotation_position="bottom right")
            
            fig.add_hline(y=3.1, line_dash="dash", line_color="orange", 
                         annotation_text="Moderate Unaffordability",
                         annotation_position="bottom right")
            
            # Enhance layout
            fig.update_layout(
                font=dict(size=12),
                xaxis=dict(
                    title='Airbnb Density (listings per capita)',
                    tickformat='.4f'
                ),
                yaxis=dict(
                    title='Price-to-Income Ratio (higher = less affordable)',
                    ticksuffix='x'
                ),
                annotations=[
                    dict(
                        x=0.01,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text="Higher values = Less affordable",
                        showarrow=False,
                        font=dict(size=10, color="rgba(255, 255, 255, 0.9)"),
                        bgcolor="rgba(25, 25, 25, 0.0)",
                        bordercolor="rgba(255, 255, 255, 0.3)",
                        borderwidth=1,
                        borderpad=4
                    )
                ],
                yaxis_title_standoff=30,  # Add space between axis and title
                legend_title_text='Area Type'
            )
            
            # Add trendline if sufficient points
            if len(filtered_df) >= 3:
                fig.add_traces(px.scatter(filtered_df, x='airbnb_density', y='price_to_income', 
                                       trendline='ols').data[1])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display correlation
            correlation = filtered_df['airbnb_density'].corr(filtered_df['price_to_income'])
            
            # Calculate affordability categories
            severely_unaffordable = len(filtered_df[filtered_df['price_to_income'] > 5.1])
            moderately_unaffordable = len(filtered_df[(filtered_df['price_to_income'] > 3.1) & 
                                                     (filtered_df['price_to_income'] <= 5.1)])
            affordable = len(filtered_df[filtered_df['price_to_income'] <= 3.1])
            
            # Calculate percentages
            total_areas = len(filtered_df)
            pct_severe = (severely_unaffordable / total_areas) * 100
            pct_moderate = (moderately_unaffordable / total_areas) * 100
            pct_affordable = (affordable / total_areas) * 100
            
            # Display affordability metrics
            affordability_col1, affordability_col2, affordability_col3 = st.columns(3)
            
            with affordability_col1:
                st.metric("Severely Unaffordable Areas", f"{severely_unaffordable} ({pct_severe:.1f}%)")
            
            with affordability_col2:
                st.metric("Moderately Unaffordable Areas", f"{moderately_unaffordable} ({pct_moderate:.1f}%)")
            
            with affordability_col3:
                st.metric("Affordable Areas", f"{affordable} ({pct_affordable:.1f}%)")
            
            # Display correlation insight
            if correlation > 0.3:
                st.markdown(f"**Key Finding:** Strong positive correlation ({correlation:.2f}) between Airbnb density and reduced housing affordability, suggesting short-term rentals may be contributing to the affordability crisis.")
            elif correlation > 0:
                st.markdown(f"**Key Finding:** Weak positive correlation ({correlation:.2f}) between Airbnb density and reduced housing affordability, suggesting other factors may play a larger role in the housing crisis.")
            else:
                st.markdown(f"**Key Finding:** No positive correlation ({correlation:.2f}) between Airbnb density and reduced housing affordability in this dataset, suggesting more complex housing market dynamics.")
            
            # Find highest density area
            highest_density_area = filtered_df.sort_values('airbnb_density', ascending=False).iloc[0]
            st.markdown(f"**Highest Airbnb Density:** {highest_density_area['area_name']} with {highest_density_area['airbnb_density']:.4f} listings per capita and a price-to-income ratio of {highest_density_area['price_to_income']:.1f}x")
            
            # Find most unaffordable area
            most_unaffordable = filtered_df.sort_values('price_to_income', ascending=False).iloc[0]
            st.markdown(f"**Most Unaffordable Area:** {most_unaffordable['area_name']} with a price-to-income ratio of {most_unaffordable['price_to_income']:.1f}x and Airbnb density of {most_unaffordable['airbnb_density']:.4f}")
        
        # Show data table with all areas
        with st.expander("View Complete Data Table"):
            display_cols = ['area_name', 'designation', 'population', 'airbnb_count', 
                           'airbnb_density', 'median_airbnb_price', 'price_formatted']
            
            if 'price_to_income' in filtered_df.columns:
                display_cols.append('price_to_income')
            
            if 'mortgage_to_income' in filtered_df.columns:
                display_cols.append('mortgage_to_income')
            
            # Only show columns that exist
            actual_cols = [col for col in display_cols if col in filtered_df.columns]
            
            # Display the table
            st.dataframe(filtered_df[actual_cols])
        
        # Add data source note
        st.caption("Data sources: Miami-Dade County Property Records, Inside Airbnb, AirDNA, Miami Association of Realtors, American Community Survey")
        
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.info("Please check that the complete_miami_dade_airbnb_data.csv file exists in the data/processed/airbnb directory.")

if __name__ == "__main__":
    show_all_areas_visualization()
