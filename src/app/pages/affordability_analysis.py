"""
Affordability Index Analysis Page

This module provides a Streamlit interface for the affordability index analysis feature.
It allows users to explore housing affordability patterns across Miami neighborhoods
using Principal Component Analysis (PCA) and interactive Plotly visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import affordability models with dependency handling
try:
    from src.models.affordability_index import AffordabilityIndexCreator
    from src.visualization.affordability_plots import (
        plot_affordability_distribution, 
        plot_component_loadings,
        plot_cluster_analysis, 
        plot_cluster_profiles,
        plot_affordability_by_feature,
        create_affordability_dashboard
    )
    affordability_available = True
except ImportError as e:
    logger.error(f"Error importing affordability analysis modules: {e}", exc_info=True)
    affordability_available = False

# Try importing data loading utilities
try:
    from src.data.load_data import (
        load_airbnb_data, 
        load_miami_housing_data,
        load_neighborhood_data, 
        load_zillow_data
    )
    data_loading_available = True
except ImportError as e:
    logger.error(f"Error importing data loading modules: {e}", exc_info=True)
    data_loading_available = False

def show_affordability_analysis(data_loader, agent=None):
    """Display the affordability index analysis page in the Streamlit app.
    
    Args:
        data_loader: DataLoader instance with access to all datasets
        agent: Optional HousingImpactAgent instance for proactive insights
    """
    
    st.title("Housing Affordability Index Analysis")
    st.write("""
    This advanced analysis uses Principal Component Analysis (PCA) to create a composite affordability index
    that measures housing affordability across Miami neighborhoods. The index combines multiple economic
    indicators into a single score to identify affordability patterns and help stakeholders make data-driven decisions.
    """)
    
    # Display loading message while importing dependencies
    if not affordability_available:
        st.error("""
        ⚠️ The affordability analysis modules could not be loaded.
        Please install the required dependencies:
        ```
        pip install scikit-learn pandas plotly numpy matplotlib seaborn
        ```
        """)
        return
    
    if not data_loading_available:
        st.error("""
        ⚠️ The data loading modules could not be loaded.
        Please check that all required data files are in the correct locations.
        """)
        return
    
    # Create tabs for different aspects of affordability analysis
    tabs = st.tabs([
        "Overview & Controls", 
        "Affordability Distribution", 
        "Feature Contribution",
        "Neighborhood Clusters",
        "Strategic Insights"
    ])
    
    # Load data
    with st.spinner("Loading data for affordability analysis..."):
        # Try different data sources in order of preference
        data = None
        data_source = None
        data_columns = None
        
        # 1. First try to load consolidated, verified data
        try:
            # Load our verified, consolidated data that contains only real data points
            consolidated_data = pd.read_csv('data/consolidated/airbnb_comprehensive.csv')
            
            # Focus on municipalities only (not neighborhoods) for cleaner analysis
            consolidated_data['is_municipality'] = consolidated_data.apply(
                lambda row: pd.notna(row['municipality']) and 
                            row['municipality'] != 'Unknown' and 
                            (row['neighborhood'] == row['municipality'] or
                            row['municipality'] in row['neighborhood'] or
                            row['neighborhood'] in row['municipality']),
                axis=1
            )
            
            # Filter for actual municipalities with property price data
            municipalities = consolidated_data[consolidated_data['is_municipality']].copy()
            
            # Load property price data from verified source
            property_data = pd.read_csv('data/processed/miami_dade_merged_data.csv')
            
            if not municipalities.empty and not property_data.empty:
                # Create property price mapping for municipalities
                property_map = dict(zip(property_data['neighborhood'], property_data['median_property_price']))
                
                # Match municipalities to property prices where possible
                for idx, row in municipalities.iterrows():
                    if row['neighborhood'] in property_map:
                        municipalities.loc[idx, 'median_property_price'] = property_map[row['neighborhood']]
                    elif row['municipality'] in property_map:
                        municipalities.loc[idx, 'median_property_price'] = property_map[row['municipality']]
                
                # Use only municipalities with complete data
                data = municipalities.dropna(subset=['median_property_price']).copy()
                data_source = "Verified Consolidated Municipal Data"
                st.success(f"Using verified real data from {len(data)} municipalities")
                
                # Select relevant columns for affordability analysis
                data_columns = [
                    'neighborhood', 'municipality', 'area_type', 
                    'population', 'median_property_price', 'airbnb_count', 'airbnb_density'
                ]
            else:
                st.warning("Insufficient real data in consolidated files. Trying alternative sources.")
                data = None
                
        except Exception as e:
            logger.error(f"Error loading consolidated real data: {e}", exc_info=True)
            data = None
            
        # 2. Try merged data if consolidated data not available
        if data is None or data.empty:
            try:
                merged_data = pd.read_csv('data/processed/miami_dade_merged_data.csv')
                if merged_data is not None and not merged_data.empty:
                    data = merged_data
                    data_source = "Miami-Dade Merged Data"
                    st.success(f"Using verified merged data with {len(data)} records")
                    data_columns = merged_data.columns.tolist()
            except Exception as e:
                logger.error(f"Error loading merged data: {e}", exc_info=True)
        
        # 3. Try original Miami housing data if other sources not available
        if data is None or data.empty:
            try:
                housing_data = load_miami_housing_data()
                if housing_data is not None and not housing_data.empty:
                    data = housing_data
                    data_source = "Miami Housing Data"
                    # Select relevant columns for affordability
                    price_cols = [col for col in data.columns if any(term in col.lower() for term in 
                                ['price', 'value', 'cost', 'income', 'mortgage', 'rent'])]
                    location_cols = [col for col in data.columns if any(term in col.lower() for term in 
                                ['zip', 'neighborhood', 'area', 'location'])]
                    data_columns = location_cols + price_cols
            except Exception as e:
                logger.error(f"Error loading Miami housing data: {e}", exc_info=True)
                
        # 4. Final fallback to Airbnb data if all else fails
        if data is None or data.empty:
            try:
                airbnb_data = load_airbnb_data()
                if airbnb_data is not None and not airbnb_data.empty:
                    data = airbnb_data
                    data_source = "Airbnb Data"
                    # Select relevant columns for affordability
                    data_columns = [col for col in data.columns if col in [
                        'neighborhood', 'airbnb_count', 'airbnb_density', 'median_airbnb_price']]
            except Exception as e:
                logger.error(f"Error loading Airbnb data: {e}", exc_info=True)
        # 3. Try Zillow data if other sources not available
        if data is None or data.empty:
            try:
                zillow_data = load_zillow_data()
                if zillow_data is not None and not zillow_data.empty:
                    data = zillow_data
                    data_source = "Zillow Data"
                    # For Zillow data, all columns might be relevant
                    data_columns = list(data.columns)
            except Exception as e:
                logger.error(f"Error loading Zillow data: {e}", exc_info=True)
    
    # Check if data is available
    if data is None or data.empty:
        st.error("Could not load any data for affordability analysis. Please check data files.")
        return
    
    # Initialize session state for affordability analysis
    if 'affordability_index' not in st.session_state:
        st.session_state.affordability_index = None
    if 'affordability_dashboard' not in st.session_state:
        st.session_state.affordability_dashboard = None
    if 'affordability_clusters' not in st.session_state:
        st.session_state.affordability_clusters = None
    if 'cluster_profiles' not in st.session_state:
        st.session_state.cluster_profiles = None
    if 'loadings' not in st.session_state:
        st.session_state.loadings = None
    
    # == TAB 1: OVERVIEW & CONTROLS ==
    with tabs[0]:
        st.header("Affordability Index Controls")
        st.write(f"Using {data_source} with {data.shape[0]} records and {data.shape[1]} features.")
        
        # Create columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Selection")
            
            # Filter by numeric columns only for analysis
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            location_cols = [col for col in data.columns if any(term in col.lower() for term in 
                            ['zip', 'neighborhood', 'area', 'location', 'city', 'region'])]
            
            # Allow user to select columns for analysis
            selected_columns = st.multiselect(
                "Select columns for affordability analysis:",
                options=data_columns if data_columns else numeric_cols,
                default=[col for col in data_columns[:5] if col in data.columns] if data_columns else numeric_cols[:5],
                help="Select numeric columns that relate to housing affordability, such as price, rent, income, etc."
            )
            
            # Allow user to select a column to group results by
            group_by_column = None
            if location_cols:
                group_by_column = st.selectbox(
                    "Group results by:",
                    options=['None'] + location_cols,
                    help="Select a categorical column to group affordability results by, such as neighborhood or zipcode."
                )
                if group_by_column == 'None':
                    group_by_column = None
        
        with col2:
            st.subheader("Analysis Options")
            
            # PCA components
            n_components = st.slider(
                "Number of principal components:",
                min_value=1,
                max_value=min(5, len(selected_columns)),
                value=2,
                help="Number of principal components to use for creating the affordability index."
            )
            
            # Clustering options
            cluster_analysis = st.checkbox(
                "Perform cluster analysis",
                value=True,
                help="Group neighborhoods into clusters based on affordability patterns."
            )
            
            if cluster_analysis:
                n_clusters = st.slider(
                    "Number of clusters:",
                    min_value=2,
                    max_value=7,
                    value=4,
                    help="Number of affordability clusters to create."
                )
            else:
                n_clusters = 0
            
        # Button to run analysis
        run_analysis = st.button("Generate Affordability Index", type="primary")
        
        if run_analysis and selected_columns:
            with st.spinner("Creating affordability index..."):
                try:
                    # Create data subset
                    data_subset = data.copy()
                    
                    # If grouping is selected, aggregate data
                    if group_by_column:
                        # Create aggregated dataset
                        agg_functions = {col: 'mean' for col in selected_columns if col != group_by_column}
                        data_subset = data_subset.groupby(group_by_column).agg(agg_functions).reset_index()
                    
                    # Create affordability index
                    affordability_creator = AffordabilityIndexCreator(data_subset)
                    affordability_index = affordability_creator.create_affordability_index(
                        n_components=n_components,
                        columns_to_use=selected_columns
                    )
                    
                    # Store in session state
                    st.session_state.affordability_index = affordability_index
                    st.session_state.loadings = affordability_creator.analyze_component_contributions()
                    
                    # Perform cluster analysis if selected
                    if cluster_analysis and n_clusters > 0:
                        affordability_clusters = affordability_creator.cluster_by_affordability(n_clusters=n_clusters)
                        cluster_profiles = affordability_creator.analyze_clusters()
                        
                        st.session_state.affordability_clusters = affordability_clusters
                        st.session_state.cluster_profiles = cluster_profiles
                    
                    # Create dashboard
                    st.session_state.affordability_dashboard = create_affordability_dashboard(
                        affordability_index,
                        loadings=st.session_state.loadings,
                        cluster_profiles=st.session_state.cluster_profiles if cluster_analysis else None
                    )
                    
                    st.success("Affordability index created successfully!")
                    
                    # Generate summary statistics
                    st.subheader("Affordability Index Summary")
                    
                    # Show summary statistics
                    summary_stats = affordability_index['affordability_score'].describe().round(2)
                    stats_df = pd.DataFrame({
                        'Statistic': summary_stats.index,
                        'Value': summary_stats.values
                    })
                    
                    # Show explained variance
                    explained_variance = affordability_creator.explained_variance * 100
                    
                    # Create columns for stats and variance
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.dataframe(stats_df)
                    
                    with col2:
                        st.write("Explained Variance:")
                        for i, var in enumerate(explained_variance):
                            st.write(f"PC{i+1}: {var:.2f}%")
                        st.write(f"Total: {sum(explained_variance):.2f}%")
                    
                except Exception as e:
                    st.error(f"Error creating affordability index: {str(e)}")
                    logger.error(f"Error creating affordability index: {e}", exc_info=True)
        
        # Display sample of affordability index if available
        if st.session_state.affordability_index is not None:
            st.subheader("Sample of Affordability Index")
            st.dataframe(st.session_state.affordability_index.head(10))
    
    # == TAB 2: AFFORDABILITY DISTRIBUTION ==
    with tabs[1]:
        st.header("Affordability Score Distribution")
        
        if st.session_state.affordability_index is not None and st.session_state.affordability_dashboard is not None:
            if 'distribution' in st.session_state.affordability_dashboard:
                st.plotly_chart(st.session_state.affordability_dashboard['distribution'], use_container_width=True)
            
            # Add contextual explanation
            st.write("""
            ### Understanding the Distribution
            
            The histogram above shows the distribution of affordability scores across different areas. 
            A higher score indicates better affordability. The distribution shape reveals how evenly 
            affordability is spread across Miami neighborhoods.
            
            - **Skewed right**: Most areas have low affordability (common in expensive housing markets)
            - **Skewed left**: Most areas have high affordability
            - **Normal distribution**: Affordability is evenly distributed
            - **Bimodal**: Suggests a divided market with both affordable and unaffordable areas
            
            The mean and median lines show the central tendency of affordability. A large gap between these 
            values indicates outliers in the market.
            """)
            
            # Add neighborhood ranking if possible
            if group_by_column and group_by_column in st.session_state.affordability_index:
                st.subheader("Most and Least Affordable Areas")
                
                # Sort by affordability score
                sorted_areas = st.session_state.affordability_index.sort_values('affordability_score', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Most Affordable Areas**")
                    top_df = sorted_areas[[group_by_column, 'affordability_score']].head(5)
                    st.dataframe(top_df, use_container_width=True)
                
                with col2:
                    st.write("**Least Affordable Areas**")
                    bottom_df = sorted_areas[[group_by_column, 'affordability_score']].tail(5).iloc[::-1]
                    st.dataframe(bottom_df, use_container_width=True)
        else:
            st.info("Please generate an affordability index first using the controls in the Overview tab.")
    
    # == TAB 3: FEATURE CONTRIBUTION ==
    with tabs[2]:
        st.header("Feature Contribution Analysis")
        
        if st.session_state.affordability_index is not None and st.session_state.loadings is not None:
            # Plot component loadings
            st.subheader("Feature Importance in Affordability Index")
            st.write("""
            The chart below shows which features contribute most to the affordability index.
            Features with larger bars have stronger influence on the affordability score.
            """)
            
            # Create component loadings visualization
            if 'loadings' in st.session_state.affordability_dashboard:
                st.plotly_chart(st.session_state.affordability_dashboard['loadings'], use_container_width=True)
            else:
                loadings_plot = plot_component_loadings(st.session_state.loadings)
                st.plotly_chart(loadings_plot, use_container_width=True)
            
            # Add feature correlation analysis
            st.subheader("Affordability Score vs. Selected Feature")
            
            # Select a feature to correlate with affordability
            numeric_cols = [col for col in data.columns if col in selected_columns]
            if numeric_cols:
                feature_column = st.selectbox(
                    "Select a feature to correlate with affordability score:",
                    options=numeric_cols
                )
                
                # Create correlation plot
                corr_plot = plot_affordability_by_feature(
                    st.session_state.affordability_index,
                    feature_column,
                    title=f"Affordability Score vs. {feature_column}",
                    hover_data=group_by_column
                )
                st.plotly_chart(corr_plot, use_container_width=True)
                
                # Calculate and display correlation
                affordability_data = st.session_state.affordability_index
                correlation = affordability_data['affordability_score'].corr(affordability_data[feature_column])
                
                st.write(f"**Correlation coefficient: {correlation:.3f}**")
                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.3:
                    strength = "moderate"
                else:
                    strength = "weak"
                    
                if correlation > 0:
                    direction = "positive"
                else:
                    direction = "negative"
                    
                st.write(f"There is a {strength} {direction} correlation between affordability score and {feature_column}.")
        else:
            st.info("Please generate an affordability index first using the controls in the Overview tab.")
    
    # == TAB 4: NEIGHBORHOOD CLUSTERS ==
    with tabs[3]:
        st.header("Affordability Cluster Analysis")
        
        if (st.session_state.affordability_clusters is not None and 
            st.session_state.cluster_profiles is not None):
            
            # Display cluster visualization
            st.subheader("Affordability Clusters")
            st.write("""
            The visualization below shows how different areas group together based on 
            their affordability patterns. Areas in the same cluster have similar affordability characteristics.
            """)
            
            if 'clusters' in st.session_state.affordability_dashboard:
                st.plotly_chart(st.session_state.affordability_dashboard['clusters'], use_container_width=True)
            else:
                clusters_plot = plot_cluster_analysis(st.session_state.affordability_clusters)
                st.plotly_chart(clusters_plot, use_container_width=True)
            
            # Display cluster profiles
            st.subheader("Cluster Profiles")
            st.write("""
            The chart below shows the characteristics of each affordability cluster.
            It reveals what makes each cluster unique in terms of affordability patterns.
            """)
            
            if 'cluster_profiles' in st.session_state.affordability_dashboard:
                st.plotly_chart(st.session_state.affordability_dashboard['cluster_profiles'], use_container_width=True)
            else:
                profiles_plot = plot_cluster_profiles(st.session_state.cluster_profiles)
                st.plotly_chart(profiles_plot, use_container_width=True)
            
            # Display cluster breakdown
            st.subheader("Cluster Breakdown")
            
            # Calculate counts and percentages
            cluster_counts = st.session_state.affordability_clusters['affordability_cluster'].value_counts().sort_index()
            cluster_percentages = (cluster_counts / cluster_counts.sum() * 100).round(1)
            
            # Create DataFrame
            cluster_breakdown = pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Count': cluster_counts.values,
                'Percentage': cluster_percentages.values
            })
            
            # Add affordability levels if available
            if 'affordability_level' in st.session_state.cluster_profiles.columns:
                cluster_levels = st.session_state.cluster_profiles['affordability_level']
                cluster_breakdown['Affordability Level'] = [
                    cluster_levels.iloc[i] if i < len(cluster_levels) else f"Cluster {i}" 
                    for i in range(len(cluster_breakdown))
                ]
            
            st.dataframe(cluster_breakdown)
            
            # Show areas in each cluster if grouping column is available
            if group_by_column:
                st.subheader("Areas by Cluster")
                selected_cluster = st.selectbox(
                    "Select a cluster to view areas:",
                    options=cluster_breakdown['Cluster'].tolist()
                )
                
                # Filter areas in selected cluster
                cluster_areas = st.session_state.affordability_clusters[
                    st.session_state.affordability_clusters['affordability_cluster'] == selected_cluster
                ]
                
                # Show areas and their affordability scores
                areas_df = cluster_areas[[group_by_column, 'affordability_score']].sort_values('affordability_score', ascending=False)
                st.dataframe(areas_df)
        
        elif st.session_state.affordability_index is not None:
            st.warning("Cluster analysis was not performed. Please enable cluster analysis in the Overview tab.")
        else:
            st.info("Please generate an affordability index first using the controls in the Overview tab.")
    
    # == TAB 5: STRATEGIC INSIGHTS ==
    with tabs[4]:
        st.header("Strategic Insights & Recommendations")
        
        if st.session_state.affordability_index is not None:
            # Calculate market statistics
            affordability_data = st.session_state.affordability_index
            mean_score = affordability_data['affordability_score'].mean()
            median_score = affordability_data['affordability_score'].median()
            std_score = affordability_data['affordability_score'].std()
            min_score = affordability_data['affordability_score'].min()
            max_score = affordability_data['affordability_score'].max()
            
            # Generate insights based on data
            st.subheader("Market Overview")
            
            if median_score < 30:
                market_state = "highly unaffordable"
            elif median_score < 50:
                market_state = "moderately unaffordable"
            elif median_score < 70:
                market_state = "moderately affordable"
            else:
                market_state = "highly affordable"
            
            if std_score > 20:
                variability = "high variability"
            elif std_score > 10:
                variability = "moderate variability"
            else:
                variability = "consistent affordability"
            
            st.write(f"""
            The Miami housing market shows **{market_state}** conditions with **{variability}** across areas.
            The median affordability score is **{median_score:.1f}** (on a scale of 0-100), with scores ranging
            from **{min_score:.1f}** to **{max_score:.1f}**.
            """)
            
            # Create recommendations for different stakeholders
            st.subheader("Stakeholder Recommendations")
            
            st.write("#### For Policymakers")
            if median_score < 50:
                st.write("""
                - **Affordable Housing Initiatives**: Implement policies to increase affordable housing supply
                - **Zoning Reviews**: Consider revising zoning laws to allow for higher density development
                - **Rent Control**: Evaluate selective rent control measures in severely unaffordable areas
                - **Tax Incentives**: Provide tax breaks for affordable housing developers
                """)
            else:
                st.write("""
                - **Maintain Affordability**: Protect existing affordable housing stock
                - **Monitor Development**: Ensure new developments include affordable options
                - **Homeownership Programs**: Expand first-time homebuyer assistance in moderately affordable areas
                """)
            
            st.write("#### For Real Estate Investors")
            
            if 'cluster_profiles' in st.session_state and st.session_state.cluster_profiles is not None:
                # Identify emerging areas (middle affordability with improving trends)
                if max_score - min_score > 30:  # Wide range of affordability
                    st.write("""
                    - **Value Opportunities**: Look for properties in mid-range affordability clusters that border higher-value areas
                    - **Long-term Growth**: Focus on areas showing improving affordability trends for long-term appreciation
                    - **Diversification**: Spread investments across different affordability clusters to balance risk
                    """)
                else:
                    st.write("""
                    - **Yield Focus**: In this uniform market, focus on properties with better yield metrics
                    - **Competitive Edge**: Invest in property improvements to stand out in this homogeneous market
                    - **Niche Markets**: Consider specialized housing types that serve specific market segments
                    """)
            else:
                if median_score < 40:
                    st.write("""
                    - **Affordable Rentals**: Focus on affordable rental properties with stable cash flow
                    - **Value-Add Opportunities**: Look for properties that can be renovated to improve value while maintaining affordability
                    - **Housing Subsidies**: Explore government programs for affordable housing investment
                    """)
                else:
                    st.write("""
                    - **Growth Areas**: Target neighborhoods with rising affordability scores for appreciation potential
                    - **Mixed-Use Development**: Consider properties that combine residential and commercial uses
                    - **Premium Segment**: In more affordable markets, luxury properties may offer differentiation
                    """)
            
            st.write("#### For Residents & Homebuyers")
            
            if median_score < 40:
                st.write("""
                - **Co-living Options**: Consider shared housing arrangements to reduce costs
                - **Commute Trade-offs**: Evaluate areas with longer commutes but better affordability
                - **First-time Buyer Programs**: Explore government assistance programs for first-time homebuyers
                - **Financial Planning**: Work with financial advisors to develop realistic housing budgets
                """)
            else:
                st.write("""
                - **Timing Considerations**: Current market conditions offer reasonable affordability
                - **Location Flexibility**: Compare multiple neighborhoods to find the best affordability
                - **Long-term Value**: Consider neighborhoods with improving services and amenities
                """)
            
            # Add methodology explanation
            st.subheader("Methodology")
            st.write("""
            The affordability index was created using Principal Component Analysis (PCA), which combines
            multiple housing metrics into a single score. This approach allows us to:
            
            1. **Reduce dimensionality**: Combine multiple variables into a single meaningful index
            2. **Remove redundancy**: Account for correlations between different metrics
            3. **Capture variance**: Focus on the aspects that vary most across neighborhoods
            4. **Weight appropriately**: Allow the data to determine the importance of each factor
            
            The resulting score provides a comprehensive measure of housing affordability that
            accounts for multiple dimensions of the housing market.
            """)
            
            if st.session_state.loadings is not None:
                st.write("#### Top Contributing Factors")
                
                # Get top 5 features by absolute loading value
                loadings = st.session_state.loadings
                top_features = loadings.sort_values('PC1_abs', ascending=False).head(5).index.tolist()
                
                for i, feature in enumerate(top_features):
                    st.write(f"{i+1}. **{feature}**")
        else:
            st.info("Please generate an affordability index first using the controls in the Overview tab.")

if __name__ == "__main__":
    show_affordability_analysis()
