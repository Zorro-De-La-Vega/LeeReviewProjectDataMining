"""
Geospatial Hotspot Analysis Page

This module provides the Streamlit interface for geospatial hotspot analysis
which identifies spatial patterns in housing and Airbnb data across Miami-Dade County.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import datetime
import traceback
from streamlit.components.v1 import html

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing both implementations - regular and fallback
try:
    from src.models.geospatial_analysis import GeospatialHotspotAnalysis
    geospatial_available = True
    using_fallback = False
    logger.info("Successfully imported GeospatialHotspotAnalysis")
except ImportError as e:
    geospatial_available = False
    using_fallback = True
    logger.warning(f"Error importing GeospatialHotspotAnalysis: {e}")
    logger.info("Will try to import fallback implementation")

# Import fallback implementation
try:
    from src.models.geospatial_fallback import LightweightGeospatialAnalysis
    geospatial_fallback_available = True
    logger.info("Successfully imported LightweightGeospatialAnalysis")
except ImportError as e:
    geospatial_fallback_available = False
    logger.error(f"Error importing LightweightGeospatialAnalysis: {e}")

# Import standard helper utilities
from ..utils.helpers import ensure_data_loaded, load_config

# Load config
CONFIG = load_config()

def show_geospatial_hotspot_analysis(data_loader, agent=None):
    """
    Show the geospatial hotspot analysis page.
    
    Args:
        data_loader: DataLoader instance
        agent: Optional agent instance for insights
    """
    st.header("Geospatial Hotspot Analysis")
    
    # Check if geospatial analysis is available
    if not geospatial_available and not geospatial_fallback_available:
        st.error("""
        ## ⚠️ Geospatial Analysis Not Available
        
        The geospatial analysis module requires additional dependencies:
        
        **For full functionality**:
        ```
        pip install pysal esda libpysal geopandas contextily folium
        ```
        
        **For basic functionality**:
        ```
        pip install folium sklearn
        ```
        
        Please install at least the basic dependencies to use this feature.
        """)
        return
    
    # Show note if using fallback implementation
    if using_fallback:
        st.info("""
        ℹ️ Using lightweight geospatial analysis implementation.
        
        For full functionality with spatial autocorrelation and advanced statistics, 
        install additional dependencies:
        ```
        pip install pysal esda libpysal geopandas contextily
        ```
        """)
    
    # Introduction
    st.markdown("""
    This analysis identifies spatial patterns and hotspots in housing and Airbnb data 
    across Miami-Dade County. Hotspots are areas with unusually high concentrations 
    of a particular feature (e.g., high housing prices, high Airbnb density).
    
    ### What This Analysis Tells You
    
    - **Where hotspots are located** across Miami-Dade County
    - **How intense** these concentrations are
    - **Patterns and relationships** between different variables
    - **Recommendations** for stakeholders based on spatial patterns
    """)
    
    # Initialize session state for analysis results if not already present
    if 'geospatial_results' not in st.session_state:
        st.session_state.geospatial_results = None
    if 'geospatial_analysis_complete' not in st.session_state:
        st.session_state.geospatial_analysis_complete = False
    if 'selected_variable' not in st.session_state:
        st.session_state.selected_variable = None
    
    # Load data
    data = None
    try:
        # Attempt to get combined or Airbnb data
        data = data_loader.get_combined_data()
        if data is None or data.empty:
            data = data_loader.get_airbnb_data()
        
        if data is None or data.empty:
            st.warning("Could not load data. Please ensure your data files are available.")
            return
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {e}", exc_info=True)
        return
    
    # Create analysis tabs
    tabs = st.tabs(["Analysis Controls", "Heatmap Visualization", "Hotspot Detection", "Hotspot Report"])
    
    # Tab 1: Analysis Controls
    with tabs[0]:
        st.subheader("Analysis Configuration")
        
        # Variable selection
        # Get numeric columns for analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Try to find the most relevant variables
        relevant_vars = [col for col in numeric_cols if any(term in col.lower() 
                        for term in ['price', 'density', 'count', 'value', 'rent', 'income'])]
        
        # Default to the first relevant variable or the first numeric column
        default_var = relevant_vars[0] if relevant_vars else numeric_cols[0] if numeric_cols else None
        
        # Let user select the variable to analyze
        variable = st.selectbox(
            "Select variable for hotspot analysis:", 
            options=numeric_cols,
            index=numeric_cols.index(default_var) if default_var in numeric_cols else 0,
            help="Choose which numeric variable to analyze for spatial patterns."
        )
        
        # Show variable statistics
        if variable in data.columns:
            var_stats = data[variable].describe()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean", f"{var_stats['mean']:.2f}")
            with col2:
                st.metric("Median", f"{var_stats['50%']:.2f}")
            with col3:
                st.metric("Std Dev", f"{var_stats['std']:.2f}")
            
            # Small histogram of the variable
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(data[variable].dropna(), bins=30, alpha=0.7)
            ax.set_title(f"Distribution of {variable}")
            ax.set_xlabel(variable)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        # Analysis parameters
        st.subheader("Hotspot Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epsilon = st.slider(
                "Spatial proximity threshold (Epsilon):", 
                min_value=0.001, 
                max_value=0.05, 
                value=0.01,
                step=0.001,
                help="Distance threshold for clustering points (higher values create larger clusters)"
            )
            
            hotspot_percentile = st.slider(
                "Hotspot threshold percentile:", 
                min_value=50, 
                max_value=95, 
                value=75,
                help="Percentile threshold to classify clusters as hotspots"
            )
        
        with col2:
            min_samples = st.slider(
                "Minimum points per cluster:", 
                min_value=3, 
                max_value=15, 
                value=5,
                help="Minimum number of points required to form a cluster"
            )
            
            include_all_points = st.checkbox(
                "Show all data points", 
                value=True,
                help="If checked, shows all individual points on the map (may be slower for large datasets)"
            )
        
        # Run analysis button
        if st.button("Run Geospatial Hotspot Analysis", type="primary"):
            with st.spinner("Running geospatial hotspot analysis..."):
                try:
                    # Initialize the appropriate analyzer
                    if using_fallback or not geospatial_available:
                        analyzer = LightweightGeospatialAnalysis()
                    else:
                        analyzer = GeospatialHotspotAnalysis()
                    
                    # Load data from DataFrame
                    # First make sure we have lat/lon columns
                    lat_cols = [col for col in data.columns if 'lat' in col.lower()]
                    lng_cols = [col for col in data.columns if any(term in col.lower() for term in ['lng', 'lon', 'long'])]
                    
                    if not lat_cols or not lng_cols:
                        st.error("Latitude and longitude columns are required for geospatial analysis.")
                        return
                    
                    # Standardize column names
                    data_copy = data.copy()
                    data_copy = data_copy.rename(columns={
                        lat_cols[0]: 'latitude',
                        lng_cols[0]: 'longitude'
                    })
                    
                    # Transfer data to analyzer
                    analyzer.data = data_copy
                    
                    # Run complete analysis
                    results = analyzer.run_full_analysis(
                        variable=variable, 
                        include_all_plots=True
                    )
                    
                    if results:
                        # Save results to session state
                        st.session_state.geospatial_results = results
                        st.session_state.geospatial_analysis_complete = True
                        st.session_state.selected_variable = variable
                        
                        # Show success message
                        st.success(f"Geospatial hotspot analysis completed successfully for {variable}!")
                        st.info("Navigate to the other tabs to view the results.")
                    else:
                        st.error("Analysis failed. See logs for details.")
                
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    logger.error(f"Error in geospatial analysis: {e}\n{traceback.format_exc()}")
    
    # Tab 2: Heatmap Visualization
    with tabs[1]:
        st.subheader("Heatmap Visualization")
        
        if st.session_state.geospatial_analysis_complete and st.session_state.geospatial_results:
            results = st.session_state.geospatial_results
            
            st.markdown(f"""
            ### Heatmap of {st.session_state.selected_variable}
            
            This heatmap shows the spatial distribution of {st.session_state.selected_variable} 
            across Miami-Dade County. Areas with higher values are shown in red, while areas 
            with lower values are shown in blue.
            
            Use this visualization to identify general patterns before looking at specific hotspots.
            """)
            
            # Display the heatmap
            if 'heatmap_path' in results and os.path.exists(results['heatmap_path']):
                # Read the HTML file
                with open(results['heatmap_path'], 'r') as f:
                    heatmap_html = f.read()
                
                # Display in a container with fixed height
                st.components.v1.html(heatmap_html, height=600)
            else:
                st.warning("Heatmap visualization not available.")
        else:
            st.info("Run the analysis in the 'Analysis Controls' tab to generate a heatmap.")
    
    # Tab 3: Hotspot Detection
    with tabs[2]:
        st.subheader("Hotspot Detection")
        
        if st.session_state.geospatial_analysis_complete and st.session_state.geospatial_results:
            results = st.session_state.geospatial_results
            
            st.markdown(f"""
            ### Hotspot Detection for {st.session_state.selected_variable}
            
            This map shows the detected hotspots (clusters of high values) for {st.session_state.selected_variable}.
            - **Red clusters**: Hotspots (high value areas)
            - **Blue clusters**: Regular clusters
            - **Gray points**: Unclustered points
            
            Click on clusters or individual points for more information.
            """)
            
            # Display the hotspot map
            if 'hotspot_map_path' in results and os.path.exists(results['hotspot_map_path']):
                # Read the HTML file
                with open(results['hotspot_map_path'], 'r') as f:
                    hotspot_html = f.read()
                
                # Display in a container with fixed height
                st.components.v1.html(hotspot_html, height=600)
            else:
                st.warning("Hotspot detection map not available.")
            
            # Display statistical visualization if available
            if 'distribution_plot_path' in results and results['distribution_plot_path'] and os.path.exists(results['distribution_plot_path']):
                st.subheader("Statistical Analysis of Hotspots")
                st.image(results['distribution_plot_path'])
                
                st.markdown("""
                This statistical analysis compares the distribution of values in hotspots versus non-hotspots.
                The significant difference between these distributions confirms that our hotspot detection is 
                identifying meaningful clusters.
                """)
        else:
            st.info("Run the analysis in the 'Analysis Controls' tab to generate hotspot detection results.")
    
    # Tab 4: Hotspot Report
    with tabs[3]:
        st.subheader("Hotspot Analysis Report")
        
        if st.session_state.geospatial_analysis_complete and st.session_state.geospatial_results:
            results = st.session_state.geospatial_results
            
            if 'report' in results and results['report']:
                report = results['report']
                
                # Overview statistics
                st.markdown(f"""
                ### Summary Statistics
                
                - **Variable analyzed**: {report['variable']}
                - **Number of hotspot clusters**: {report['num_hotspot_clusters']}
                - **Total points in hotspots**: {report['total_hotspots']:.0f} ({report['percent_in_hotspots']:.1f}% of total)
                - **Average points per hotspot**: {report.get('avg_hotspot_size', 'N/A'):.1f}
                - **Average distance to downtown**: {report.get('avg_distance_to_downtown', 'N/A'):.2f} km
                - **Highest intensity ratio**: {report.get('highest_intensity', 'N/A'):.2f}x the average
                """)
                
                # Top hotspots table
                st.subheader("Top Hotspots")
                
                if 'hotspot_details' in report and report['hotspot_details']:
                    # Create a dataframe for the table
                    hotspot_df = pd.DataFrame(report['hotspot_details'])
                    
                    # Reorder and rename columns for display
                    display_cols = [
                        'location', 'mean_value', 'point_count', 
                        'relative_intensity', 'distance_to_downtown_km'
                    ]
                    rename_map = {
                        'location': 'Location',
                        'mean_value': f'Mean {report["variable"]}',
                        'point_count': 'Count',
                        'relative_intensity': 'Intensity Ratio',
                        'distance_to_downtown_km': 'Distance to Downtown (km)'
                    }
                    
                    # Filter and display
                    if all(col in hotspot_df.columns for col in display_cols):
                        display_df = hotspot_df[display_cols].rename(columns=rename_map)
                        display_df = display_df.set_index('Location')
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.dataframe(hotspot_df, use_container_width=True)
                else:
                    st.info("No hotspot details available in the report.")
                
                # Strategic insights
                st.subheader("Strategic Insights")
                
                # Generate insights based on the variable
                var_lower = report['variable'].lower()
                
                if any(term in var_lower for term in ['price', 'value', 'cost']):
                    st.markdown("""
                    #### For Investors and Developers
                    - **High-Value Areas**: Focus on these hotspots for luxury developments or premium services
                    - **Growth Potential**: Look at hotspots with high intensity but moderate housing prices
                    - **Return on Investment**: Compare price hotspots with rental rate or occupancy hotspots
                    
                    #### For Policymakers
                    - **Affordability Concerns**: Monitor price hotspots for potential affordability issues
                    - **Tax Revenue**: Higher property values in these areas contribute more to tax base
                    - **Opportunity Areas**: Consider incentives for development outside of price hotspots
                    """)
                elif any(term in var_lower for term in ['density', 'count', 'listing']):
                    st.markdown("""
                    #### For Regulators and Policymakers
                    - **Regulation Focus**: Prioritize these concentration hotspots for rental regulations
                    - **Market Saturation**: High concentration areas may be at risk of market saturation
                    - **Impact Assessment**: Monitor these areas for impacts on long-term rental availability
                    
                    #### For Tourism Industry
                    - **Service Opportunities**: Hotspots represent areas with high visitor concentration
                    - **Marketing Strategy**: Target these areas for tourism-related services and amenities
                    - **Diversification**: Consider promoting less concentrated areas to spread tourism impact
                    """)
                elif any(term in var_lower for term in ['income', 'wealth']):
                    st.markdown("""
                    #### For Businesses and Retailers
                    - **Market Opportunity**: These income hotspots represent areas with high spending power
                    - **Retail Strategy**: Target premium or luxury offerings in these high-income clusters
                    - **Service Differentiation**: Adjust pricing and offerings based on income geography
                    
                    #### For Community Development
                    - **Resource Allocation**: Compare income hotspots with other socioeconomic indicators
                    - **Opportunity Gaps**: Identify areas with large income disparities near hotspots
                    - **Educational Investment**: Compare income patterns with educational achievement
                    """)
                elif any(term in var_lower for term in ['rent', 'rental']):
                    st.markdown("""
                    #### For Rental Property Investors
                    - **Yield Optimization**: Focus on hotspots with high rent-to-price ratios
                    - **Competitive Positioning**: Understand rental pricing benchmarks in hotspot areas
                    - **Target Markets**: Different hotspots may appeal to different tenant demographics
                    
                    #### For Housing Advocates
                    - **Affordability Challenges**: Monitor rent hotspots for displacement risks
                    - **Housing Policy**: Use this data to advocate for affordable housing in high-rent areas
                    - **Tenant Protection**: Areas with rapidly increasing rent may need tenant protections
                    """)
                else:
                    st.markdown("""
                    #### General Strategic Insights
                    - **Comparative Analysis**: Compare these hotspots with other variables to identify relationships
                    - **Temporal Tracking**: Monitor how these hotspots evolve over time
                    - **Stakeholder Engagement**: Use this analysis to facilitate data-driven conversations
                    - **Decision Support**: Prioritize resources and attention based on hotspot intensity
                    """)
            else:
                st.warning("Report data not available. The analysis may not have completed successfully.")
        else:
            st.info("Run the analysis in the 'Analysis Controls' tab to generate a hotspot analysis report.")


if __name__ == "__main__":
    # For testing
    show_geospatial_hotspot_analysis(None)
