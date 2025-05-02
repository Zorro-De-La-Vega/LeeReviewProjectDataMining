#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Data Mining Analysis Page

This module contains the Streamlit interface for advanced data mining
visualizations and insights for the Miami Housing Impact Hub.
"""

import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import importlib.util
import time
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# Import optimization utilities
from ..utils.streamlit_optimizations import st_cached_data, st_cached_resource, lazy_load, optimize_dataframe, ProgressSpinner

# Import core dependencies and utilities
from ..utils.data_loader import DataLoader
from ..utils.helpers import ensure_data_loaded, load_config
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import traceback

# Import advanced models
from ...models.advanced_mining import NeighborhoodSegmentation
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the feature helpers module
from .feature_helpers import show_feature_coming_soon
from .sentiment_analysis import show_sentiment_analysis
from .affordability_analysis import show_affordability_analysis
from .geospatial_analysis import show_geospatial_hotspot_analysis

# Import models with try/except to handle missing dependencies
try:
    from src.models.advanced_mining import NeighborhoodSegmentation, RandomForestPredictor, MonteCarloInvestmentSimulator
    advanced_mining_available = True
except ImportError as e:
    logging.error(f"Error importing advanced_mining: {e}", exc_info=True)
    advanced_mining_available = False

# Check if sentiment analysis is available
try:
    from src.models.sentiment_analysis import ListingSentimentAnalyzer
    sentiment_analysis_available = True
except ImportError as e:
    logging.error(f"Error importing sentiment_analysis: {e}", exc_info=True)
    sentiment_analysis_available = False

# Check if affordability analysis is available
try:
    from src.models.affordability_index import AffordabilityIndexCreator
    affordability_analysis_available = True
except ImportError as e:
    logging.error(f"Error importing affordability_index: {e}", exc_info=True)
    affordability_analysis_available = False

# Check if geospatial analysis is available
try:
    from src.models.geospatial_fallback import LightweightGeospatialAnalysis
    geospatial_available = True
except ImportError as e:
    logging.error(f"Error importing geospatial_fallback: {e}", exc_info=True)
    geospatial_available = False

try:
    from src.models.time_series_analysis import TimeSeriesDecomposer
    time_series_available = True
except ImportError as e:
    logging.error(f"Error importing time_series_analysis: {e}", exc_info=True)
    time_series_available = False

try:
    from src.models.association_rules import AmenityAssociationMiner
    association_rules_available = True
except ImportError as e:
    logging.error(f"Error importing association_rules: {e}", exc_info=True)
    association_rules_available = False

# Check if geospatial packages are available
geospatial_available = False

# Define a lightweight fallback implementation for geospatial analysis
class LightweightGeospatialAnalysis:
    """A lightweight fallback implementation when geospatial dependencies are missing."""
    
    def __init__(self, data=None):
        self.data = data
        logging.info("Using lightweight geospatial analysis implementation")
    
    def run_hotspot_analysis(self, data=None, variable=None):
        """Simplified hotspot analysis that returns the data with a mock score."""
        if data is None:
            data = self.data
        if data is None:
            return pd.DataFrame()
            
        # Create a copy to avoid modifying the original
        result = data.copy()
        
        # Add mock hotspot scores - just ranks the values
        if variable and variable in result.columns:
            result['hotspot_score'] = result[variable].rank(pct=True)
        else:
            result['hotspot_score'] = 0.5  # Neutral score
            
        return result

# Try to import the real implementation, fall back to lightweight version if not available
try:
    import pysal
    import contextily as ctx
    from src.models.geospatial_analysis import GeospatialHotspotAnalysis
    geospatial_available = True
    logging.info("Successfully imported full geospatial analysis dependencies")
except ImportError as e:
    logging.warning(f"Error importing GeospatialHotspotAnalysis: {e}")
    logging.info("Will try to import fallback implementation")
    
    try:
        # Use the lightweight implementation defined above
        GeospatialHotspotAnalysis = LightweightGeospatialAnalysis
        geospatial_available = True
        logging.info("Successfully imported LightweightGeospatialAnalysis")
    except Exception as inner_e:
        logging.error(f"Error setting up fallback implementation: {inner_e}", exc_info=True)
        GeospatialHotspotAnalysis = None

# Import visualization functions with try/except blocks
try:
    from src.visualization.association_plots import plot_association_rules
    plots_association_available = True
except ImportError as e:
    logging.error(f"Error importing association plot functions: {e}", exc_info=True)
    plots_association_available = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def show_advanced_analysis_page(data_loader, agent=None):
    """
    Display the advanced data mining analysis page.
    
    This page provides access to sophisticated data mining insights
    like k-means clustering, predictive modeling, and more.
    
    Args:
        data_loader: DataLoader instance with access to all datasets
        agent: HousingImpactAgent instance for proactive insights
    """
    logger.info("Displaying Advanced Analysis page.") # Added

    # Ensure data is loaded (attempt primary file, then fallback) before proceeding
    data_loader.load_data()
    
    # Performance Tracking - measure page load time
    start_time = time.time()
    
    st.header("Advanced Data Mining Analysis")
    
    # Performance optimization: Cache heavyweight page elements
    @st_cached_data(ttl=3600)
    def get_page_description():
        return """
        This section provides advanced data mining insights derived from Miami-Dade housing and Airbnb data.
        These analyses go beyond basic statistics to identify patterns, predict trends, and suggest 
        strategies for different stakeholders.
        """
    
    # Explanation of this page
    st.markdown(get_page_description())
    
    # Performance optimization: Create tabs more efficiently by avoiding redundant checks
    # Create tabs for different analyses based on available features
    feature_availability = {
        "Neighborhood Segmentation": True,  # Always available 
        "Price Prediction": True,          # Always available
        "Investment Simulation": True,     # Always available
        "Time Series Decomposition": time_series_available,
        "Association Rule Mining": association_rules_available,
        "Sentiment Analysis": sentiment_analysis_available,
        "Affordability Index": affordability_analysis_available,
        "Geospatial Hotspots": geospatial_available
    }
    
    # Build tab titles list efficiently
    tab_titles = []
    for feature, is_available in feature_availability.items():
        if is_available:
            tab_titles.append(feature)
        elif feature == "Geospatial Hotspots":
            # Special case for geospatial which shows "Coming Soon" when unavailable
            tab_titles.append(f"{feature} (Coming Soon)")
    
    # Create tabs efficiently
    tabs = st.tabs(tab_titles)
    
    # Performance optimization: Create function map more efficiently
    function_map = {}
    
    # Define wrapper functions to handle parameter mismatches
    def neighborhood_segmentation_wrapper(data_loader, agent):
        logger.debug("Calling show_neighborhood_segmentation") # Added
        return show_neighborhood_segmentation(data_loader)
        
    # Define all available functions
    all_functions = {
        "Neighborhood Segmentation": neighborhood_segmentation_wrapper,
        "Price Prediction": show_price_prediction,
        "Investment Simulation": show_investment_simulation,
        "Time Series Decomposition": show_time_series_decomposition,
        "Association Rule Mining": show_association_rules_analysis,
        "Sentiment Analysis": show_sentiment_analysis,
        "Affordability Index": show_affordability_analysis,
        "Geospatial Hotspots": show_geospatial_hotspot_analysis,
    }
    
    # Build function map for available tabs
    for i, title in enumerate(tab_titles):
        # Handle the "Coming Soon" suffix
        base_title = title.split(' (Coming Soon)')[0]
        
        if base_title in all_functions:
            function_map[title] = all_functions[base_title]
        else:
            # Fallback to coming soon for any unmatched or unavailable features
            function_map[title] = show_feature_coming_soon
    
    # Performance optimization: Execute tab functions only when the tab is selected
    # This is a critical optimization that prevents all tabs from being computed simultaneously
    
    # Store the last tab selection to prevent recomputation when tab doesn't change
    if 'last_tab_selection' not in st.session_state:
        st.session_state['last_tab_selection'] = tab_titles[0]
    
    # Get current tab from query params or default to first tab
    tab_options = list(tab_titles)
    try:
        # Use the updated experimental API for getting query params
        current_tab_index = tab_options.index(st.experimental_get_query_params().get('tab', [tab_options[0]])[0])
    except ValueError:
        current_tab_index = 0 # Default to first tab if param is invalid
    except Exception as e: # Catch other potential errors during param access
        logger.warning(f"Error getting query parameter 'tab': {e}")
        current_tab_index = 0
    
    selected_tab = st.radio("Select Analysis:", tab_options, index=current_tab_index, horizontal=True, label_visibility="collapsed")
    
    # Update query params if tab changed
    if selected_tab != tab_options[current_tab_index]:
        try:
             # Use the updated experimental API for setting query params
             st.experimental_set_query_params(tab=selected_tab)
        except Exception as e:
             logger.warning(f"Could not set query params: {e}") # Log instead of print
    
    # Display content for the selected tab
    if selected_tab in function_map:
        try:
            # Check if tab selection changed
            if selected_tab != st.session_state['last_tab_selection']:
                # Clear previous tab's outputs from cache
                st.cache_data.clear()
                st.session_state['last_tab_selection'] = selected_tab
            
            # Execute with timing measurements
            tab_start_time = time.time()
            logger.info(f"Rendering tab: {selected_tab}") # Added
            function_map[selected_tab](data_loader, agent)
            tab_elapsed = time.time() - tab_start_time
            
            # Log performance metrics
            st.session_state.setdefault('tab_performance', {})
            st.session_state['tab_performance'][selected_tab] = {
                'last_render_time': tab_elapsed,
                'renders': st.session_state.get('tab_performance', {}).get(selected_tab, {}).get('renders', 0) + 1
            }
            
        except Exception as e:
            st.error(f"An error occurred in the '{selected_tab}' section.")
            with st.expander("View Error Details"):
                st.code(traceback.format_exc())
            logger.error(f"Error in {selected_tab}: {e}\n{traceback.format_exc()}")
    else:
        logger.warning(f"No function mapped for tab: {selected_tab}") # Added
        st.warning(f"Analysis '{selected_tab}' is not yet implemented.")
    
    # Log overall page performance
    page_elapsed = time.time() - start_time
    st.session_state.setdefault('page_performance', {})
    st.session_state['page_performance']['advanced_analysis'] = {
        'last_render_time': page_elapsed,
        'renders': st.session_state.get('page_performance', {}).get('advanced_analysis', {}).get('renders', 0) + 1
    }


def show_time_series_decomposition(data_loader):
    """Displays time series decomposition analysis for ZORI.

    Args:
        data_loader: DataLoader instance.
    """
    st.subheader("Time Series Decomposition of Rental Prices (ZORI)")
    st.markdown("""
    This analysis decomposes the Zillow Observed Rent Index (ZORI) for the Miami-Dade area 
    into its underlying components: trend, seasonality, and residual (noise). 
    Understanding these components helps identify long-term price movements, 
    cyclical patterns, and irregular fluctuations.
    """)

    try:
        # Load ZORI data
        zori_df = data_loader.get_zori_data()
        if zori_df is None or zori_df.empty:
            st.warning("ZORI data could not be loaded. Cannot perform time series analysis.")
            return
        
        # --- Data Preprocessing --- 
        # Ensure 'Date' column exists and is datetime
        if 'Date' not in zori_df.columns:
             st.error("ZORI DataFrame must contain a 'Date' column.")
             return
        zori_df['Date'] = pd.to_datetime(zori_df['Date'], errors='coerce')
        zori_df = zori_df.dropna(subset=['Date'])
        
        # Ensure 'ZORI' column exists and is numeric
        if 'ZORI' not in zori_df.columns:
            st.error("ZORI DataFrame must contain a 'ZORI' column.")
            return
        zori_df['ZORI'] = pd.to_numeric(zori_df['ZORI'], errors='coerce')
        
        # Set index, sort, and select the series
        zori_df = zori_df.set_index('Date').sort_index()
        time_series = zori_df['ZORI'].dropna() # Drop NaNs before decomposition
        
        if time_series.empty:
            st.warning("No valid ZORI data points found after preprocessing.")
            return
            
        # Display raw data preview
        st.markdown("#### ZORI Time Series Preview")
        st.line_chart(time_series)
        
        with st.expander("View Raw ZORI Data"):
            st.dataframe(time_series)

        # --- Analysis Configuration --- 
        st.markdown("#### Decomposition Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Decomposition Model", ('additive', 'multiplicative'), index=0,
                                      help="'additive' assumes seasonal effects are constant, 'multiplicative' assumes they scale with the trend.")
        with col2:
            # Infer frequency or default to monthly (12)
            freq = pd.infer_freq(time_series.index)
            default_period = 12 if freq and ('M' in freq or 'm' in freq) else 4 # Default quarterly if not monthly
            if len(time_series) < default_period * 2:
                 default_period = max(2, len(time_series) // 2) # Adjust if series too short
            
            period = st.number_input("Seasonality Period", min_value=2, 
                                     max_value=max(24, len(time_series)//2), 
                                     value=default_period, step=1, 
                                     help="The number of observations per seasonal cycle (e.g., 12 for monthly data with annual seasonality).")

        # --- Run Analysis --- 
        if st.button("Decompose Time Series"):
            if len(time_series) < period * 2:
                 st.warning(f"Time series is too short for the selected period ({period}). Needs at least {period*2} data points.")
            else:
                with st.spinner("Performing decomposition..."):
                    try:
                        decomposer = TimeSeriesDecomposer(time_series)
                        success = decomposer.decompose(model=model_type, period=period)

                        if success:
                            st.success("Decomposition successful!")
                            fig = decomposer.plot_decomposition(title=f"ZORI Time Series Decomposition ({model_type.capitalize()})")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Interpretation guidance
                            st.markdown("**Interpretation:**")
                            st.markdown("""
                            - **Observed:** The original ZORI data.
                            - **Trend:** The long-term progression of rental prices, ignoring seasonality and noise.
                            - **Seasonal:** Patterns that repeat over the specified period (e.g., yearly cycles).
                            - **Residual:** The leftover noise or irregular component after removing trend and seasonality.
                            """)
                        else:
                            st.warning("Decomposition could not be performed. The time series might be too short, have too many missing values, or other issues. Check logs.")
                    
                    except Exception as e:
                        st.error("An error occurred during decomposition.")
                        logger.error(f"Time Series Decomposition failed: {e}\n{traceback.format_exc()}")

    except FileNotFoundError:
        st.error("ZORI data file not found. Please ensure 'zori_data.csv' is in the processed data directory.")
    except Exception as e:
        st.error("An unexpected error occurred while loading or processing ZORI data.")
        logger.error(f"Error in time series decomposition section: {e}\n{traceback.format_exc()}")


def show_association_rules_analysis(data_loader, agent=None):
    st.subheader("Association Rule Mining for Listing Amenities")
    st.markdown("Discover relationships between listing amenities and characteristics like price range or rating.")
 
    listings_df = data_loader.get_data('listings_processed')
    if listings_df is None or listings_df.empty:
        st.error("Processed listings data is not available. Please run the data processing pipeline.")
        return
 
    # --- Configuration --- 
    st.sidebar.subheader("Association Rules Settings")
    target_options = ['price', 'review_scores_rating'] # Add other potential numeric columns if needed
    target_col = st.sidebar.selectbox("Target Variable (to discretize)", target_options, index=0,
                                    help="Select the variable to find associations with (e.g., high price, good rating).")
 
    min_support = st.sidebar.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001, format="%.3f",
                                   help="Minimum frequency of an itemset in the dataset.")
    metric = st.sidebar.selectbox("Rule Metric", ['lift', 'confidence'], index=0,
                                help="Metric used to evaluate the strength of the association.")
    min_threshold = st.sidebar.slider(f"Minimum {metric.capitalize()}", 0.1, 2.0, 1.0 if metric == 'lift' else 0.5, 0.1,
                                     help=f"Minimum threshold for the selected rule metric ({metric}).")
    n_bins = st.sidebar.slider("Number of Target Bins", 2, 5, 3, 1,
                              help="How many categories to divide the target variable into (e.g., Low/Med/High Price).")
    min_amenity_freq = st.sidebar.slider("Minimum Amenity Frequency", 5, 100, 10, 5,
                                        help="Minimum number of listings an amenity must appear in to be included.")
 
    # --- Analysis Trigger --- 
    if st.button("Run Association Rule Mining"):
        st.info("Running analysis... This might take a moment.")
        progress_bar = st.progress(0)
        status_text = st.empty()
 
        try:
            status_text.text("Initializing Miner...")
            miner = AmenityAssociationMiner(listings_df)
            progress_bar.progress(10)
 
            status_text.text("Preprocessing data (parsing amenities, discretizing target, one-hot encoding)...")
            start_time = time.time()
            transaction_df = miner.preprocess_data(
                target_col=target_col,
                n_bins=n_bins,
                min_amenity_freq=min_amenity_freq
            )
            preprocess_time = time.time() - start_time
            status_text.text(f"Preprocessing complete ({preprocess_time:.2f}s). Found {transaction_df.shape[1]} items.")
            progress_bar.progress(50)
 
            if transaction_df is None or transaction_df.empty:
                st.warning("Preprocessing failed or resulted in no data. Check logs or adjust parameters (e.g., min amenity frequency).")
                st.stop()
 
            status_text.text(f"Finding frequent itemsets (min_support={min_support})...")
            start_time = time.time()
            # Adjust metric/threshold based on selection for find_rules
            min_conf = min_threshold if metric == 'confidence' else 0.1 # Use default confidence if metric is lift
            min_lift = min_threshold if metric == 'lift' else 1.0 # Use default lift if metric is confidence
            rules = miner.find_rules(
                transaction_df,
                min_support=min_support,
                min_confidence=min_conf, # Pass appropriate confidence threshold
                metric=metric,
                min_lift=min_lift # Pass appropriate lift threshold
            )
            mining_time = time.time() - start_time
            status_text.text(f"Rule mining complete ({mining_time:.2f}s).")
            progress_bar.progress(90)
 
            if rules is None:
                st.error("An error occurred during rule mining. Check logs.")
                st.stop()
            elif rules.empty:
                st.warning(f"No association rules found meeting the specified criteria (min_support={min_support}, min_{metric}={min_threshold}). Try relaxing the thresholds.")
            else:
                st.success(f"Found {len(rules)} association rules.")
 
                # --- Display Results ---
                st.subheader("Discovered Association Rules")
 
                # Filter for rules where consequent is a target category bin
                target_bin_prefix = f"{target_col}_Bin"
                rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                target_rules = rules[rules['consequents_str'].str.startswith(target_bin_prefix)]
 
                if target_rules.empty:
                    st.info("No rules found with the target variable as the consequent. Displaying all rules.")
                    display_rules = rules
                else:
                    st.info(f"Showing {len(target_rules)} rules with '{target_col}' bins as the consequent.")
                    display_rules = target_rules
 
                # Format frozensets for display
                display_rules['antecedents_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                # display_rules['consequents_str'] is already created
 
                st.dataframe(display_rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].round(4))
 
                st.subheader("Rules Visualization")
                status_text.text("Generating plot...")
                rules_plot = plot_association_rules(display_rules, metric=metric)
                if rules_plot:
                    st.pyplot(rules_plot)
                else:
                    st.warning("Could not generate the rules visualization plot.")
 
            progress_bar.progress(100)
            status_text.text("Analysis complete.")
 
        except Exception as e:
            logger.error(f"Error during Association Rule Mining analysis: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")
            status_text.text("Analysis failed.")
            if progress_bar: progress_bar.progress(100)
        finally:
            # Ensure spinner/progress stops
            pass
 
 
def show_custom_analysis_interface():
    """Interface for users to request custom analyses."""
    st.subheader("Custom Analysis Request")
    
    st.markdown("""
        Need a specific data mining analysis for your decision-making? 
        Submit your request here, and our data science team will evaluate it.
    """)
    
    analysis_type = st.selectbox(
        "Analysis Type",
        options=[
            "Select an analysis type...",
            "Clustering (K-means, DBSCAN, etc.)",
            "Regression/Prediction",
            "Classification",
            "Time Series Analysis",
            "Association Rules Mining",
            "Text Mining/NLP",
            "Other (please specify)"
        ]
    )
    
    business_goal = st.text_area(
        "Business Goal",
        placeholder="What decision or insight are you trying to gain from this analysis?"
    )
    
    data_needed = st.multiselect(
        "Data Sources Needed",
        options=[
            "Airbnb listings", 
            "Property sales", 
            "Census demographics",
            "Rental rates",
            "Crime statistics",
            "Economic indicators",
            "Geographic/spatial data",
            "Other (please specify)"
        ]
    )
    
    additional_notes = st.text_area(
        "Additional Notes",
        placeholder="Any other details or requirements for your analysis?"
    )
    
    if st.button("Submit Analysis Request"):
        if analysis_type == "Select an analysis type..." or not business_goal:
            st.error("Please select an analysis type and describe your business goal.")
        else:
            st.success("Your analysis request has been submitted! We'll evaluate it for inclusion in our roadmap.")
            
            # In a real application, this would be logged or emailed to the development team
            # Here, we're just displaying a confirmation message
            with st.expander("Request Details", expanded=True):
                st.write(f"**Analysis Type:** {analysis_type}")
                st.write(f"**Business Goal:** {business_goal}")
                st.write(f"**Data Sources Needed:** {', '.join(data_needed)}")
                st.write(f"**Additional Notes:** {additional_notes}")

@ensure_data_loaded(datasets=['neighborhood_stats'])
def show_neighborhood_segmentation(data_loader, agent=None):
    st.subheader("Neighborhood Segmentation using K-Means Clustering")
    st.markdown("Group similar neighborhoods based on statistical features.")
    logger.info("Executing Neighborhood Segmentation.") # Added
    
    try:
        # Use a spinner to indicate loading
        with ProgressSpinner("Loading neighborhood data..."):
            stats_df = data_loader.get_data('neighborhood_stats')
            if stats_df is None or stats_df.empty:
                st.error("Neighborhood stats data is not available. Please check data processing steps.")
                return

            logger.info(f"Columns in loaded stats_df: {stats_df.columns.tolist()}")
            logger.info(f"Data types of loaded stats_df:\n{stats_df.dtypes}")

            logger.debug(f"Neighborhood stats data shape: {stats_df.shape}") # Added

            # --- Dynamic Feature Selection based on available data ---
            available_columns = stats_df.columns.tolist()
            logger.debug(f"Available columns for clustering: {available_columns}")

            # Define potential features using names found in fallback data
            potential_feature_options = [
                'listing_count', 'mean_price', 'median_price', 'price_std',
                'mean_bedrooms', 'median_bedrooms', 'entire_home_percent', 
                'mean_guests', 'median_guests', 'home_type_score', 
                'B19013_001E', # Median Household Income
                'B01003_001E', # Total Population
                'B25064_001E', # Median Gross Rent
                'B25077_001E', # Median Home Value
                'B25002_003E'  # Vacant Housing Units
            ]
            # Select a sensible subset of these as ideal defaults
            ideal_default_features = [
                'listing_count', 'mean_price', 'mean_bedrooms', 'entire_home_percent', 'B19013_001E'
            ]
            
            # Filter options and defaults based on actual columns present and numeric type
            available_options = [f for f in potential_feature_options if f in available_columns and pd.api.types.is_numeric_dtype(stats_df[f])]
            actual_default_features = [f for f in ideal_default_features if f in available_options] # Defaults must be from available numeric options

            # Warn user if ideal defaults are missing (less likely now, but good practice)
            missing_defaults = set(ideal_default_features) - set(actual_default_features)
            if missing_defaults:
                st.sidebar.warning(f"Note: Default clustering features missing/non-numeric in loaded data: {', '.join(missing_defaults)}. Using available features.")
                logger.warning(f"Missing/non-numeric default clustering features: {missing_defaults}")

            # Check if any features are available at all
            if not available_options:
                st.error("No suitable numeric features found in the loaded data for clustering.")
                logger.error("No available numeric features for clustering.")
                return

            # Feature selection widget using dynamic options/defaults
            features = st.sidebar.multiselect(
                "Features for Clustering", 
                options=available_options, 
                default=actual_default_features
            )
            logger.debug(f"Selected features for clustering: {features}") # Added

            # Ensure at least one feature is selected
            if not features:
                st.warning("Please select at least one feature for clustering.")
                return

            # --- Impute Missing Values (instead of dropping) ---
            # Keep only the selected features for imputation
            data_to_impute = stats_df[features]
            
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            imputed_data_array = imputer.fit_transform(data_to_impute)
            
            # Create a new DataFrame with imputed values and original index/columns
            logger.debug(f"Shape of imputed_data_array: {imputed_data_array.shape}")
            logger.debug(f"Length of features list: {len(features)}")
            logger.debug(f"Features list content: {features}")
            stats_df_processed = pd.DataFrame(imputed_data_array, columns=features, index=stats_df.index)
            logger.debug(f"Shape after imputing NaNs based on {features}: {stats_df_processed.shape}")
            logger.info(f"Imputed {imputer.statistics_.size} features using median strategy.")

            # Check if data is still empty after imputation (shouldn't happen with imputation unless input was empty)
            if stats_df_processed.empty:
                st.error("Data is unexpectedly empty after attempting imputation.")
                logger.error("DataFrame empty after imputation step.") 
                return
                
            # --- Data Scaling (use imputed data) ---
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(stats_df_processed) # Use the processed (imputed) data
            logger.debug(f"Scaled data shape: {scaled_data.shape}") # Added

            # --- K-Means Clustering (Using Cached Function on scaled, imputed data) ---
            n_clusters = st.slider("Select number of clusters (K)", 2, 10, 4, key='kmeans_k')
            logger.debug(f"Selected K={n_clusters} for K-Means.")
            
            # Call the cached clustering function
            cluster_labels, error_msg = perform_kmeans_clustering(scaled_data, n_clusters)

            if error_msg:
                st.error(f"Clustering Error: {error_msg}")
                logger.error(f"perform_kmeans_clustering returned error: {error_msg}")
                return
                
            if cluster_labels is None:
                st.error("Failed to perform clustering. Check logs.")
                logger.error("perform_kmeans_clustering returned None labels without specific error message.") # Should ideally not happen if error handling is correct
                return

            stats_df_processed['Cluster'] = cluster_labels
            logger.info(f"Cluster labels (shape: {cluster_labels.shape}) added to DataFrame.")

            # --- Visualization ---
            st.write("### Clustering Results")
            st.dataframe(stats_df_processed[['neighborhood', 'Cluster']].sort_values('Cluster'))
            
            # Plot clusters using the first two features
            if len(features) >= 2:
                cluster_plot = plot_clusters(stats_df_processed, features[0], features[1], 'Cluster')
                if cluster_plot:
                    # Use st.plotly_chart instead of st.pyplot
                    st.plotly_chart(cluster_plot, use_container_width=True)
                else:
                    st.warning("Could not generate cluster plot.")
                    logger.warning(f"plot_clusters returned None for features: {features[0]}, {features[1]}")
            else:
                st.warning("Please select at least two features to visualize clusters.")
                logger.warning("Cannot plot clusters: Less than two features selected.")
        
    except FileNotFoundError:
        st.error("Neighborhood stats file not found. Please ensure 'airbnb_census_by_zipcode.csv' exists in 'data/processed/combined'.")
        logger.error("Neighborhood stats file not found during segmentation.") # Added
    except KeyError as e:
        st.error(f"Missing expected column for clustering: {e}. Please check the dataset.")
        logger.error(f"KeyError during segmentation: {e}", exc_info=True) # Added
    except Exception as e:
        st.error(f"An unexpected error occurred during neighborhood segmentation: {e}")
        logger.error(f"Unexpected error in show_neighborhood_segmentation: {e}", exc_info=True) # Added

# --- Caching Helper Function for K-Means Clustering ---
@st.cache_data(ttl=3600)
def perform_kmeans_clustering(scaled_data, n_clusters):
    """Performs K-Means clustering and returns cluster labels."""
    logger.info("Executing perform_kmeans_clustering (cached). K=%d", n_clusters)
    try:
        if len(scaled_data) < n_clusters:
            logger.warning(f"Insufficient data points ({len(scaled_data)}) for K={n_clusters}.")
            return None, f"Cannot form {n_clusters} clusters with only {len(scaled_data)} data points."
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        logger.info(f"K-Means clustering completed successfully with K={n_clusters}.")
        return cluster_labels, None # Success
    except ValueError as ve:
        logger.error(f"ValueError during K-Means fit (K={n_clusters}, data points={len(scaled_data)}): {ve}", exc_info=True)
        return None, f"Error during K-Means fitting: {ve}. Check data."
    except Exception as e:
        logger.error(f"Error in perform_kmeans_clustering: {e}", exc_info=True)
        return None, f"An unexpected error occurred during clustering: {e}"

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler # Added back for potential future use
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split # Added

# Assuming visualization functions are in a separate module
# from ..visualization.plots import plot_clusters, plot_feature_importance # Example
# Placeholder plot functions if not externalized
def plot_clusters(df, feature1, feature2, cluster_col):
    """Generates an interactive scatter plot of clusters using Plotly WebGL."""
    try:
        logger.debug(f"Plotting clusters with Plotly GL. Features: {feature1}, {feature2}. Cluster col: {cluster_col}")
        fig = go.Figure(data=go.Scattergl(
            x=df[feature1],
            y=df[feature2],
            mode='markers',
            marker=dict(
                color=df[cluster_col], # Color points by cluster label
                colorscale='Viridis', # Color scale for clusters
                showscale=True,
                colorbar=dict(title='Cluster')
            ),
            text=df.get('neighborhood', df.index), # Tooltip text (optional)
            hovertemplate=
            f'<b>Neighborhood:</b> %{{text}}<br>' +
            f'<b>{feature1}:</b> %{{x}}<br>' +
            f'<b>{feature2}:</b> %{{y}}<br>' +
            f'<b>Cluster:</b> %{{marker.color}}<extra></extra>'
        ))

        fig.update_layout(
            title='Neighborhood Clusters based on Features',
            xaxis_title=feature1,
            yaxis_title=feature2,
            hovermode='closest',
            autosize=True,
            # Optional: Adjust margins if needed
            # margin=dict(l=40, r=40, t=40, b=40)
        )
        logger.info("Plotly GL cluster plot created successfully.")
        return fig
    except KeyError as e:
        logger.error(f"KeyError plotting clusters: Missing column {e}. Available: {df.columns.tolist()}", exc_info=True)
        st.warning(f"Could not generate cluster plot: Column '{e}' not found.")
        return None
    except Exception as e:
        logger.error(f"Error plotting clusters with Plotly: {e}", exc_info=True)
        st.warning(f"An error occurred while generating the cluster plot: {e}")
        return None

def plot_feature_importance(importance_df):
    """Generates an interactive bar chart of feature importance using Plotly."""
    try:
        logger.debug("Plotting feature importance with Plotly.")
        # Ensure importance_df is sorted for plotting
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        fig = go.Figure(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h', # Horizontal bar chart
            hovertemplate=
            '<b>Feature:</b> %{y}<br>' +
            '<b>Importance:</b> %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title='Feature Importance for Price Prediction',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            yaxis=dict(tickmode='linear'), # Ensure all feature names are shown
            height=max(400, len(importance_df) * 30), # Dynamic height based on number of features
            margin=dict(l=100, r=20, t=50, b=40) # Adjust left margin for feature names
        )
        logger.info("Plotly feature importance plot created successfully.")
        return fig
    except Exception as e:
        logger.error(f"Error plotting feature importance with Plotly: {e}", exc_info=True)
        st.warning(f"An error occurred while generating the feature importance plot: {e}")
        return None

# --- Caching Helper Function for Price Prediction ---
@st.cache_data(ttl=3600) # Cache results for 1 hour
def train_evaluate_price_model(df_processed, features):
    """Trains RF model, evaluates, and gets feature importance."""
    logger.info("Executing train_evaluate_price_model (cached). Features: %s", features)
    try:
        X = df_processed[features]
        y = df_processed['price']
        logger.debug(f"Data shapes - X: {X.shape}, y: {y.shape}")

        if X.empty or y.empty or len(X) != len(y):
            logger.error("Invalid data shapes for training.")
            return None, None, None, None # Model, Metrics, Importance DF, Error

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.debug(f"Split shapes - Train: {X_train.shape}, Test: {X_test.shape}")

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        logger.info("Model training complete inside cached function.")

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {'MAE': mae, 'MSE': mse, 'R2': r2}
        logger.info(f"Evaluation complete: {metrics}")

        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})
        importance_df = importance_df.sort_values('importance', ascending=False)
        logger.debug("Feature importance calculated.")

        return model, metrics, importance_df, None # Success
    except Exception as e:
        logger.error(f"Error in train_evaluate_price_model: {e}", exc_info=True)
        return None, None, None, str(e) # Return error message

# --- Analysis Functions ---
@ensure_data_loaded(datasets=['cleaned_airbnb'])
def show_price_prediction(data_loader, agent=None):
    st.subheader("Short-Term Rental Price Prediction")
    st.markdown("Predict rental prices based on property features using a machine learning model.")
    logger.info("Executing Price Prediction analysis.")

    try:
        # --- Data Preparation ---
        with ProgressSpinner("Loading and preparing data for prediction..."):
            df = data_loader.get_data('cleaned_airbnb')
            if df is None or df.empty:
                st.error("Cleaned Airbnb data is not available. Please check data loading.")
                logger.error("Cleaned Airbnb data is missing for price prediction.") # Added
                return
            
            logger.debug(f"Cleaned Airbnb data shape: {df.shape}") # Added

            # --- Feature Selection ---
            default_features = ['bedrooms', 'bathrooms', 'accommodates', 'latitude', 'longitude', 'review_scores_rating']
            # Ensure only available columns are defaults
            available_columns = [col for col in default_features if col in df.columns]
            features = st.sidebar.multiselect("Features for Prediction", df.columns.tolist(), default=available_columns)
            logger.debug(f"Selected features for prediction: {features}")
            
            if not features:
                st.warning("Please select at least one feature for prediction.")
                logger.warning("No features selected for price prediction.")
                return

            # Ensure 'price' column exists
            if 'price' not in df.columns:
                st.error("The 'price' column is missing from the dataset.")
                logger.error("Missing 'price' column in cleaned_airbnb data.")
                return
                
            df_processed = df[features + ['price']].copy()
            df_processed.dropna(inplace=True)
            logger.debug(f"Data shape after selecting features and dropping NaNs: {df_processed.shape}")
            
            if df_processed.empty or len(df_processed) < 10: # Check for sufficient data
                st.warning("Insufficient data available for training/prediction after cleaning.")
                logger.warning(f"Insufficient data ({len(df_processed)} rows) for price prediction after cleaning.") # Added
                return

            # --- Model Training & Evaluation (Using Cached Function) ---
            st.write("### Model Training & Evaluation (Random Forest Regressor)")
            # Call the cached function
            model, metrics, importance_df, error_msg = train_evaluate_price_model(df_processed, features)
            
            if error_msg:
                st.error(f"Error during model training/evaluation: {error_msg}")
                logger.error(f"Cached function train_evaluate_price_model returned error: {error_msg}")
                return
                
            if model is None or metrics is None or importance_df is None:
                st.error("Failed to train or evaluate the model. Check logs.")
                logger.error("Cached function train_evaluate_price_model returned None values.")
                return

            # --- Display Results ---
            st.subheader("Model Evaluation")
            st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['MAE']:.2f}")
            st.metric(label="Mean Squared Error (MSE)", value=f"{metrics['MSE']:.2f}")
            st.metric(label="R-squared (R2)", value=f"{metrics['R2']:.2f}")
            logger.info(f"Displayed Model Evaluation - MAE: {metrics['MAE']:.2f}, MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.2f}")

            # --- Feature Importance ---
            st.write("### Feature Importance")
            logger.debug(f"Feature importances: {importance_df}")
            
            fig = plot_feature_importance(importance_df)
            if fig:
                # Use st.plotly_chart instead of st.pyplot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate feature importance plot.")
                logger.warning("Failed to generate feature importance plot.")

    except FileNotFoundError:
        st.error("Cleaned Airbnb data file not found. Please ensure 'cleaned_airbnb_listings.csv' exists in 'data/processed/airbnb'.")
        logger.error("Cleaned Airbnb data file not found for price prediction.") # Added
    except KeyError as e:
        st.error(f"Missing expected column for prediction: {e}. Please check the dataset.")
        logger.error(f"KeyError during price prediction: {e}", exc_info=True) # Added
    except ValueError as ve:
        st.error(f"Data error during prediction: {ve}. Check feature selection and data cleaning.")
        logger.error(f"ValueError during price prediction: {ve}", exc_info=True) # Added
    except Exception as e:
        st.error(f"An unexpected error occurred during price prediction: {e}")
        logger.error(f"Unexpected error in show_price_prediction: {e}", exc_info=True) # Added

@ensure_data_loaded(datasets=['cleaned_airbnb', 'investment_potential'])
def show_investment_simulation(data_loader, agent=None):
    st.subheader("Investment Potential Simulation")
    st.markdown("Simulate potential returns based on investment amount and property characteristics.")
    logger.info("Executing Investment Potential Simulation.") # Added

    try:
        # --- Configuration ---
        st.sidebar.subheader("Simulation Parameters")
        investment_amount = st.sidebar.number_input("Investment Amount", min_value=10000, value=50000, step=1000, help="The amount to invest.")
        logger.debug(f"Simulation parameters - Investment: {investment_amount}") # Added

        risk_tolerance = st.sidebar.select_slider("Risk Tolerance", options=['Low', 'Medium', 'High'], value='Medium')
        logger.debug(f"Simulation parameters - Risk Tolerance: {risk_tolerance}") # Added

        # --- Data Preparation ---
        with ProgressSpinner("Loading data for simulation..."):
            listings_df = data_loader.get_data('cleaned_airbnb')
            if listings_df is None or listings_df.empty:
                st.error("Cleaned Airbnb data is not available.")
                logger.error("Cleaned Airbnb data is missing for investment simulation.") # Added
                return
            
            investment_df = data_loader.get_data('investment_potential')
            if investment_df is None or investment_df.empty:
                st.error("Investment potential data is not available.")
                logger.error("Investment potential data is missing for simulation.") # Added
                return
            
            logger.debug(f"Listings data shape: {listings_df.shape}, Investment data shape: {investment_df.shape}") # Added
            
            # Merge or align data if necessary (assuming investment_df aligns with listings or has common keys)
            # For this example, let's assume investment_df contains pre-calculated potential scores indexed appropriately
            # Example: Merge on a common identifier like 'id' or 'zipcode'
            # combined_df = pd.merge(listings_df, investment_df, on='common_key', how='inner')
            # Placeholder: Using investment_df directly if it contains all needed info
            simulation_data = investment_df # Adapt as per actual data structure
            logger.debug(f"Data prepared for simulation, shape: {simulation_data.shape}") # Added

        # --- Simulation Logic ---
        st.write("### Simulation Results")
        logger.info("Starting investment simulation calculation.") # Added
        # Placeholder simulation logic: Filter properties based on potential and simulate returns
        # This logic needs to be defined based on the project's specific goals
        # Example: Select top N properties based on 'investment_score' and calculate estimated ROI
        
        # Example: Filter by investment score (assuming higher is better)
        potential_threshold = 0.7 # Example threshold
        high_potential_properties = simulation_data[simulation_data['investment_score'] > potential_threshold]
        logger.debug(f"Found {len(high_potential_properties)} high potential properties.") # Added
        
        if high_potential_properties.empty:
            st.warning("No properties found matching the high potential criteria for simulation.")
            logger.warning("No high potential properties found for simulation.") # Added
            return
            
        # Simulate returns (Simplified example)
        # Assuming 'estimated_roi' column exists in investment_df
        simulated_results = high_potential_properties[['neighborhood', 'zipcode', 'estimated_roi', 'investment_score']].copy()
        simulated_results['estimated_return'] = investment_amount * simulated_results['estimated_roi']
        simulated_results = simulated_results.sort_values('estimated_return', ascending=False).head(10)
        logger.info(f"Investment simulation completed. Displaying top {len(simulated_results)} results.") # Added

        # --- Display Results ---
        if not simulated_results.empty:
            st.write("### Top Investment Opportunities")
            st.dataframe(simulated_results)
        else:
            st.warning("Simulation did not produce valid results.")
            logger.warning("Investment simulation returned empty results.") # Added
            
    except FileNotFoundError as fnf_error:
        st.error(f"Required data file not found: {fnf_error}. Please check 'data/processed'.")
        logger.error(f"FileNotFoundError during investment simulation: {fnf_error}") # Added
    except KeyError as e:
        st.error(f"Missing expected column for simulation: {e}. Check 'investment_potential' data.")
        logger.error(f"KeyError during investment simulation: {e}", exc_info=True) # Added
    except Exception as e:
        st.error(f"An unexpected error occurred during the investment simulation: {e}")
        logger.error(f"Unexpected error in show_investment_simulation: {e}", exc_info=True) # Added


@ensure_data_loaded(datasets=['zillow_processed'])
def show_time_series_decomposition(data_loader):
    st.subheader("Time Series Decomposition for Zillow Observed Rent Index (ZORI)")
    st.markdown("Analyze the trend, seasonal, and residual components of ZORI data over time.")

    zori_df = data_loader.get_data('zillow_processed')
    if zori_df is None or zori_df.empty:
        st.error("Zillow data is not available. Please run the data processing pipeline.")
        return

    # --- Data Preparation ---
    selected_region = st.selectbox("Select Region", zori_df['RegionName'].unique())
    region_data = zori_df[zori_df['RegionName'] == selected_region].sort_index()
    if region_data.empty:
        st.warning(f"No ZORI data found for the selected region: {selected_region}")
        return

    # Prepare the series for decomposition
    target_column = 'ZORI_NSA'
    if target_column not in region_data.columns:
        st.error(f"Target column '{target_column}' not found in the data for {selected_region}.")
        return

    time_series = region_data[target_column].dropna() # Drop NA values
    if time_series.empty:
        st.warning(f"No valid time series data for '{target_column}' in {selected_region} after dropping NA.")
        return

    # Initialize the decomposer
    decomposer = TimeSeriesDecomposer(time_series)

    # --- Analysis Configuration ---
    st.markdown("#### Decomposition Configuration")
    col1, col2 = st.columns(2)
    with col1:
        decomposition_model = st.selectbox("Decomposition Model", ('additive', 'multiplicative'), index=0, help="'additive' assumes seasonal effects are constant, 'multiplicative' assumes they scale with the trend.")
    with col2:
        seasonal_period = st.number_input("Seasonality Period", min_value=2, value=12, step=1, help="The number of observations per seasonal cycle (e.g., 12 for monthly data with annual seasonality).")

    # --- Run Analysis ---
    if st.button("Decompose Time Series"):
        if len(time_series) < seasonal_period * 2:
            st.warning(f"Time series is too short for the selected period ({seasonal_period}). Needs at least {seasonal_period*2} data points.")
        else:
            with st.spinner("Performing decomposition..."):
                try:
                    decomposition_result = decomposer.decompose(model=decomposition_model, period=seasonal_period)

                    if decomposition_result:
                        st.success("Decomposition successful!")
                        decomp_plot = plot_time_series_decomposition(decomposition_result)
                        if decomp_plot:
                            st.plotly_chart(decomp_plot, use_container_width=True)
                        else:
                            st.warning("Could not generate decomposition plot.")
                    else:
                        st.warning("Decomposition could not be performed. The time series might be too short, have too many missing values, or other issues. Check logs.")
                except Exception as e:
                    st.error("An error occurred during decomposition.")
                    logger.error(f"Time Series Decomposition failed: {e}\n{traceback.format_exc()}")
