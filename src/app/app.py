#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Miami Housing Impact Hub - Main Application

This is the main entry point for the Streamlit application that visualizes
the impact of short-term rentals on housing affordability in Miami-Dade County.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import gc
import psutil
from functools import wraps
from datetime import datetime
import logging

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

# Add the project root to the Python path so we can import modules
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Import application modules
from src.app.utils.data_loader import DataLoader
from src.app.utils.agent import HousingImpactAgent
from src.app.utils.summary_logger import SummaryLogger
from src.app.pages.home import show_home_page
# Import app utils for optimization
from src.app.utils.streamlit_optimizations import st_cached_data, st_cached_resource, lazy_load, optimize_dataframe, ProgressSpinner

# Import page modules
from src.app.pages.dashboard import show_dashboard_page
from src.app.pages.prediction import show_prediction_page
from src.app.pages.about import show_about_page
from src.app.pages.assistant import show_assistant_page
from src.app.pages.advanced_analysis import show_advanced_analysis_page
from src.app.pages.performance_metrics import show_performance_metrics_page
from src.app.pages.all_areas_visualization import show_all_areas_visualization

def main():
    """Main entry point for the Streamlit application."""
    logger.info("Starting Streamlit application")
    # Configure the Streamlit page with performance optimizations
    st.set_page_config(
        page_title="Miami Housing Impact Hub",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# Miami Housing Impact Hub\nAnalyzing the impact of short-term rentals on housing affordability in Miami-Dade County."
        }
    )
    
    # Performance optimization: Reduce automatic reruns and global configuration
    # This improves reactive performance
    # Clear query params if the 'clear_params' flag is set (e.g., navigating away)
    if st.session_state.get('clear_params', False):
        try:
            st.query_params.clear()
        except Exception as e:
            print(f"Could not clear query params: {e}") # Handle potential errors
        st.session_state['clear_params'] = False # Reset flag
    
    # Record page load start time for performance tracking
    current_time = time.time()
    if 'page_load_start_time' not in st.session_state:
        st.session_state['page_load_start_time'] = current_time
        
    # Initialize performance metrics tracking
    if 'performance_metrics' not in st.session_state:
        st.session_state['performance_metrics'] = {}
        
    # Memory optimization - run garbage collection periodically
    # This helps prevent memory leaks in long-running sessions
    if 'last_gc_time' not in st.session_state:
        st.session_state['last_gc_time'] = current_time
    elif current_time - st.session_state['last_gc_time'] > 300:  # Run GC every 5 minutes
        gc.collect()
        st.session_state['last_gc_time'] = current_time
    
    # Use st.cache_resource to initialize application components
    # This prevents re-initialization on each rerun
    @st.cache_resource(show_spinner=False)
    def get_data_loader(root_path):
        return DataLoader(root_path)
    
    @st.cache_resource(show_spinner=False)
    def get_summary_logger(root_path):
        return SummaryLogger(root_path)
    
    @st.cache_resource(show_spinner=False)
    def get_agent(_data_loader, _summary_logger):
        # Underscore prefix tells Streamlit not to hash these parameters
        return HousingImpactAgent(_data_loader, _summary_logger)
    
    # Initialize application state in session state if not already present
    if 'data_loader' not in st.session_state:
        with st.spinner("Loading data resources..."):
            st.session_state['data_loader'] = get_data_loader(project_root)
        
    # Initialize the summary logger
    if 'summary_logger' not in st.session_state:
        st.session_state['summary_logger'] = get_summary_logger(project_root)
        
    # Initialize the agent with data loader and summary logger
    if 'agent' not in st.session_state:
        st.session_state['agent'] = get_agent(
            st.session_state['data_loader'],
            st.session_state['summary_logger']
        )
        
    # Initialize user interaction tracking
    if 'page_views' not in st.session_state:
        st.session_state['page_views'] = {}

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Display sidebar navigation and logo
    with st.sidebar:
        # Check if image exists, otherwise display a title instead
        try:
            from pathlib import Path
            image_path = "src/app/assets/miami_skyline.png"
            if Path(image_path).exists():
                st.image(image_path, width=250)
            else:
                st.header("🏙️ Miami Housing Impact Hub")
        except Exception:
            st.header("🏙️ Miami Housing Impact Hub")
            
        st.caption("Analyzing the impact of short-term rentals on housing affordability in Miami-Dade County.") 
        
        # Create navigation menu with performance dashboard option
        pages = {
            "Home": show_home_page,
            "All 71 Areas Analysis": show_all_areas_visualization,  
            "Interactive Dashboard": show_dashboard_page,
            "Housing Predictions": show_prediction_page,
            "Advanced Analysis": show_advanced_analysis_page,
            "Housing Assistant": show_assistant_page,
            "About": show_about_page,
            "Performance Metrics": show_performance_metrics_page
        }
        
        page = st.selectbox("Navigate", list(pages.keys()))

    # Application title and description
    st.title("Miami Housing Impact Hub")
    st.markdown("""
        ### Analyzing the Impact of Short-Term Rentals on Housing Affordability in Miami-Dade County
        
        This application provides interactive visualizations and predictive analytics to understand 
        how short-term rentals affect housing affordability across Miami-Dade neighborhoods.
    """)

    # Track page views for the agent
    if page not in st.session_state['page_views']:
        st.session_state['page_views'][page] = 0
    st.session_state['page_views'][page] += 1

    # Log page view and update agent with page interaction
    st.session_state['summary_logger'].log_interaction(page, 'page_view')
    st.session_state['agent'].update_from_user_interaction({'page': page})

    # Performance improvement: Measure page loading times
    start_time = time.time()
    
    # Store current page in session state for performance tracking
    st.session_state['current_page'] = page
    
    # Show selected page with performance tracking
    try:
        # Handle pages differently based on their requirements
        if page == "Home":
            show_home_page(st.session_state['agent'])
        elif page == "All 71 Areas Analysis":
            show_all_areas_visualization()
        elif page == "Interactive Dashboard":
            show_dashboard_page(st.session_state['data_loader'], st.session_state['agent'])
        elif page == "Housing Predictions":
            show_prediction_page(st.session_state['data_loader'], st.session_state['agent'])
        elif page == "Advanced Analysis":
            show_advanced_analysis_page(st.session_state['data_loader'], st.session_state['agent'])
        elif page == "Housing Assistant":
            show_assistant_page(st.session_state['agent'])
        elif page == "Performance Metrics":
            show_performance_metrics_page(st.session_state['data_loader'], st.session_state['agent'])
        else:  # About page
            show_about_page()
    except Exception as e:
        st.error(f"Error loading {page} page: {str(e)}")
        logger.error(f"Error loading page: {str(e)}", exc_info=True)
        import traceback
        st.exception(e)
        
    # Log page performance metrics
    elapsed = time.time() - start_time
    st.session_state.setdefault('page_performance', {})
    st.session_state['page_performance'][page] = {
        'last_render_time': elapsed,
        'renders': st.session_state.get('page_performance', {}).get(page, {}).get('renders', 0) + 1
    }

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Sources")
    st.sidebar.markdown("""
    - Census Bureau American Community Survey
    - Miami-Dade County Housing Data
    - Airbnb Listings Data
    """)

    st.sidebar.markdown("### Project Information")
    st.sidebar.markdown("""
    Miami Housing Impact Hub v0.1.0  
    Data Mining Final Project  
    2025
    """)

    # Add option to generate summary statistics
    st.sidebar.markdown("---")
    if st.sidebar.button("Generate Summary Reports"):
        with st.sidebar.spinner("Generating summary statistics..."):
            st.sidebar.write("Processing application logs...")
            st.session_state['summary_logger'].generate_summary_stats()
            st.sidebar.write("Exporting agent insights...")
            export_file = st.session_state['summary_logger'].export_insights_to_csv()
            
            if export_file:
                st.sidebar.success(f"Key insights exported to {export_file.name}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main application loop: {e}", exc_info=True)
        st.error(f"An critical error occurred in the application: {e}")
