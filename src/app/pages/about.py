#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
About Project Page Module

This module contains the code for the application's about page,
which provides information about the project methodology, data sources,
and technical implementation.
"""

import streamlit as st
import pandas as pd
import numpy as np

def show_about_page():
    """
    Display the about page content.
    
    This page provides information about the project methodology,
    data sources, and technical implementation.
    """
    st.header("About the Project")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Project Overview", "Methodology", "Technical Details"])
    
    with tab1:
        show_project_overview()
    
    with tab2:
        show_methodology()
    
    with tab3:
        show_technical_details()

def show_project_overview():
    """Show the project overview section."""
    st.subheader("Miami-Dade Short-Term Rental Impact Analysis")
    
    st.markdown("""
        This project analyzes the impact of short-term rentals (particularly Airbnb listings)
        on housing affordability in Miami-Dade County. The research aims to understand how
        the concentration of short-term rentals in specific neighborhoods affects housing
        costs and availability for residents.
        
        ### Research Questions
        
        1. How does the density of short-term rentals correlate with housing affordability metrics?
        2. Which neighborhoods show the strongest relationship between Airbnb concentration and housing costs?
        3. How might different regulatory approaches impact housing affordability?
        4. What socioeconomic factors influence the relationship between short-term rentals and affordability?
        
        ### Key Findings
        
        Our analysis reveals several important patterns:
        
        * Short-term rentals are heavily concentrated in specific Miami-Dade ZIP codes
        * Neighborhoods with high Airbnb density show significantly higher rent-to-income ratios
        * 80-95% of Airbnb listings in high-impact areas are entire homes/apartments, not shared rooms
        * There is a strong positive correlation between Airbnb density and housing costs
        * The affordability gap between high and low Airbnb density areas is substantial
    """)
    
    # Project value proposition
    st.markdown("""
        ### Value to Stakeholders
        
        This analysis provides valuable insights for:
        
        * **Government Officials**: Data-driven guidance for short-term rental regulations
        * **Community Organizations**: Evidence for housing advocacy efforts
        * **Residents**: Understanding of neighborhood housing market dynamics
        * **Property Owners**: Market context for investment and rental decisions
        * **Researchers**: Quantitative data on short-term rental impacts in Miami-Dade
    """)

def show_methodology():
    """Show the methodology section."""
    st.subheader("Data Collection and Analysis Methodology")
    
    st.markdown("""
        This project follows a comprehensive data science methodology, integrating
        multiple data sources and analytical approaches to understand the relationship
        between short-term rentals and housing affordability.
        
        ### Data Sources
        
        * **Census Data**: American Community Survey (ACS) 5-Year Estimates for Miami-Dade County
        * **Airbnb Data**: Comprehensive dataset of Airbnb listings including location, price, and property type
        * **Housing Data**: Miami-Dade County housing market metrics including rent prices and vacancy rates
        
        ### Analysis Approach
        
        Our analysis followed these key steps:
        
        1. **Data Collection and Cleaning**:
           * Processed raw data from multiple sources
           * Cleaned and standardized variable formats
           * Geocoded listings to census tracts and ZIP codes
        
        2. **Exploratory Data Analysis**:
           * Visualized distribution of Airbnb listings across neighborhoods
           * Analyzed patterns in housing affordability metrics
           * Identified correlations between key variables
        
        3. **Statistical Analysis**:
           * Calculated correlation coefficients between Airbnb metrics and housing variables
           * Measured statistical significance of observed relationships
           * Controlled for demographic and economic factors
        
        4. **Geospatial Analysis**:
           * Mapped short-term rental density against affordability metrics
           * Identified neighborhood clusters with similar patterns
           * Visualized geographic distribution of impacts
        
        5. **Predictive Modeling**:
           * Developed regression models to quantify relationships
           * Created scenario testing framework for policy evaluation
           * Validated models against historical data
    """)
    
    # Limitations section
    st.markdown("""
        ### Limitations and Considerations
        
        While our analysis provides valuable insights, several limitations should be noted:
        
        * **Correlation vs. Causation**: Our analysis identifies correlations between variables
          but cannot definitively establish causation.
        
        * **Data Timeframes**: Data sources have different collection periods, which may
          affect the precision of combined analyses.
        
        * **External Factors**: Our models cannot account for all external factors affecting
          the housing market, such as economic trends, development patterns, and zoning changes.
        
        * **Data Granularity**: Some analyses use ZIP code level data, which may mask
          variation within neighborhoods.
    """)

def show_technical_details():
    """Show the technical details section."""
    st.subheader("Technical Implementation")
    
    st.markdown("""
        This application was developed using a modular architecture following data science
        best practices. The codebase is structured to maximize maintainability and flexibility.
        
        ### Project Structure
        
        ```
        /miami-housing-impact-hub
          /src
            /data - Data processing modules
            /models - Prediction and forecasting models
            /visualization - Visualization components
            /api - API implementation modules
            /app - Streamlit application pages
          /data
            /raw - Original datasets
            /processed - Cleaned and transformed data
          /docs - Documentation
        ```
        
        ### Technology Stack
        
        * **Python**: Core programming language
        * **Pandas/NumPy**: Data processing and analysis
        * **Streamlit**: Web application framework
        * **Plotly/Matplotlib**: Data visualization
        * **scikit-learn**: Machine learning and modeling
        * **GeoPandas**: Geospatial data handling
        
        ### Data Processing Pipeline
        
        Our data processing pipeline follows these steps:
        
        1. **Raw Data Ingestion**: Import data from various sources
        2. **Data Cleaning**: Handle missing values, standardize formats
        3. **Feature Engineering**: Create derived metrics and indicators
        4. **Data Integration**: Combine datasets using common identifiers
        5. **Analysis Ready Data**: Generate final datasets for analysis
        
        ### Visualization and Interaction
        
        The application offers several interactive components:
        
        * **Interactive Maps**: Explore geographic patterns
        * **Dynamic Filtering**: Focus on specific neighborhoods or metrics
        * **Scenario Testing**: Model potential policy impacts
        * **Metric Comparisons**: Compare different housing variables
    """)
    
    # Future development
    st.markdown("""
        ### Future Development
        
        Planned enhancements to the application include:
        
        * **Time Series Analysis**: Track changes in relationships over time
        * **Additional Data Sources**: Integrate property sales and tourism data
        * **Advanced Modeling**: Implement machine learning for more sophisticated predictions
        * **Mobile Optimization**: Enhance accessibility on mobile devices
        * **API Expansion**: Provide programmatic access to insights and data
        
        ### Contact and Contribution
        
        This project was developed as part of a data mining final project. For more
        information or to contribute to the project, please contact the development team.
    """)
    
    # Credits
    st.markdown("""
        ### Credits
        
        This project was made possible by the availability of public datasets
        and open-source software libraries. Special thanks to the following:
        
        * US Census Bureau American Community Survey
        * Miami-Dade County Open Data Portal
        * Streamlit framework and community
        * Python data science ecosystem
    """)
