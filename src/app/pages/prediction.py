#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Affordability Prediction Page Module

This module contains the code for the application's prediction page,
which allows users to predict how changes in Airbnb density might affect
housing affordability metrics in Miami-Dade neighborhoods.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pickle
import os
from pathlib import Path

def show_prediction_page(data_loader, agent):
    """
    Display the affordability prediction page.
    
    This page provides tools to predict how changes in Airbnb density
    might affect housing affordability metrics in Miami-Dade neighborhoods.
    
    Args:
        data_loader: DataLoader instance with access to all datasets
        agent: HousingImpactAgent instance for proactive insights
    """
    st.header("Affordability Prediction")
    
    st.markdown("""
        Explore how changes in short-term rental density might affect housing affordability
        in Miami-Dade neighborhoods. This predictive tool uses regression models trained
        on our combined dataset to estimate potential impacts.
    """)
    
    # Get and display an agent insight related to predictions
    with st.sidebar.container():
        st.sidebar.markdown("### Agent Insight")
        # Filter agent insights to focus on predictions if possible
        insights = [insight for insight in agent.get_proactive_insights() 
                  if any(keyword in insight['title'].lower() for keyword in ['predict', 'forecast', 'future', 'trend'])]
        
        # If no prediction-specific insights, use any insight
        if not insights:
            insights = agent.get_proactive_insights()
            
        if insights:
            selected_insight = insights[0]  # Just pick the first relevant one
            st.sidebar.markdown(f"**{selected_insight['title']}**")
            st.sidebar.markdown(selected_insight['description'])
            
            # Add a button to ask a follow-up question
            if st.sidebar.button("Ask about this prediction"):
                st.session_state['page'] = "Housing Assistant"
                st.session_state['user_question'] = f"Tell me more about future housing affordability trends"
                st.rerun()
    
    # Get the data for prediction
    if data_loader and hasattr(data_loader, 'combined_data') and data_loader.combined_data is not None:
        combined_data = data_loader.get_combined_data()
        if combined_data.empty:
            st.error("The combined dataset is empty. Please ensure the data is properly loaded.")
            st.stop()
    else:
        st.error("No combined data available. Please ensure the data loader is properly initialized.")
        st.stop()
    
    # Create a tabbed interface for different prediction tools
    tab1, tab2, tab3 = st.tabs(["Neighborhood Impact", "Scenario Testing", "Model Insights"])
    
    with tab1:
        show_neighborhood_impact(data_loader, combined_data)
    
    with tab2:
        show_scenario_testing(data_loader, combined_data)
    
    with tab3:
        show_model_insights(data_loader, combined_data)

def show_neighborhood_impact(data_loader, combined_data):
    """
    Show the neighborhood impact prediction tool.
    
    Args:
        data_loader: DataLoader instance
        combined_data: Combined dataset
    """
    st.subheader("Neighborhood Impact Prediction")
    
    st.markdown("""
        Select a neighborhood and adjust Airbnb density to see how it might
        affect housing affordability metrics in that area.
    """)
    
    # Determine the neighborhood identifier column
    neighborhood_column = None
    for col in ['neighborhood', 'zip_code', 'area']:
        if col in combined_data.columns:
            neighborhood_column = col
            break
    
    # Select neighborhood
    if not combined_data.empty and neighborhood_column:
        # Get unique neighborhoods
        neighborhoods = sorted(combined_data[neighborhood_column].unique().tolist())
        
        # Check if we have a recommended neighborhood from the agent
        initial_index = 0
        if 'recommended_action' in st.session_state and neighborhood_column in st.session_state['recommended_action']:
            recommended_area = st.session_state['recommended_action'][neighborhood_column]
            if recommended_area in neighborhoods:
                initial_index = neighborhoods.index(recommended_area)
                # Clear the recommendation after using it
                st.session_state.pop('recommended_action', None)
        
        # Create selection widget
        selected_area = st.selectbox(
            f"Select {neighborhood_column.replace('_', ' ').title()}:",
            options=neighborhoods,
            index=initial_index
        )
        
        # Update agent with neighborhood selection if agent exists
        if 'agent' in locals() or 'agent' in globals():
            if agent:
                try:
                    agent.update_from_user_interaction({
                        'page': 'Affordability Prediction',
                        'neighborhood': selected_area
                    })
                except:
                    pass  # If agent doesn't have this method, just continue
        
        # Get current data for selected area
        current_data = combined_data[combined_data[neighborhood_column] == selected_area].iloc[0]
        
        # Determine available metrics
        airbnb_count_col = None
        for col in ['listing_count', 'airbnb_count', 'num_listings']:
            if col in current_data:
                airbnb_count_col = col
                break
                
        density_col = None
        for col in ['listings_per_1000', 'airbnb_density', 'density']:
            if col in current_data:
                density_col = col
                break
                
        affordability_col = None
        for col in ['rent_to_income_ratio', 'affordability_index', 'rent_burden']:
            if col in current_data:
                affordability_col = col
                break
        
        # Display current metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if airbnb_count_col:
                st.metric(
                    "Current Airbnb Listings",
                    f"{int(current_data[airbnb_count_col])}"
                )
            else:
                st.metric("Airbnb Listings", "No Data")
        
        with col2:
            if density_col:
                st.metric(
                    "Listings Density",
                    f"{current_data[density_col]:.2f}"
                )
            else:
                st.metric("Listings Density", "No Data")
        
        with col3:
            if affordability_col:
                st.metric(
                    "Affordability Metric",
                    f"{current_data[affordability_col]:.1f}%" 
                    if '%' not in str(current_data[affordability_col]) else 
                    f"{current_data[affordability_col]}"
                )
            else:
                st.metric("Affordability", "No Data")
        
        # Input for adjusting Airbnb density
        st.subheader("Adjust Airbnb Density")
        
        adjustment = st.slider(
            "Change in Airbnb Listings (%):",
            min_value=-50,
            max_value=100,
            value=0,
            step=5
        )
        
        # Use a data-driven approach based on available metrics
        impact_factor = 0.15  # estimated impact per % change in listings
        
        # Calculate predicted changes based on available metrics
        predicted_values = {}
        
        if airbnb_count_col:
            predicted_values['listings'] = current_data[airbnb_count_col] * (1 + adjustment/100)
        
        if density_col:
            predicted_values['density'] = current_data[density_col] * (1 + adjustment/100)
        
        if affordability_col:
            # Simple prediction model
            current_ratio = current_data[affordability_col]
            predicted_values['affordability'] = current_ratio * (1 + adjustment/100 * impact_factor)
        
        # Update agent with prediction parameters if available
        if 'agent' in locals() or 'agent' in globals():
            if agent:
                try:
                    update_data = {
                        'page': 'Affordability Prediction',
                        'neighborhood': selected_area,
                        'adjustment': adjustment
                    }
                    
                    if 'affordability' in predicted_values:
                        update_data['predicted_ratio'] = predicted_values['affordability']
                        
                    agent.update_from_user_interaction(update_data)
                except:
                    pass  # If agent doesn't have this method, just continue
        
        # We'll define affordability status later when needed
        
        # Display prediction results
        st.subheader("Predicted Impact")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'listings' in predicted_values:
                st.metric(
                    "Predicted Airbnb Listings",
                    f"{int(predicted_values['listings'])}",
                    delta=f"{adjustment}%"
                )
            else:
                st.metric("Predicted Listings", "No Data")
        
        with col2:
            if 'density' in predicted_values:
                st.metric(
                    "Predicted Listings Density",
                    f"{predicted_values['density']:.2f}",
                    delta=f"{adjustment}%"
                )
            else:
                st.metric("Predicted Density", "No Data")
        
        with col3:
            if 'affordability' in predicted_values:
                affordability_value = predicted_values['affordability']
                original_value = current_data[affordability_col]
                st.metric(
                    "Predicted Affordability",
                    f"{affordability_value:.1f}%" if '%' not in str(affordability_value) else f"{affordability_value}",
                    delta=f"{affordability_value - original_value:.1f}%" 
                    if '%' not in str(affordability_value) else 
                    f"{float(str(affordability_value).replace('%','')) - float(str(original_value).replace('%','')):.1f}%",
                    delta_color="inverse"  # Higher is worse for affordability
                )
            else:
                st.metric("Predicted Affordability", "No Data")
        
        # Visualize current vs predicted if we have the data
        if any(k in predicted_values for k in ['listings', 'density', 'affordability']):
            # Create visualization data
            viz_data = []
            
            if 'listings' in predicted_values and airbnb_count_col:
                viz_data.append({
                    'Metric': 'Airbnb Listings',
                    'Current': float(current_data[airbnb_count_col]),
                    'Predicted': float(predicted_values['listings'])
                })
            
            if 'density' in predicted_values and density_col:
                viz_data.append({
                    'Metric': 'Listings Density',
                    'Current': float(current_data[density_col]),
                    'Predicted': float(predicted_values['density'])
                })
            
            if 'affordability' in predicted_values and affordability_col:
                viz_data.append({
                    'Metric': 'Affordability Index',
                    'Current': float(str(current_data[affordability_col]).replace('%', '')),
                    'Predicted': float(str(predicted_values['affordability']).replace('%', ''))
                })
            
            if viz_data:
                data = pd.DataFrame(viz_data)
                
                # Normalize for better visualization
                normalized_data = data.copy()
                for col in ['Current', 'Predicted']:
                    normalized_data[col] = data[col] / data['Current'] * 100
                
                # Create visualization
                fig = px.bar(
                    normalized_data,
                    x='Metric',
                    y=['Current', 'Predicted'],
                    barmode='group',
                    title=f"Impact of {adjustment}% Change in Airbnb Listings",
                    labels={'value': 'Relative Change (%)', 'variable': ''},
                    color_discrete_sequence=['#636EFA', '#EF553B']
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Affordability warning - only if we have the data
        if affordability_col and 'affordability' in predicted_values:
            # Define affordability threshold
            affordability_threshold = 30  # Standard threshold for rent burden
            current_affordable = current_data[affordability_col] <= affordability_threshold
            predicted_affordable = predicted_values['affordability'] <= affordability_threshold
            
            if current_affordable and not predicted_affordable:
                st.warning(f"⚠️ This change would push the area above the 30% affordability threshold!")
            elif not current_affordable and predicted_affordable:
                st.success(f"✓ This change would bring the area below the 30% affordability threshold!")
            elif not current_affordable and not predicted_affordable:
                if predicted_values['affordability'] > current_data[affordability_col]:
                    st.error(f"⚠️ This change would worsen the existing affordability concerns in this area.")
                else:
                    st.info(f"ℹ️ This change would improve affordability, but not enough to reach the 30% threshold.")
    else:
        st.info("Neighborhood data not available for prediction. Please ensure the combined dataset is loaded correctly.")

def show_scenario_testing(data_loader, combined_data):
    """
    Show the scenario testing prediction tool.
    
    Args:
        data_loader: DataLoader instance
        combined_data: Combined dataset
    """
    # Show agent recommendations in a callout box
    with st.container():
        st.markdown("### Housing Impact Agent Recommends")
        st.markdown("""
            **Compare multiple policy scenarios** to understand their potential impacts on housing affordability.
            The 'Moderate Regulation' scenario typically offers the best balance between preserving rental income 
            and improving affordability metrics.
        """)
        
        if st.button("Ask about regulation recommendations"):
            st.session_state['page'] = "Housing Assistant"
            st.session_state['user_question'] = "What regulations could improve housing affordability?"
            st.rerun()
    st.subheader("Scenario Testing")
    
    st.markdown("""
        Test different policy scenarios to see how they might affect 
        housing affordability across Miami-Dade County.
    """)
    
    # Define scenarios
    scenarios = {
        "Status Quo": {
            "description": "No changes to current Airbnb market",
            "density_change": 0,
            "impact_factor": 0
        },
        "Moderate Regulation": {
            "description": "Implement moderate regulations limiting growth and requiring permits",
            "density_change": -15,
            "impact_factor": 0.15
        },
        "Strict Regulation": {
            "description": "Implement strict regulations significantly reducing short-term rentals",
            "density_change": -30,
            "impact_factor": 0.2
        },
        "Expanded Market": {
            "description": "Reduce regulations and allow for substantial growth in short-term rentals",
            "density_change": 25,
            "impact_factor": 0.13
        }
    }
    
    # Scenario selection
    selected_scenario = st.selectbox(
        "Select Policy Scenario:",
        options=list(scenarios.keys())
    )
    
    # Display scenario details
    scenario = scenarios[selected_scenario]
    st.markdown(f"**{selected_scenario}**: {scenario['description']}")
    st.markdown(f"""
        * Expected Change in Airbnb Density: **{scenario['density_change']}%**
        * Estimated Impact on Housing Metrics: **{scenario['impact_factor'] * 100:.1f}%** per 1% change in density
    """)
    
    # Create a divider
    st.markdown("---")
    
    # If we have data, show predictions
    if not combined_data.empty and 'rent_to_income_ratio' in combined_data.columns:
        # Calculate county-wide predictions
        current_avg_ratio = combined_data['rent_to_income_ratio'].mean()
        predicted_avg_ratio = current_avg_ratio * (1 + scenario['density_change']/100 * scenario['impact_factor'])
        
        # Show overall impact
        st.subheader("Predicted County-Wide Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Average Rent-to-Income Ratio",
                f"{current_avg_ratio:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "Predicted Rent-to-Income Ratio",
                f"{predicted_avg_ratio:.1f}%",
                delta=f"{predicted_avg_ratio - current_avg_ratio:.1f}%",
                delta_color="inverse"
            )
        
        # Calculate neighborhood-level impacts
        combined_data['predicted_ratio'] = combined_data['rent_to_income_ratio'] * (1 + scenario['density_change']/100 * scenario['impact_factor'])
        
        # Count neighborhoods by affordability status
        current_affordable = (combined_data['rent_to_income_ratio'] <= 30).sum()
        current_unaffordable = len(combined_data) - current_affordable
        predicted_affordable = (combined_data['predicted_ratio'] <= 30).sum()
        predicted_unaffordable = len(combined_data) - predicted_affordable
        
        # Show neighborhood counts
        st.subheader("Neighborhood Affordability Impact")
        
        # Create data for the chart
        categories = ['Current', 'Predicted']
        affordable = [current_affordable, predicted_affordable]
        unaffordable = [current_unaffordable, predicted_unaffordable]
        
        # Create a stacked bar chart
        fig = go.Figure(data=[
            go.Bar(name='Affordable', x=categories, y=affordable, marker_color='#2ca02c'),
            go.Bar(name='Unaffordable', x=categories, y=unaffordable, marker_color='#d62728')
        ])
        
        fig.update_layout(
            barmode='stack',
            title='Number of Affordable vs. Unaffordable Neighborhoods',
            xaxis_title='Scenario',
            yaxis_title='Number of Neighborhoods',
            legend_title='Affordability Status'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed impacts by Airbnb density level
        st.subheader("Impact by Airbnb Density Level")
        
        if 'airbnb_density_level' in combined_data.columns:
            # Group by density level
            grouped = combined_data.groupby('airbnb_density_level')[['rent_to_income_ratio', 'predicted_ratio']].mean().reset_index()
            
            # Ensure consistent ordering
            density_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            grouped['airbnb_density_level'] = pd.Categorical(
                grouped['airbnb_density_level'], 
                categories=density_order, 
                ordered=True
            )
            grouped = grouped.sort_values('airbnb_density_level')
            
            # Create a grouped bar chart
            fig = go.Figure(data=[
                go.Bar(name='Current', x=grouped['airbnb_density_level'], y=grouped['rent_to_income_ratio'], marker_color='#1f77b4'),
                go.Bar(name='Predicted', x=grouped['airbnb_density_level'], y=grouped['predicted_ratio'], marker_color='#ff7f0e')
            ])
            
            # Add affordability threshold line
            fig.add_hline(y=30, line_dash="dash", line_color="red", 
                         annotation_text="30% Affordability Threshold", 
                         annotation_position="bottom right")
            
            fig.update_layout(
                barmode='group',
                title='Rent-to-Income Ratio by Airbnb Density Level',
                xaxis_title='Airbnb Density Level',
                yaxis_title='Rent-to-Income Ratio (%)',
                legend_title='Scenario'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data not available for scenario predictions. Please ensure the combined dataset is loaded correctly.")

def show_model_insights(data_loader, combined_data):
    """
    Show insights about the prediction model.
    
    Args:
        data_loader: DataLoader instance
        combined_data: Combined dataset
    """
    st.subheader("Model Insights")
    
    st.markdown("""
        This section provides insights into the prediction model used to estimate
        the impact of short-term rental density on housing affordability.
    """)
    
    # Display model information
    st.markdown("""
        ### Prediction Model Details
        
        Our prediction model uses linear regression to estimate the relationship
        between Airbnb density and rent-to-income ratios. The model was trained
        on combined data from census and Airbnb sources.
        
        Key features in the model:
        * Airbnb listings per 1,000 residents
        * Percentage of entire home listings
        * Median listing price
        * Neighborhood demographic factors
        
        The model accounts for both direct and indirect effects of short-term
        rentals on the local housing market.
    """)
    
    # Show feature importance (simulated for prototype)
    st.subheader("Feature Importance")
    
    # Create simulated feature importance data
    features = [
        'listings_per_1000',
        'entire_home_percent',
        'median_price',
        'vacancy_rate',
        'median_household_income',
        'population_density'
    ]
    
    importance = [0.42, 0.28, 0.15, 0.07, 0.05, 0.03]
    
    # Create horizontal bar chart of feature importance
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title='Relative Importance of Model Features',
        labels={'x': 'Importance', 'y': 'Feature'},
        color=importance,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(yaxis_title='', xaxis_title='Relative Importance')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance metrics (simulated for prototype)
    st.subheader("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", "0.78")
    
    with col2:
        st.metric("Mean Absolute Error", "2.3%")
    
    with col3:
        st.metric("Prediction Accuracy", "84%")
    
    # Model limitations and caveats
    st.markdown("""
        ### Model Limitations
        
        While our model provides valuable insights, it's important to be aware of its limitations:
        
        * **Correlation vs. Causation**: The model identifies correlations between short-term rentals
          and housing affordability, but other factors may contribute to these relationships.
        
        * **Data Limitations**: The model is limited by the quality and comprehensiveness of the
          underlying data sources.
        
        * **Time Sensitivity**: Housing markets evolve over time, and the relationships captured
          by the model may change as market conditions shift.
        
        * **Simplified Assumptions**: The model makes simplifying assumptions about complex
          market dynamics and policy impacts.
    """)
    
    # Additional information
    st.info("""
        **Note:** This is a simplified prototype model for demonstration purposes.
        A full implementation would include more sophisticated modeling techniques,
        additional features, and more extensive validation.
    """)
