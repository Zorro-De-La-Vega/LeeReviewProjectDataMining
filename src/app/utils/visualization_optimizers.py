"""
Visualization Optimization Utilities

This module provides optimized visualization functions and wrappers
to improve performance of data visualization in Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import functools

@st.cache_data(ttl=3600)
def optimized_plotly_figure(func):
    """
    Decorator to cache Plotly figure generation.
    
    Args:
        func: Function that generates a Plotly figure
        
    Returns:
        Decorated function that caches the figure
    """
    def wrapper(*args, **kwargs):
        # Record execution time for performance monitoring
        start_time = time.time()
        fig = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Log performance metrics for expensive operations
        if elapsed > 0.5:
            func_name = func.__name__
            st.session_state.setdefault('vis_performance', {})
            st.session_state['vis_performance'][func_name] = {
                'last_execution_time': elapsed,
                'call_count': st.session_state.get('vis_performance', {}).get(func_name, {}).get('call_count', 0) + 1
            }
        
        # Apply Streamlit-specific optimizations to the figure
        if fig is not None:
            # Add performance optimization configurations to layout
            if hasattr(fig, 'update_layout'):
                fig.update_layout(
                    uirevision='constant',  # Preserve UI state on updates
                    modebar_remove=['sendDataToCloud', 'autoScale', 'resetScale'],
                    hovermode='closest'
                )
        
        return fig
    
    return wrapper

def optimize_df_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame for display in Streamlit.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    if df is None or df.empty:
        return df
    
    # If DataFrame is too large, sample it
    if len(df) > 10000:
        df = df.sample(5000, random_state=42)
    
    # Optimize memory usage by downcasting numeric columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    return df

def create_efficient_map(
    df: pd.DataFrame, 
    lat_col: str = 'latitude', 
    lon_col: str = 'longitude',
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    zoom: int = 10,
    max_points: int = 1000
) -> go.Figure:
    """
    Create an efficient map visualization optimized for performance.
    
    Args:
        df: DataFrame with geographic data
        lat_col: Column name for latitude
        lon_col: Column name for longitude
        color_col: Column name for color mapping
        size_col: Column name for point sizing
        zoom: Initial zoom level
        max_points: Maximum number of points to display
        
    Returns:
        Plotly figure with map
    """
    # Ensure required columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"DataFrame must contain {lat_col} and {lon_col} columns")
    
    # Filter out invalid coordinates
    df = df.dropna(subset=[lat_col, lon_col])
    df = df[(df[lat_col] != 0) & (df[lon_col] != 0)]
    
    # Subsample for performance if needed
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    
    # Create the map
    if color_col and color_col in df.columns:
        fig = px.scatter_mapbox(
            df, 
            lat=lat_col, 
            lon=lon_col, 
            color=color_col,
            size=size_col if size_col and size_col in df.columns else None,
            zoom=zoom,
            mapbox_style="carto-darkmatter",  # Changed to dark theme for better visibility
            height=600,
            opacity=0.7
            # Removed render_mode parameter for compatibility
        )
    else:
        fig = px.scatter_mapbox(
            df, 
            lat=lat_col, 
            lon=lon_col,
            size=size_col if size_col and size_col in df.columns else None,
            zoom=zoom,
            mapbox_style="carto-darkmatter",  # Changed to dark theme for better visibility
            height=600,
            opacity=0.7
            # Removed render_mode parameter for compatibility
        )
    
    # Performance optimizations
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        uirevision='constant',
        modebar_remove=['sendDataToCloud', 'autoScale', 'resetScale'],
        hovermode='closest'
    )
    
    return fig

@st.cache_data(ttl=3600)
def create_efficient_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    max_categories: int = 20
) -> go.Figure:
    """
    Create an efficient bar chart optimized for performance.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for color
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        max_categories: Maximum number of categories to display
        
    Returns:
        Plotly figure with bar chart
    """
    # Ensure required columns exist
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"DataFrame must contain {x_col} and {y_col} columns")
    
    # For categorical x variables, limit number of categories for performance
    if df[x_col].dtype == 'object' or df[x_col].dtype.name == 'category':
        if df[x_col].nunique() > max_categories:
            # Get top categories by y_col value
            top_categories = df.groupby(x_col)[y_col].sum().nlargest(max_categories).index.tolist()
            df = df[df[x_col].isin(top_categories)]
    
    # Create the bar chart
    if color_col and color_col in df.columns:
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            labels={
                x_col: x_label if x_label else x_col,
                y_col: y_label if y_label else y_col
            }
        )
    else:
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col,
            title=title,
            labels={
                x_col: x_label if x_label else x_col,
                y_col: y_label if y_label else y_col
            }
        )
    
    # Performance optimizations
    fig.update_layout(
        uirevision='constant',
        modebar_remove=['sendDataToCloud', 'autoScale', 'resetScale']
    )
    
    return fig

def show_visualization_performance_metrics():
    """Display visualization performance metrics in a Streamlit expander."""
    if 'vis_performance' not in st.session_state:
        return
    
    with st.expander("Visualization Performance Metrics", expanded=False):
        metrics_data = []
        
        for func_name, stats in st.session_state['vis_performance'].items():
            metrics_data.append({
                'Visualization Function': func_name,
                'Last Execution Time (s)': round(stats['last_execution_time'], 3),
                'Call Count': stats['call_count']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df = metrics_df.sort_values('Last Execution Time (s)', ascending=False)
        
        st.dataframe(metrics_df, use_container_width=True)
        
        if st.button("Clear Visualization Metrics"):
            st.session_state['vis_performance'] = {}
            st.experimental_rerun()
