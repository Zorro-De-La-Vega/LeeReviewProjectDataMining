"""
Streamlit Performance Optimization Utilities

This module provides a collection of decorators and utilities to optimize
Streamlit application performance following official Streamlit best practices.
"""

import streamlit as st
import time
import functools
import pandas as pd
from pathlib import Path
import hashlib
import pickle
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

def st_cached_data(ttl: int = 3600, show_spinner: bool = False):
    """
    Decorator for caching data loading functions with proper TTL.
    
    Args:
        ttl (int): Time to live in seconds for the cached data
        show_spinner (bool): Whether to show a loading spinner
        
    Returns:
        Callable: Decorated function with caching
    """
    def decorator(func):
        # Convert Path objects to strings to avoid hashing issues
        @functools.wraps(func)
        def path_converter(*args, **kwargs):
            # Convert Path objects to strings to make them hashable
            new_args = [str(arg) if isinstance(arg, Path) else arg for arg in args]
            new_kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
            
        # Use Streamlit's cache_data decorator
        return st.cache_data(ttl=ttl, show_spinner=show_spinner)(path_converter)
    return decorator

def st_cached_resource(show_spinner: bool = False):
    """
    Decorator for caching model and resource initialization.
    
    Args:
        show_spinner (bool): Whether to show a loading spinner
        
    Returns:
        Callable: Decorated function with caching
    """
    def decorator(func):
        # Convert Path objects to strings to avoid hashing issues
        @functools.wraps(func)
        def path_converter(*args, **kwargs):
            # Convert Path objects to strings to make them hashable
            new_args = [str(arg) if isinstance(arg, Path) else arg for arg in args]
            new_kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
            
        # Use Streamlit's cache_resource decorator
        return st.cache_resource(show_spinner=show_spinner)(path_converter)
    return decorator

def lazy_load(func: Callable) -> Callable:
    """
    Creates a lazy-loading function that only executes when called.
    
    Args:
        func (Callable): Function to lazy load
        
    Returns:
        Callable: Lazy loading function wrapper
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        # Log performance metrics for long operations
        if elapsed > 0.5:  # Only log operations that take significant time
            function_name = func.__name__
            st.session_state.setdefault('performance_metrics', {})
            st.session_state['performance_metrics'][function_name] = {
                'last_execution_time': elapsed,
                'call_count': st.session_state.get('performance_metrics', {}).get(function_name, {}).get('call_count', 0) + 1
            }
        return result
    return wrapper

def optimize_dataframe(df: pd.DataFrame, sampling_threshold: int = 10000, 
                     sampling_size: int = 5000) -> pd.DataFrame:
    """
    Optimizes a DataFrame for Streamlit display by downsampling large DataFrames.
    
    Args:
        df (pd.DataFrame): DataFrame to optimize
        sampling_threshold (int): Row count threshold above which to sample
        sampling_size (int): Number of rows to sample 
        
    Returns:
        pd.DataFrame: Optimized DataFrame (sampled if needed)
    """
    if df is None or df.empty:
        return df
        
    # Only sample if exceeding threshold
    if len(df) > sampling_threshold:
        return df.sample(sampling_size, random_state=42)
    
    return df

class ProgressSpinner:
    """Context manager for showing a progress spinner during long operations."""
    
    def __init__(self, text: str = "Processing...", key: Optional[str] = None):
        """
        Initialize the progress spinner.
        
        Args:
            text (str): Text to display next to the spinner
            key (str, optional): Unique key for this spinner
        """
        self.text = text
        self.key = key or f"spinner_{text}"
        self.spinner = None
        
    def __enter__(self):
        """Enter the context manager and display the spinner."""
        self.spinner = st.spinner(self.text)
        self.spinner.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and hide the spinner."""
        self.spinner.__exit__(exc_type, exc_val, exc_tb)
        
    def update(self, text: str):
        """
        Update the spinner text.
        
        Args:
            text (str): New text to display
        """
        self.__exit__(None, None, None)
        self.text = text
        self.spinner = st.spinner(self.text)
        self.spinner.__enter__()

def get_performance_dashboard() -> None:
    """
    Display a performance dashboard with execution metrics.
    
    This function is useful for debugging performance issues.
    """
    if 'performance_metrics' not in st.session_state:
        st.info("No performance metrics collected yet.")
        return
        
    metrics = st.session_state['performance_metrics']
    
    # Convert to DataFrame for display
    data = []
    for function_name, stats in metrics.items():
        data.append({
            'Function': function_name,
            'Last Execution Time (s)': round(stats['last_execution_time'], 3),
            'Call Count': stats['call_count']
        })
        
    metrics_df = pd.DataFrame(data)
    
    # Sort by execution time (slowest first)
    metrics_df = metrics_df.sort_values('Last Execution Time (s)', ascending=False)
    
    st.dataframe(metrics_df)
    
    # Add clear button
    if st.button("Clear Performance Metrics"):
        st.session_state['performance_metrics'] = {}
        st.experimental_rerun()
