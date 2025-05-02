"""
Helper functions for the Miami Housing Impact Hub Streamlit app.
"""
import os
import json
import functools
import streamlit as st
import pandas as pd

def load_config():
    """
    Load the application configuration from config.json.
    If the file doesn't exist, return default configuration.
    """
    default_config = {
        "paths": {
            "raw": "data/raw",
            "processed": "data/processed",
            "visualizations": "visualizations"
        }
    }
    
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return default_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return default_config

def ensure_data_loaded(datasets):
    """
    Decorator to ensure required datasets are loaded before executing a function.
    
    Args:
        datasets (list): List of dataset names that need to be loaded
    
    Returns:
        Function decorator
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data_loader, *args, **kwargs):
            missing_datasets = []
            
            for dataset in datasets:
                if data_loader.get_data(dataset) is None:
                    missing_datasets.append(dataset)
            
            if missing_datasets:
                st.warning(f"The following datasets are required but not loaded: {', '.join(missing_datasets)}")
                st.info("Please load the required datasets from the data processing section.")
                return
            
            return func(data_loader, *args, **kwargs)
        return wrapper
    return decorator
