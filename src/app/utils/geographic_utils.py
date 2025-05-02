"""
Geographic Utilities Module

This module provides utilities for handling geographic data in the Miami Housing Impact Hub application.
It includes functions to add coordinates to neighborhood data for mapping visualizations.
"""

import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

def enrich_neighborhood_data_with_coordinates(combined_data):
    """
    Enrich the neighborhood data with geographic coordinates from existing datasets.
    
    This function looks for geographic coordinates in various data sources and adds them
    to the neighborhood data for mapping visualizations.
    
    Args:
        combined_data (pd.DataFrame): The combined dataset with neighborhood data
        
    Returns:
        pd.DataFrame: The enriched dataset with latitude and longitude columns
    """
    # Make a copy to avoid modifying the original dataframe
    enriched_data = combined_data.copy()
    
    # Check if coordinates already exist
    if 'latitude' in enriched_data.columns and 'longitude' in enriched_data.columns:
        # Filter out rows with missing or invalid coordinates
        valid_coords = (
            enriched_data['latitude'].notna() & 
            enriched_data['longitude'].notna() & 
            (enriched_data['latitude'] != 0) & 
            (enriched_data['longitude'] != 0)
        )
        
        if valid_coords.sum() > 0:
            logger.info(f"Using existing coordinates for {valid_coords.sum()} of {len(enriched_data)} neighborhoods")
            return enriched_data
    
    # Initialize coordinate columns if they don't exist
    if 'latitude' not in enriched_data.columns:
        enriched_data['latitude'] = np.nan
    if 'longitude' not in enriched_data.columns:
        enriched_data['longitude'] = np.nan
    
    # Path to zipcode analysis data which contains coordinates
    zipcode_file_path = os.path.join('data', 'consolidated', 'zipcode_analysis.csv')
    
    try:
        if os.path.exists(zipcode_file_path):
            # Load zipcode data with coordinates
            zipcode_data = pd.read_csv(zipcode_file_path)
            
            # Check if the file has the necessary columns
            required_cols = ['neighborhood', 'latitude_airbnb', 'longitude_airbnb']
            if all(col in zipcode_data.columns for col in required_cols):
                # Drop rows with missing neighborhoods or coordinates
                valid_zipcode_data = zipcode_data.dropna(subset=required_cols)
                
                # Create a dictionary mapping neighborhoods to coordinates
                # Use the mean coordinates if multiple entries exist for the same neighborhood
                neighborhood_coords = {}
                
                for neighborhood, group in valid_zipcode_data.groupby('neighborhood'):
                    if pd.notna(neighborhood) and neighborhood:
                        avg_lat = group['latitude_airbnb'].mean()
                        avg_lon = group['longitude_airbnb'].mean()
                        if pd.notna(avg_lat) and pd.notna(avg_lon):
                            neighborhood_coords[neighborhood] = (avg_lat, avg_lon)
                
                # Add coordinates to neighborhoods in enriched_data
                for idx, row in enriched_data.iterrows():
                    neighborhood = row['neighborhood']
                    
                    # Try direct match
                    if neighborhood in neighborhood_coords:
                        enriched_data.at[idx, 'latitude'] = neighborhood_coords[neighborhood][0]
                        enriched_data.at[idx, 'longitude'] = neighborhood_coords[neighborhood][1]
                    
                    # Try case-insensitive match
                    elif neighborhood.lower() in {k.lower(): k for k in neighborhood_coords.keys()}:
                        # Find the actual key with case-insensitive match
                        for k in neighborhood_coords.keys():
                            if k.lower() == neighborhood.lower():
                                enriched_data.at[idx, 'latitude'] = neighborhood_coords[k][0]
                                enriched_data.at[idx, 'longitude'] = neighborhood_coords[k][1]
                                break
                
                # Log stats about the enrichment
                coords_count = enriched_data.dropna(subset=['latitude', 'longitude']).shape[0]
                logger.info(f"Added coordinates to {coords_count} of {len(enriched_data)} neighborhoods")
                
                # Add a comprehensive fallback for neighborhoods without coordinates
                # These are real geographic coordinates for Miami-Dade neighborhoods
                # Data sourced from official Miami-Dade County GIS data and OpenStreetMap
                fallback_coords = {
                    # Downtown and Urban Core
                    'Downtown': (25.7742, -80.1936),
                    'Brickell': (25.7616, -80.1948),
                    'Edgewater': (25.7983, -80.1893),
                    'Overtown': (25.7865, -80.1989),
                    'Omni': (25.7912, -80.1869),
                    'Park West': (25.7848, -80.1895),
                    'Midtown': (25.8040, -80.1927),
                    
                    # Miami Beach
                    'Miami Beach': (25.7907, -80.1300),
                    'South Beach': (25.7825, -80.1340),
                    'North Beach': (25.8555, -80.1197),
                    'Mid-Beach': (25.8127, -80.1223),
                    'South Pointe': (25.7684, -80.1341),
                    'Bal Harbour': (25.8886, -80.1253),
                    'Surfside': (25.8784, -80.1256),
                    'Sunny Isles Beach': (25.9373, -80.1220),
                    
                    # Arts and Culture Districts
                    'Wynwood': (25.8049, -80.1986),
                    'Design District': (25.8131, -80.1925),
                    'Arts & Entertainment District': (25.7901, -80.1872),
                    'Little Haiti': (25.8156, -80.1908),
                    
                    # Residential Neighborhoods
                    'Coconut Grove': (25.7242, -80.2456),
                    'Coral Gables': (25.7215, -80.2684),
                    'Key Biscayne': (25.6906, -80.1628),
                    'Little Havana': (25.7748, -80.2289),
                    'Allapattah': (25.7995, -80.2262),
                    'Upper Eastside': (25.8379, -80.1849),
                    'Buena Vista': (25.8218, -80.1912),
                    'Morningside': (25.8254, -80.1762),
                    'Belle Meade': (25.8430, -80.1695),
                    'Shorecrest': (25.8535, -80.1790),
                    
                    # Northern Miami Neighborhoods
                    'Little River': (25.8335, -80.1911),
                    'El Portal': (25.8511, -80.1926),
                    'Miami Shores': (25.8626, -80.1787),
                    'North Miami': (25.8900, -80.1867),
                    'North Miami Beach': (25.9326, -80.1624),
                    'Aventura': (25.9559, -80.1391),
                    
                    # West Miami Neighborhoods
                    'Flagami': (25.7597, -80.3042),
                    'Westchester': (25.7530, -80.3447),
                    'Sweetwater': (25.7755, -80.3741),
                    'Doral': (25.8124, -80.3554),
                    
                    # Southern Miami Neighborhoods
                    'Coral Way': (25.7476, -80.2424),
                    'The Roads': (25.7566, -80.2124),
                    'Silver Bluff': (25.7472, -80.2344),
                    'Shenandoah': (25.7517, -80.2216),
                    'Grapeland Heights': (25.7855, -80.2470),
                    
                    # South Miami
                    'South Miami': (25.7060, -80.2960),
                    'Pinecrest': (25.6677, -80.3038),
                    'Palmetto Bay': (25.6286, -80.3211),
                    'Kendall': (25.6669, -80.3566),
                    'The Falls': (25.6214, -80.3438),
                    
                    # Western Suburbs
                    'Hialeah': (25.8575, -80.2781),
                    'Hialeah Gardens': (25.8917, -80.3471),
                    'Miami Lakes': (25.9087, -80.3089),
                    'Medley': (25.9110, -80.3341),
                    'Miami Springs': (25.8226, -80.2889)
                }
                
                # Normalize neighborhood names for better matching
                def normalize_name(name):
                    if not name or not isinstance(name, str):
                        return ''
                    # Remove common prefixes/suffixes and normalize case
                    normalized = name.lower()
                    normalized = normalized.replace('the ', '')
                    normalized = normalized.replace(' area', '')
                    normalized = normalized.replace(' district', '')
                    normalized = normalized.replace(' neighborhood', '')
                    normalized = normalized.replace(' community', '')
                    return normalized.strip()
                
                # Create a normalized version of the fallback dictionary for better matching
                normalized_fallbacks = {}
                for k, v in fallback_coords.items():
                    normalized_fallbacks[normalize_name(k)] = v
                
                # Apply fallbacks with improved matching for missing coordinates
                for idx, row in enriched_data.iterrows():
                    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                        neighborhood = row['neighborhood']
                        
                        # Try direct match first
                        if neighborhood in fallback_coords:
                            enriched_data.at[idx, 'latitude'] = fallback_coords[neighborhood][0]
                            enriched_data.at[idx, 'longitude'] = fallback_coords[neighborhood][1]
                            continue
                            
                        # Try normalized match
                        normalized = normalize_name(neighborhood)
                        if normalized in normalized_fallbacks:
                            enriched_data.at[idx, 'latitude'] = normalized_fallbacks[normalized][0]
                            enriched_data.at[idx, 'longitude'] = normalized_fallbacks[normalized][1]
                            continue
                            
                        # Try fuzzy matching for close matches
                        # This handles minor spelling variations or word order differences
                        best_match = None
                        best_score = 0
                        for k in fallback_coords.keys():
                            # Simple overlap score between words
                            k_words = set(normalize_name(k).split())
                            n_words = set(normalized.split())
                            if not k_words or not n_words:
                                continue
                                
                            # Calculate overlap score
                            common_words = k_words.intersection(n_words)
                            if len(common_words) > 0:
                                score = len(common_words) / max(len(k_words), len(n_words))
                                if score > 0.5 and score > best_score:  # Require at least 50% word match
                                    best_score = score
                                    best_match = k
                        
                        # Apply the best fuzzy match if found
                        if best_match:
                            enriched_data.at[idx, 'latitude'] = fallback_coords[best_match][0]
                            enriched_data.at[idx, 'longitude'] = fallback_coords[best_match][1]
            else:
                logger.warning(f"Zipcode file missing required columns. Available: {zipcode_data.columns.tolist()}")
        else:
            logger.warning(f"Zipcode data file not found at {zipcode_file_path}")
    except Exception as e:
        logger.error(f"Error enriching neighborhood data with coordinates: {str(e)}")
    
    # Return the enriched data
    return enriched_data

def get_miami_map_center():
    """
    Returns the default center coordinates for Miami-Dade County maps.
    
    Returns:
        tuple: (latitude, longitude) for the center of Miami-Dade County
    """
    return (25.7617, -80.1918)  # Downtown Miami coordinates
