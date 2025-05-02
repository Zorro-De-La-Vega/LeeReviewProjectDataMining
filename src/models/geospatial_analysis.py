#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Geospatial Analysis Module

This module implements advanced geospatial analysis techniques for the Miami Housing Impact Hub,
including spatial autocorrelation, hotspot detection, and spatial clustering.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.cm as cm
import contextily as ctx
from shapely.geometry import Point, Polygon, MultiPolygon
import pysal
from esda.moran import Moran, Moran_Local
from esda.getisord import G, G_Local
from libpysal.weights import KNN, Queen, Kernel, DistanceBand
import warnings
warnings.filterwarnings('ignore')

class GeospatialHotspotAnalysis:
    """
    Implementation of geospatial hotspot analysis techniques for identifying
    spatial patterns in housing and Airbnb data in Miami-Dade County.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the geospatial analysis module.
        
        Args:
            data_dir: Path to data directory (optional)
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[2]
            self.data_dir = self.project_root / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        self.output_dir = self.project_root / "visualizations" / "geospatial"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize attributes
        self.data = None
        self.geojson_path = None
        self.neighborhoods_gdf = None
        self.points_gdf = None
        self.hotspot_results = {}
        self.weights = None
        self.moran_results = {}
        self.getis_results = {}
        self.dbscan_results = {}
        
        # Metric columns for analysis
        self.potential_metrics = [
            'airbnb_count', 
            'airbnb_density',
            'median_property_price', 
            'median_airbnb_price',
            'median_income',
            'price_to_income_ratio',
            'rent_to_income_ratio'
        ]
        
        # Set default parameters
        self.default_params = {
            'knn': 5,                 # Number of nearest neighbors
            'p_threshold': 0.05,      # Significance threshold
            'queen_contiguity': True, # Use queen contiguity for weights
            'eps_distance': 0.02,     # Distance threshold for DBSCAN
            'min_samples': 5          # Minimum samples for DBSCAN
        }
    
    def load_data(self):
        """Load the processed data for geospatial analysis."""
        try:
            # Try to load the merged dataset
            data_path = self.data_dir / "miami_dade_merged_data.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                print(f"Loaded {len(self.data)} records from {data_path}")
                
                # Check if we have geospatial coordinates
                if not all(col in self.data.columns for col in ['latitude', 'longitude']):
                    print("Warning: Missing coordinate data. Loading additional spatial data...")
                    
                    # Try to load Airbnb data with coordinates
                    airbnb_path = self.data_dir / "airbnb" / "processed_airbnb_data.csv"
                    if airbnb_path.exists():
                        airbnb_data = pd.read_csv(airbnb_path)
                        if all(col in airbnb_data.columns for col in ['latitude', 'longitude', 'neighborhood']):
                            print(f"Using coordinates from Airbnb data ({len(airbnb_data)} records)")
                            # Aggregate by neighborhood and get centroid
                            airbnb_agg = airbnb_data.groupby('neighborhood').agg({
                                'latitude': 'mean',
                                'longitude': 'mean',
                                'id': 'count'
                            }).reset_index().rename(columns={'id': 'airbnb_count'})
                            
                            # Merge with main data
                            self.data = pd.merge(
                                self.data, 
                                airbnb_agg[['neighborhood', 'latitude', 'longitude']], 
                                on='neighborhood', 
                                how='left'
                            )
                
                # Try to find neighborhood boundary data (GeoJSON format)
                geojson_path = self.project_root / "data" / "raw" / "boundaries" / "miami_neighborhoods.geojson"
                if geojson_path.exists():
                    self.geojson_path = geojson_path
                    print(f"Found neighborhood boundaries at {geojson_path}")
                else:
                    print("Warning: No neighborhood boundary data found at expected location.")
                    print("Will use point data for analysis but spatial weights will be limited.")
                
                # Check for sufficient coordinate data
                if not self.data[['latitude', 'longitude']].notna().all().all():
                    print("Warning: Missing coordinate data for some records.")
                    # Fill in missing coordinates with neighborhood averages
                    self.data = self.data.groupby('neighborhood').apply(
                        lambda x: x.fillna({
                            'latitude': x['latitude'].mean(),
                            'longitude': x['longitude'].mean()
                        })
                    ).reset_index(drop=True)
                
                # Select metrics available in the data
                self.metrics = [m for m in self.potential_metrics if m in self.data.columns]
                if not self.metrics:
                    print("Warning: No analysis metrics found in data.")
                    # Try to use any numeric columns instead
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                    self.metrics = [c for c in numeric_cols if c not in ['latitude', 'longitude', 'id']]
                    print(f"Using available numeric columns: {', '.join(self.metrics)}")
                
                return True
            else:
                print(f"Error: Could not find dataset at {data_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_spatial_data(self):
        """Prepare spatial data for analysis by converting to GeoDataFrame."""
        if self.data is None:
            if not self.load_data():
                return False
        
        try:
            # Create point geometry from latitude and longitude
            geometry = [Point(xy) for xy in zip(self.data['longitude'], self.data['latitude'])]
            self.points_gdf = gpd.GeoDataFrame(self.data, geometry=geometry, crs="EPSG:4326")
            print(f"Created GeoDataFrame with {len(self.points_gdf)} points")
            
            # Load neighborhood boundaries if available
            if self.geojson_path is not None:
                try:
                    self.neighborhoods_gdf = gpd.read_file(self.geojson_path)
                    print(f"Loaded {len(self.neighborhoods_gdf)} neighborhood boundaries")
                    
                    # Make sure we have a common field to join on
                    if 'neighborhood' in self.neighborhoods_gdf.columns:
                        join_field = 'neighborhood'
                    elif 'name' in self.neighborhoods_gdf.columns:
                        join_field = 'name'
                        self.neighborhoods_gdf = self.neighborhoods_gdf.rename(columns={'name': 'neighborhood'})
                    else:
                        # Try to find a field that might contain neighborhood names
                        possible_fields = ['NAME', 'NBHD', 'NEIGHBORHOOD', 'AREA_NAME', 'LABEL']
                        for field in possible_fields:
                            if field in self.neighborhoods_gdf.columns:
                                self.neighborhoods_gdf = self.neighborhoods_gdf.rename(columns={field: 'neighborhood'})
                                break
                    
                    if 'neighborhood' not in self.neighborhoods_gdf.columns:
                        print("Warning: Could not identify neighborhood field in GeoJSON.")
                        # Create a simple name based on index
                        self.neighborhoods_gdf['neighborhood'] = [f"Area_{i}" for i in range(len(self.neighborhoods_gdf))]
                    
                    # Ensure neighborhoods_gdf has the same CRS as points_gdf
                    if self.neighborhoods_gdf.crs != self.points_gdf.crs:
                        self.neighborhoods_gdf = self.neighborhoods_gdf.to_crs(self.points_gdf.crs)
                    
                    # Spatial join to associate points with neighborhoods if needed
                    if 'neighborhood' not in self.points_gdf.columns:
                        self.points_gdf = gpd.sjoin(self.points_gdf, self.neighborhoods_gdf[['neighborhood', 'geometry']], 
                                                   how='left', op='within')
                    
                    # Calculate neighborhood-level metrics
                    neighborhood_metrics = {}
                    
                    for metric in self.metrics:
                        # Skip if metric doesn't exist in points_gdf
                        if metric not in self.points_gdf.columns:
                            continue
                            
                        # Aggregate by neighborhood
                        agg_data = self.points_gdf.groupby('neighborhood')[metric].mean().reset_index()
                        
                        # Rename to match the neighborhood field and merge
                        self.neighborhoods_gdf = pd.merge(
                            self.neighborhoods_gdf,
                            agg_data,
                            on='neighborhood',
                            how='left'
                        )
                        
                        # Store the aggregated data
                        neighborhood_metrics[metric] = agg_data
                    
                    print(f"Joined point data to {len(self.neighborhoods_gdf)} neighborhoods")
                    print(f"Available metrics at neighborhood level: {list(neighborhood_metrics.keys())}")
                
                except Exception as e:
                    print(f"Error processing neighborhood boundaries: {e}")
                    self.neighborhoods_gdf = None
            
            # If we don't have neighborhoods, we'll use the point data directly
            if self.neighborhoods_gdf is None:
                print("No valid neighborhood boundaries found. Using point data for analysis.")
            
            return True
        
        except Exception as e:
            print(f"Error preparing spatial data: {e}")
            return False
    
    def create_spatial_weights(self, weight_type='queen', **kwargs):
        """
        Create spatial weights matrix for spatial autocorrelation.
        
        Args:
            weight_type: Type of spatial weights to create ('queen', 'knn', 'distance', 'kernel')
            **kwargs: Additional parameters for weight creation
        """
        if self.neighborhoods_gdf is not None:
            gdf = self.neighborhoods_gdf
            print(f"Creating {weight_type} weights for neighborhood polygons...")
        else:
            gdf = self.points_gdf
            print(f"Creating {weight_type} weights for point data...")
        
        try:
            if weight_type == 'queen':
                # Contiguity-based weights (only works for polygons)
                if self.neighborhoods_gdf is not None:
                    self.weights = Queen.from_dataframe(gdf)
                else:
                    print("Cannot create Queen weights for point data. Switching to KNN.")
                    weight_type = 'knn'
            
            if weight_type == 'knn':
                # K-nearest neighbors weights
                k = kwargs.get('k', self.default_params['knn'])
                self.weights = KNN.from_dataframe(gdf, k=k)
                print(f"Created KNN weights with k={k}")
            
            elif weight_type == 'distance':
                # Distance-based weights
                threshold = kwargs.get('threshold', None)
                if threshold is None:
                    # Calculate threshold as average distance to k nearest neighbors
                    k = kwargs.get('k', self.default_params['knn'])
                    knn = KNN.from_dataframe(gdf, k=k)
                    threshold = knn.neighbors.to_numpy()
                    threshold = np.mean([np.max(gdf.distance(gdf.iloc[i].geometry)) for i in range(len(gdf))])
                
                self.weights = DistanceBand.from_dataframe(gdf, threshold=threshold, binary=True)
                print(f"Created distance band weights with threshold={threshold}")
            
            elif weight_type == 'kernel':
                # Kernel weights
                bandwidth = kwargs.get('bandwidth', None)
                if bandwidth is None:
                    # Calculate bandwidth as average distance to k nearest neighbors
                    k = kwargs.get('k', self.default_params['knn'])
                    knn = KNN.from_dataframe(gdf, k=k)
                    bandwidth = np.mean([np.max(gdf.distance(gdf.iloc[i].geometry)) for i in range(len(gdf))])
                
                self.weights = Kernel.from_dataframe(gdf, bandwidth=bandwidth)
                print(f"Created kernel weights with bandwidth={bandwidth}")
            
            # Ensure weights are row-standardized
            self.weights.transform = 'r'
            
            return True
        
        except Exception as e:
            print(f"Error creating spatial weights: {e}")
            return False

    def compute_global_moran(self, variable):
        """
        Compute Global Moran's I to measure spatial autocorrelation.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Dictionary with Moran's I results
        """
        if self.weights is None:
            if not self.create_spatial_weights():
                return None
        
        if self.neighborhoods_gdf is not None:
            gdf = self.neighborhoods_gdf
        else:
            gdf = self.points_gdf
        
        # Check if variable exists in the dataframe
        if variable not in gdf.columns:
            print(f"Error: Variable '{variable}' not found in data")
            return None
        
        # Extract values and filter out NaN values
        values = gdf[variable].values
        if np.isnan(values).any():
            print(f"Warning: {np.isnan(values).sum()} NaN values found in '{variable}'")
            # Use only non-NaN indices for analysis
            valid_indices = ~np.isnan(values)
            values = values[valid_indices]
            # Create a subset of weights for valid indices
            weights_subset = self.weights.subset(valid_indices)
        else:
            weights_subset = self.weights
        
        try:
            # Compute Global Moran's I
            moran = Moran(values, weights_subset)
            
            # Store results
            self.moran_results[variable] = {
                'I': moran.I,  # Moran's I statistic
                'EI': moran.EI,  # Expected value
                'p_norm': moran.p_norm,  # p-value (normal approximation)
                'p_rand': moran.p_sim,  # p-value (permutation)
                'z_norm': moran.z_norm,  # z-score (normal approximation)
                'sim': moran.sim,  # Array of simulated values
                'variable': variable,  # Variable analyzed
                'weights': weights_subset  # Spatial weights matrix
            }
            
            print(f"Global Moran's I for {variable}: {moran.I:.4f} (p-value: {moran.p_norm:.4f})")
            return self.moran_results[variable]
        
        except Exception as e:
            print(f"Error computing Global Moran's I: {e}")
            return None
    
    def compute_local_moran(self, variable):
        """
        Compute Local Moran's I to identify clusters and outliers.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Dictionary with Local Moran's I results
        """
        if self.weights is None:
            if not self.create_spatial_weights():
                return None
        
        if self.neighborhoods_gdf is not None:
            gdf = self.neighborhoods_gdf
        else:
            gdf = self.points_gdf
        
        # Check if variable exists in the dataframe
        if variable not in gdf.columns:
            print(f"Error: Variable '{variable}' not found in data")
            return None
        
        # Extract values and filter out NaN values
        values = gdf[variable].values
        if np.isnan(values).any():
            print(f"Warning: {np.isnan(values).sum()} NaN values found in '{variable}'")
            # Use only non-NaN indices for analysis
            valid_indices = ~np.isnan(values)
            values = values[valid_indices]
            # Create a subset of weights for valid indices
            weights_subset = self.weights.subset(valid_indices)
            # Create filtered GeoDataFrame
            gdf_filtered = gdf.iloc[valid_indices]
        else:
            weights_subset = self.weights
            gdf_filtered = gdf
        
        try:
            # Compute Local Moran's I
            lisa = Moran_Local(values, weights_subset, permutations=999)
            
            # Create a copy of the geodataframe with results
            results_gdf = gdf_filtered.copy()
            
            # Add Local Moran statistics to the dataframe
            results_gdf['local_moran'] = lisa.Is
            results_gdf['p_value'] = lisa.p_sim
            results_gdf['quad'] = lisa.q
            results_gdf['significant'] = lisa.p_sim < self.default_params['p_threshold']
            
            # Create cluster classification
            # 1=HH (hot spot), 2=LH (outlier), 3=LL (cold spot), 4=HL (outlier)
            results_gdf['cluster_type'] = pd.Series(lisa.q).map({1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'})
            
            # Filter significant clusters
            results_gdf['sig_cluster'] = None
            mask = results_gdf['significant']
            results_gdf.loc[mask, 'sig_cluster'] = results_gdf.loc[mask, 'cluster_type']
            
            # Store results
            self.moran_results[f"local_{variable}"] = {
                'lisa': lisa,  # Local Moran statistic object
                'results_gdf': results_gdf,  # GeoDataFrame with results
                'variable': variable,  # Variable analyzed
                'weights': weights_subset  # Spatial weights matrix
            }
            
            # Count significant clusters by type
            cluster_counts = results_gdf[results_gdf['significant']]['cluster_type'].value_counts()
            print(f"Local Moran's I for {variable} identified:")
            for cluster_type, count in cluster_counts.items():
                print(f"  {count} {cluster_type} clusters/outliers")
                
            return self.moran_results[f"local_{variable}"]
        
        except Exception as e:
            print(f"Error computing Local Moran's I: {e}")
            return None
    
    def compute_getis_ord(self, variable):
        """
        Compute Getis-Ord G* statistic to identify hot and cold spots.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Dictionary with Getis-Ord G* results
        """
        if self.weights is None:
            if not self.create_spatial_weights():
                return None
        
        if self.neighborhoods_gdf is not None:
            gdf = self.neighborhoods_gdf
        else:
            gdf = self.points_gdf
        
        # Check if variable exists in the dataframe
        if variable not in gdf.columns:
            print(f"Error: Variable '{variable}' not found in data")
            return None
        
        # Extract values and filter out NaN values
        values = gdf[variable].values
        if np.isnan(values).any():
            print(f"Warning: {np.isnan(values).sum()} NaN values found in '{variable}'")
            # Use only non-NaN indices for analysis
            valid_indices = ~np.isnan(values)
            values = values[valid_indices]
            # Create a subset of weights for valid indices
            weights_subset = self.weights.subset(valid_indices)
            # Create filtered GeoDataFrame
            gdf_filtered = gdf.iloc[valid_indices]
        else:
            weights_subset = self.weights
            gdf_filtered = gdf
        
        try:
            # Compute Getis-Ord G* statistic
            g_star = G_Local(values, weights_subset, permutations=999)
            
            # Create a copy of the geodataframe with results
            results_gdf = gdf_filtered.copy()
            
            # Add G* statistics to the dataframe
            results_gdf['g_star'] = g_star.Gs
            results_gdf['p_value'] = g_star.p_sim
            results_gdf['z_score'] = g_star.z_sim
            results_gdf['significant'] = g_star.p_sim < self.default_params['p_threshold']
            
            # Create hot/cold spot classification
            # Positive z-scores = hot spots, negative z-scores = cold spots
            results_gdf['hotspot_type'] = 'Not Significant'
            
            # Hot spots
            # 90% confidence: p < 0.1 and z > 0
            mask = (results_gdf['p_value'] < 0.1) & (results_gdf['z_score'] > 0)
            results_gdf.loc[mask, 'hotspot_type'] = 'Hot Spot (90% confidence)'
            
            # 95% confidence: p < 0.05 and z > 0
            mask = (results_gdf['p_value'] < 0.05) & (results_gdf['z_score'] > 0)
            results_gdf.loc[mask, 'hotspot_type'] = 'Hot Spot (95% confidence)'
            
            # 99% confidence: p < 0.01 and z > 0
            mask = (results_gdf['p_value'] < 0.01) & (results_gdf['z_score'] > 0)
            results_gdf.loc[mask, 'hotspot_type'] = 'Hot Spot (99% confidence)'
            
            # Cold spots
            # 90% confidence: p < 0.1 and z < 0
            mask = (results_gdf['p_value'] < 0.1) & (results_gdf['z_score'] < 0)
            results_gdf.loc[mask, 'hotspot_type'] = 'Cold Spot (90% confidence)'
            
            # 95% confidence: p < 0.05 and z < 0
            mask = (results_gdf['p_value'] < 0.05) & (results_gdf['z_score'] < 0)
            results_gdf.loc[mask, 'hotspot_type'] = 'Cold Spot (95% confidence)'
            
            # 99% confidence: p < 0.01 and z < 0
            mask = (results_gdf['p_value'] < 0.01) & (results_gdf['z_score'] < 0)
            results_gdf.loc[mask, 'hotspot_type'] = 'Cold Spot (99% confidence)'
            
            # Store results
            self.getis_results[variable] = {
                'g_star': g_star,  # G* statistic object
                'results_gdf': results_gdf,  # GeoDataFrame with results
                'variable': variable,  # Variable analyzed
                'weights': weights_subset  # Spatial weights matrix
            }
            
            # Count hotspots by type
            hotspot_counts = results_gdf[results_gdf['hotspot_type'] != 'Not Significant']['hotspot_type'].value_counts()
            print(f"Getis-Ord G* for {variable} identified:")
            for hotspot_type, count in hotspot_counts.items():
                print(f"  {count} {hotspot_type}")
                
            return self.getis_results[variable]
        
        except Exception as e:
            print(f"Error computing Getis-Ord G*: {e}")
            return None
    
    def run_dbscan_clustering(self, variable, eps=None, min_samples=None):
        """
        Run DBSCAN spatial clustering algorithm.
        
        Args:
            variable: Name of the variable to analyze
            eps: Distance threshold for DBSCAN (optional)
            min_samples: Minimum samples for DBSCAN (optional)
        
        Returns:
            Dictionary with DBSCAN results
        """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        if self.neighborhoods_gdf is not None:
            gdf = self.neighborhoods_gdf
            print("Using neighborhood polygons for DBSCAN clustering...")
        else:
            gdf = self.points_gdf
            print("Using point data for DBSCAN clustering...")
        
        # Check if variable exists in the dataframe
        if variable not in gdf.columns:
            print(f"Error: Variable '{variable}' not found in data")
            return None
        
        # Set default parameters if not provided
        if eps is None:
            eps = self.default_params['eps_distance']
        if min_samples is None:
            min_samples = self.default_params['min_samples']
        
        try:
            # Create a copy of the geodataframe
            clustering_df = gdf.copy()
            
            # Extract coordinates and variable value
            if isinstance(gdf.geometry.iloc[0], Point):
                # For point data, use the point coordinates
                X = np.vstack([
                    clustering_df.geometry.x,
                    clustering_df.geometry.y,
                    clustering_df[variable]
                ]).T
            else:
                # For polygon data, use centroid coordinates
                X = np.vstack([
                    clustering_df.geometry.centroid.x,
                    clustering_df.geometry.centroid.y,
                    clustering_df[variable]
                ]).T
            
            # Scale the data
            X_scaled = StandardScaler().fit_transform(X)
            
            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
            
            # Get cluster labels
            labels = db.labels_
            
            # Add cluster labels to the geodataframe
            clustering_df['dbscan_cluster'] = labels
            
            # Count clusters
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"DBSCAN clustering for {variable} identified {n_clusters} clusters and {n_noise} noise points.")
            
            # Store results
            self.dbscan_results[variable] = {
                'dbscan': db,  # DBSCAN object
                'results_gdf': clustering_df,  # GeoDataFrame with results
                'variable': variable,  # Variable analyzed
                'n_clusters': n_clusters,  # Number of clusters
                'n_noise': n_noise,  # Number of noise points
                'params': {'eps': eps, 'min_samples': min_samples}  # Parameters used
            }
            
            return self.dbscan_results[variable]
        
        except Exception as e:
            print(f"Error running DBSCAN clustering: {e}")
            return None
    
    def visualize_global_moran(self, variable):
        """
        Create visualization of Global Moran's I results.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Path to the saved visualization
        """
        if variable not in self.moran_results:
            print(f"Computing Global Moran's I for {variable} first...")
            self.compute_global_moran(variable)
        
        if variable not in self.moran_results:
            print(f"Error: No Global Moran's I results for {variable}")
            return None
        
        try:
            # Get results
            results = self.moran_results[variable]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot histogram of simulated values
            sns.histplot(results['sim'], kde=True, ax=ax)
            
            # Add lines for observed and expected values
            ax.axvline(results['I'], color='red', linestyle='--', 
                      label=f"Observed I: {results['I']:.4f}")
            ax.axvline(results['EI'], color='blue', linestyle='-', 
                      label=f"Expected I: {results['EI']:.4f}")
            
            # Add p-value and z-score to plot
            p_value = results['p_norm']
            z_score = results['z_norm']
            
            # Determine autocorrelation type
            if z_score > 0:
                autocorr_type = "Positive spatial autocorrelation"
            elif z_score < 0:
                autocorr_type = "Negative spatial autocorrelation"
            else:
                autocorr_type = "No spatial autocorrelation"
            
            # Add annotation for p-value and z-score
            ax.annotate(
                f"p-value: {p_value:.4f}\nz-score: {z_score:.4f}\n{autocorr_type}", 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                verticalalignment='top'
            )
            
            # Title and labels
            ax.set_title(f"Global Moran's I for {variable.replace('_', ' ').title()}")
            ax.set_xlabel("Moran's I value")
            ax.set_ylabel("Frequency")
            ax.legend()
            
            # Save figure
            output_path = self.output_dir / f"global_moran_{variable}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved Global Moran's I visualization to {output_path}")
            plt.close()
            
            # Create a summary file with key statistics
            summary_file = self.output_dir / f"moran_summary_{variable}.md"
            with open(summary_file, 'w') as f:
                f.write(f"# Global Moran's I Analysis for {variable.replace('_', ' ').title()}\n\n")
                f.write(f"## Spatial Autocorrelation Results\n\n")
                f.write(f"- **Moran's I**: {results['I']:.4f}\n")
                f.write(f"- **Expected I**: {results['EI']:.4f}\n")
                f.write(f"- **Z-Score**: {z_score:.4f}\n")
                f.write(f"- **P-Value**: {p_value:.4f}\n")
                f.write(f"- **Interpretation**: {autocorr_type}\n\n")
                
                if p_value < 0.05:
                    if z_score > 0:
                        f.write("### Conclusion\n\n")
                        f.write(f"There is statistically significant positive spatial autocorrelation in {variable} ")
                        f.write("across Miami-Dade neighborhoods. This means that areas with high values tend to be ")
                        f.write("near other areas with high values, and areas with low values tend to be near other ")
                        f.write("areas with low values. This clustering pattern suggests that this variable is not ")
                        f.write("randomly distributed but exhibits spatial dependence.\n")
                    else:
                        f.write("### Conclusion\n\n")
                        f.write(f"There is statistically significant negative spatial autocorrelation in {variable} ")
                        f.write("across Miami-Dade neighborhoods. This means that areas with high values tend to be ")
                        f.write("near areas with low values, and vice versa. This dispersed pattern suggests that this variable ")
                        f.write("is not randomly distributed but exhibits spatial competition or avoidance.\n")
                else:
                    f.write("### Conclusion\n\n")
                    f.write(f"There is no statistically significant spatial autocorrelation in {variable} ")
                    f.write("across Miami-Dade neighborhoods. This suggests that the values are randomly distributed in space.\n")
            
            # Create CSV summary for analysis according to user preference
            summary_dir = self.project_root / "data" / "processed" / "geospatial"
            summary_dir.mkdir(exist_ok=True, parents=True)
            
            summary_csv = summary_dir / f"summary_moran_analysis.csv"
            summary_df = pd.DataFrame({
                'variable': [variable],
                'morans_i': [results['I']],
                'expected_i': [results['EI']],
                'z_score': [z_score],
                'p_value': [p_value],
                'autocorrelation_type': [autocorr_type],
                'is_significant': [p_value < 0.05],
                'analysis_date': [pd.Timestamp.now().strftime('%Y-%m-%d')]
            })
            
            # Append to existing CSV or create new one
            if summary_csv.exists():
                existing_df = pd.read_csv(summary_csv)
                # Update if variable already exists, otherwise append
                if variable in existing_df['variable'].values:
                    existing_df.loc[existing_df['variable'] == variable] = summary_df.iloc[0]
                else:
                    existing_df = pd.concat([existing_df, summary_df], ignore_index=True)
                existing_df.to_csv(summary_csv, index=False)
            else:
                summary_df.to_csv(summary_csv, index=False)
            
            print(f"Saved summary statistics to {summary_csv}")
            
            return output_path
        
        except Exception as e:
            print(f"Error visualizing Global Moran's I: {e}")
            return None
    
    def visualize_local_moran(self, variable):
        """
        Create visualization of Local Moran's I results.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Path to the saved visualization
        """
        local_key = f"local_{variable}"
        if local_key not in self.moran_results:
            print(f"Computing Local Moran's I for {variable} first...")
            self.compute_local_moran(variable)
        
        if local_key not in self.moran_results:
            print(f"Error: No Local Moran's I results for {variable}")
            return None
        
        try:
            # Get results
            results = self.moran_results[local_key]
            gdf = results['results_gdf']
            
            # Create a folium map centered on Miami-Dade County
            miami_center = [25.7617, -80.1918]  # Latitude, Longitude of Miami
            m = folium.Map(location=miami_center, zoom_start=10, 
                        tiles='CartoDB positron', control_scale=True)
            
            # Create choropleth maps for different results
            # 1. Value
            if variable in gdf.columns:
                # Define color map for values
                value_colormap = cm.LinearColormap(
                    colors=['blue', 'white', 'red'],
                    vmin=gdf[variable].quantile(0.1),
                    vmax=gdf[variable].quantile(0.9),
                    caption=f"{variable.replace('_', ' ').title()}"
                )
                
                # Add the variable choropleth
                folium.GeoJson(
                    data=gdf.__geo_interface__,
                    name=f"{variable}",
                    style_function=lambda feature: {
                        'fillColor': value_colormap(feature['properties'][variable]) 
                                    if feature['properties'][variable] is not None else 'gray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(fields=[variable], aliases=[variable.replace('_', ' ').title()]),
                    highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.9}
                ).add_to(m)
                
                # Add the colormap to the map
                value_colormap.add_to(m)
            
            # 2. Local Moran's I Cluster Map
            if 'sig_cluster' in gdf.columns:
                # Define color map for significant clusters
                cluster_colors = {
                    'HH': '#d7191c',  # High-High (hot spot) - red
                    'HL': '#fdae61',  # High-Low (outlier) - orange
                    'LH': '#abd9e9',  # Low-High (outlier) - light blue
                    'LL': '#2c7bb6',  # Low-Low (cold spot) - dark blue
                    None: '#efefef'   # Not significant - light gray
                }
                
                # Add the cluster choropleth
                folium.GeoJson(
                    data=gdf.__geo_interface__,
                    name="LISA Clusters",
                    style_function=lambda feature: {
                        'fillColor': cluster_colors[feature['properties']['sig_cluster']],
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['cluster_type', 'local_moran', 'p_value'],
                        aliases=['Cluster Type', 'Local Moran', 'p-value'],
                        localize=True
                    ),
                    highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.9}
                ).add_to(m)
                
                # Add a legend for cluster types
                legend_html = """
                <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; padding: 10px; 
                background-color: white; border: 2px solid grey; border-radius: 5px">
                <p><b>LISA Cluster Map</b></p>
                <p><i>Significant clusters and outliers at p < {}</i></p>
                <p><!-- HH --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> High-High (Hot spot)</p>
                <p><!-- HL --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> High-Low (Outlier)</p>
                <p><!-- LH --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Low-High (Outlier)</p>
                <p><!-- LL --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Low-Low (Cold spot)</p>
                <p><!-- NS --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Not Significant</p>
                </div>
                """.format(
                    self.default_params['p_threshold'],
                    cluster_colors['HH'],
                    cluster_colors['HL'],
                    cluster_colors['LH'],
                    cluster_colors['LL'],
                    cluster_colors[None]
                )
                m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            output_path = self.output_dir / f"local_moran_{variable}.html"
            m.save(str(output_path))
            print(f"Saved Local Moran's I visualization to {output_path}")
            
            # Also create a static plot for reports
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # Variable value map
            gdf.plot(column=variable, ax=axes[0], legend=True, cmap='RdBu_r',
                   legend_kwds={'shrink': 0.8, 'label': variable.replace('_', ' ').title()})
            axes[0].set_title(f"{variable.replace('_', ' ').title()} by Neighborhood")
            axes[0].axis('off')
            
            # Significant cluster map
            if 'sig_cluster' in gdf.columns:
                # Create a categorical colormap for clusters
                cluster_cmap = {None: '#efefef'}  # Not significant - light gray
                for cluster, color in cluster_colors.items():
                    if cluster is not None:  # Skip None value as it's already defined
                        cluster_cmap[cluster] = color
                        
                # Create a custom patch for each cluster type for the legend
                patches = [mpatches.Patch(color=color, label=label) for label, color in [
                    ('High-High (Hot spot)', cluster_colors['HH']),
                    ('High-Low (Outlier)', cluster_colors['HL']),
                    ('Low-High (Outlier)', cluster_colors['LH']),
                    ('Low-Low (Cold spot)', cluster_colors['LL']),
                    ('Not Significant', cluster_colors[None])
                ]]
                
                # Plot the map with custom colors based on sig_cluster
                gdf.plot(column='sig_cluster', ax=axes[1], categorical=True,
                       legend=False, color=gdf['sig_cluster'].map(cluster_cmap))
                axes[1].set_title(f"LISA Clusters for {variable.replace('_', ' ').title()}")
                axes[1].axis('off')
                
                # Add the legend
                axes[1].legend(handles=patches, loc='lower right')
            
            # Save the static plot
            static_path = self.output_dir / f"local_moran_{variable}_static.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"Saved static visualization to {static_path}")
            plt.close()
            
            # Create a summary file with key statistics
            cluster_summary = gdf['sig_cluster'].value_counts().to_dict()
            # Replace None with 'Not Significant' in the summary
            if None in cluster_summary:
                cluster_summary['Not Significant'] = cluster_summary.pop(None)
            
            summary_file = self.output_dir / f"local_moran_summary_{variable}.md"
            with open(summary_file, 'w') as f:
                f.write(f"# Local Moran's I Analysis for {variable.replace('_', ' ').title()}\n\n")
                f.write(f"## Spatial Cluster Analysis Results\n\n")
                f.write(f"### Significant Clusters and Outliers (p < {self.default_params['p_threshold']})\n\n")
                
                # Write cluster counts
                for cluster_type, count in cluster_summary.items():
                    if cluster_type == 'HH':
                        f.write(f"- **High-High (Hot spots)**: {count} neighborhoods\n")
                    elif cluster_type == 'LL':
                        f.write(f"- **Low-Low (Cold spots)**: {count} neighborhoods\n")
                    elif cluster_type == 'HL':
                        f.write(f"- **High-Low (Spatial outliers)**: {count} neighborhoods\n")
                    elif cluster_type == 'LH':
                        f.write(f"- **Low-High (Spatial outliers)**: {count} neighborhoods\n")
                    elif cluster_type == 'Not Significant':
                        f.write(f"- **Not Significant**: {count} neighborhoods\n")
                
                # Write interpretation
                f.write("\n### Interpretation\n\n")
                f.write(f"**High-High clusters (Hot spots)**: Areas with high {variable} values surrounded by high values.\n")
                f.write(f"**Low-Low clusters (Cold spots)**: Areas with low {variable} values surrounded by low values.\n")
                f.write(f"**High-Low outliers**: Areas with high {variable} values surrounded by low values.\n")
                f.write(f"**Low-High outliers**: Areas with low {variable} values surrounded by high values.\n\n")
                
                # Write conclusion based on results
                f.write("### Conclusion\n\n")
                significant_count = sum(count for cluster_type, count in cluster_summary.items() 
                                    if cluster_type != 'Not Significant')
                total_count = sum(cluster_summary.values())
                significant_pct = (significant_count / total_count) * 100 if total_count > 0 else 0
                
                if significant_pct > 50:
                    f.write(f"The analysis reveals strong spatial patterns in {variable} across Miami-Dade neighborhoods. ")
                elif significant_pct > 25:
                    f.write(f"The analysis reveals moderate spatial patterns in {variable} across Miami-Dade neighborhoods. ")
                elif significant_pct > 10:
                    f.write(f"The analysis reveals some spatial patterns in {variable} across Miami-Dade neighborhoods. ")
                else:
                    f.write(f"The analysis reveals limited spatial patterns in {variable} across Miami-Dade neighborhoods. ")
                
                # Add more specific interpretation based on most common cluster type
                most_common_type = max((count, cluster_type) for cluster_type, count in cluster_summary.items() 
                                     if cluster_type != 'Not Significant')[1] if significant_count > 0 else None
                
                if most_common_type == 'HH':
                    f.write(f"The dominant pattern is clustering of high values, indicating areas where {variable} is consistently high ")
                    f.write("and potentially creating concentrated effects or opportunities.")
                elif most_common_type == 'LL':
                    f.write(f"The dominant pattern is clustering of low values, indicating areas where {variable} is consistently low ")
                    f.write("and potentially creating concentrated challenges or market opportunities.")
                elif most_common_type in ['HL', 'LH']:
                    f.write(f"The dominant pattern is spatial outliers, indicating a heterogeneous landscape for {variable} ")
                    f.write("with significant local variations that break regional trends.")
                elif significant_count > 0:
                    f.write(f"The pattern shows a mix of different cluster types, indicating complex spatial dynamics for {variable}.")
                else:
                    f.write(f"Most areas do not show significant spatial clustering, suggesting that {variable} is relatively ")
                    f.write("randomly distributed across the region.")
            
            # Create CSV summary for analysis according to user preference
            summary_dir = self.project_root / "data" / "processed" / "geospatial"
            summary_dir.mkdir(exist_ok=True, parents=True)
            
            summary_csv = summary_dir / f"summary_local_moran_analysis.csv"
            
            # Convert cluster_summary to flat structure for DataFrame
            flat_summary = {
                'variable': variable,
                'total_neighborhoods': total_count,
                'significant_neighborhoods': significant_count,
                'significant_percent': significant_pct,
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
            # Add counts for each cluster type
            for cluster_type in ['HH', 'LL', 'HL', 'LH', 'Not Significant']:
                count = cluster_summary.get(cluster_type, 0)
                flat_summary[f'{cluster_type}_count'] = count
                flat_summary[f'{cluster_type}_percent'] = (count / total_count) * 100 if total_count > 0 else 0
            
            summary_df = pd.DataFrame([flat_summary])
            
            # Append to existing CSV or create new one
            if summary_csv.exists():
                existing_df = pd.read_csv(summary_csv)
                # Update if variable already exists, otherwise append
                if variable in existing_df['variable'].values:
                    existing_df.loc[existing_df['variable'] == variable] = summary_df.iloc[0]
                else:
                    existing_df = pd.concat([existing_df, summary_df], ignore_index=True)
                existing_df.to_csv(summary_csv, index=False)
            else:
                summary_df.to_csv(summary_csv, index=False)
            
            print(f"Saved summary statistics to {summary_csv}")
            
            return output_path
        
        except Exception as e:
            print(f"Error visualizing Local Moran's I: {e}")
            traceback.print_exc()
            return None
            
    def visualize_getis_ord(self, variable):
        """
        Create visualization of Getis-Ord G* results.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Path to the saved visualization
        """
        if variable not in self.getis_results:
            print(f"Computing Getis-Ord G* for {variable} first...")
            self.compute_getis_ord(variable)
        
        if variable not in self.getis_results:
            print(f"Error: No Getis-Ord G* results for {variable}")
            return None
        
        try:
            # Get results
            results = self.getis_results[variable]
            gdf = results['results_gdf']
            
            # Create a folium map centered on Miami-Dade County
            miami_center = [25.7617, -80.1918]  # Latitude, Longitude of Miami
            m = folium.Map(location=miami_center, zoom_start=10, 
                        tiles='CartoDB positron', control_scale=True)
            
            # Create choropleth maps for different results
            # 1. Value
            if variable in gdf.columns:
                # Define color map for values
                value_colormap = cm.LinearColormap(
                    colors=['blue', 'white', 'red'],
                    vmin=gdf[variable].quantile(0.1),
                    vmax=gdf[variable].quantile(0.9),
                    caption=f"{variable.replace('_', ' ').title()}"
                )
                
                # Add the variable choropleth
                folium.GeoJson(
                    data=gdf.__geo_interface__,
                    name=f"{variable}",
                    style_function=lambda feature: {
                        'fillColor': value_colormap(feature['properties'][variable]) 
                                    if feature['properties'][variable] is not None else 'gray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(fields=[variable], aliases=[variable.replace('_', ' ').title()]),
                    highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.9}
                ).add_to(m)
                
                # Add the colormap to the map
                value_colormap.add_to(m)
            
            # 2. Getis-Ord G* Hot/Cold Spot Map
            if 'hotspot_type' in gdf.columns:
                # Define color map for hotspots
                hotspot_colors = {
                    'Hot Spot (99% confidence)': '#d7191c',  # Very hot - dark red
                    'Hot Spot (95% confidence)': '#fdae61',  # Hot - orange
                    'Hot Spot (90% confidence)': '#fed976',  # Somewhat hot - yellow
                    'Not Significant': '#efefef',           # Not significant - light gray
                    'Cold Spot (90% confidence)': '#abd9e9', # Somewhat cold - light blue
                    'Cold Spot (95% confidence)': '#74add1', # Cold - medium blue
                    'Cold Spot (99% confidence)': '#2c7bb6'  # Very cold - dark blue
                }
                
                # Add the hotspot choropleth
                folium.GeoJson(
                    data=gdf.__geo_interface__,
                    name="G* Hot Spots",
                    style_function=lambda feature: {
                        'fillColor': hotspot_colors[feature['properties']['hotspot_type']],
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['hotspot_type', 'g_star', 'z_score', 'p_value'],
                        aliases=['Hotspot Type', 'G* Statistic', 'Z-Score', 'p-value'],
                        localize=True
                    ),
                    highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.9}
                ).add_to(m)
                
                # Add a legend for hotspot types
                legend_html = """
                <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; padding: 10px; 
                background-color: white; border: 2px solid grey; border-radius: 5px">
                <p><b>Getis-Ord G* Hot Spots</b></p>
                <p><!-- Hot 99 --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Hot Spot (99% confidence)</p>
                <p><!-- Hot 95 --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Hot Spot (95% confidence)</p>
                <p><!-- Hot 90 --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Hot Spot (90% confidence)</p>
                <p><!-- Not Sig --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Not Significant</p>
                <p><!-- Cold 90 --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Cold Spot (90% confidence)</p>
                <p><!-- Cold 95 --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Cold Spot (95% confidence)</p>
                <p><!-- Cold 99 --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Cold Spot (99% confidence)</p>
                </div>
                """.format(
                    hotspot_colors['Hot Spot (99% confidence)'],
                    hotspot_colors['Hot Spot (95% confidence)'],
                    hotspot_colors['Hot Spot (90% confidence)'],
                    hotspot_colors['Not Significant'],
                    hotspot_colors['Cold Spot (90% confidence)'],
                    hotspot_colors['Cold Spot (95% confidence)'],
                    hotspot_colors['Cold Spot (99% confidence)']
                )
                m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            output_path = self.output_dir / f"getis_ord_{variable}.html"
            m.save(str(output_path))
            print(f"Saved Getis-Ord G* visualization to {output_path}")
            
            # Also create a static plot for reports
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # Variable value map
            gdf.plot(column=variable, ax=axes[0], legend=True, cmap='RdBu_r',
                   legend_kwds={'shrink': 0.8, 'label': variable.replace('_', ' ').title()})
            axes[0].set_title(f"{variable.replace('_', ' ').title()} by Neighborhood")
            axes[0].axis('off')
            
            # Hotspot map
            if 'hotspot_type' in gdf.columns:
                # Create a categorical colormap for hotspots
                hotspot_cmap = {}
                for hotspot, color in hotspot_colors.items():
                    hotspot_cmap[hotspot] = color
                        
                # Create a custom patch for each hotspot type for the legend
                patches = [mpatches.Patch(color=color, label=label) for label, color in [
                    ('Hot Spot (99% confidence)', hotspot_colors['Hot Spot (99% confidence)']),
                    ('Hot Spot (95% confidence)', hotspot_colors['Hot Spot (95% confidence)']),
                    ('Hot Spot (90% confidence)', hotspot_colors['Hot Spot (90% confidence)']),
                    ('Not Significant', hotspot_colors['Not Significant']),
                    ('Cold Spot (90% confidence)', hotspot_colors['Cold Spot (90% confidence)']),
                    ('Cold Spot (95% confidence)', hotspot_colors['Cold Spot (95% confidence)']),
                    ('Cold Spot (99% confidence)', hotspot_colors['Cold Spot (99% confidence)'])
                ]]
                
                # Plot the map with custom colors based on hotspot_type
                gdf.plot(column='hotspot_type', ax=axes[1], categorical=True,
                       legend=False, color=gdf['hotspot_type'].map(hotspot_cmap))
                axes[1].set_title(f"Getis-Ord G* Hot/Cold Spots for {variable.replace('_', ' ').title()}")
                axes[1].axis('off')
                
                # Add the legend
                axes[1].legend(handles=patches, loc='lower right', fontsize='small')
            
            # Save the static plot
            static_path = self.output_dir / f"getis_ord_{variable}_static.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"Saved static visualization to {static_path}")
            plt.close()
            
            # Create a summary file with key statistics
            hotspot_summary = gdf['hotspot_type'].value_counts().to_dict()
            
            summary_file = self.output_dir / f"getis_ord_summary_{variable}.md"
            with open(summary_file, 'w') as f:
                f.write(f"# Getis-Ord G* Analysis for {variable.replace('_', ' ').title()}\n\n")
                f.write(f"## Hot and Cold Spot Analysis Results\n\n")
                
                # Write hotspot counts
                f.write("### Hot and Cold Spots by Confidence Level\n\n")
                for hotspot_type, count in hotspot_summary.items():
                    f.write(f"- **{hotspot_type}**: {count} neighborhoods\n")
                
                # Write interpretation
                f.write("\n### Interpretation\n\n")
                f.write(f"**Hot Spots**: Areas with statistically significant high {variable} values.\n")
                f.write(f"**Cold Spots**: Areas with statistically significant low {variable} values.\n")
                f.write(f"**Confidence Level**: Higher confidence (99% > 95% > 90%) indicates stronger evidence of clustering.\n\n")
                
                # Calculate hot and cold spot totals
                hot_spots = sum(count for hotspot_type, count in hotspot_summary.items() 
                              if 'Hot Spot' in hotspot_type)
                cold_spots = sum(count for hotspot_type, count in hotspot_summary.items() 
                               if 'Cold Spot' in hotspot_type)
                not_sig = hotspot_summary.get('Not Significant', 0)
                total_count = sum(hotspot_summary.values())
                
                # Write conclusion based on results
                f.write("### Conclusion\n\n")
                significant_count = hot_spots + cold_spots
                significant_pct = (significant_count / total_count) * 100 if total_count > 0 else 0
                
                f.write(f"This analysis identified {hot_spots} hot spots and {cold_spots} cold spots ")
                f.write(f"out of {total_count} total neighborhoods ({significant_pct:.1f}% significant). ")
                
                # More detailed interpretation
                if hot_spots > cold_spots * 2:
                    f.write(f"Hot spots dominate the landscape, suggesting strong clustering of high {variable} values ")
                    f.write("across specific regions of Miami-Dade County.")
                elif cold_spots > hot_spots * 2:
                    f.write(f"Cold spots dominate the landscape, suggesting strong clustering of low {variable} values ")
                    f.write("across specific regions of Miami-Dade County.")
                elif hot_spots > 0 and cold_spots > 0:
                    f.write(f"Both hot spots and cold spots are present, indicating a polarized pattern for {variable} ")
                    f.write("with distinct regions of high and low values.")
                elif significant_count == 0:
                    f.write(f"No significant hot or cold spots were detected, suggesting that {variable} values ")
                    f.write("are relatively randomly distributed across Miami-Dade County.")
            
            # Create CSV summary for analysis according to user preference
            summary_dir = self.project_root / "data" / "processed" / "geospatial"
            summary_dir.mkdir(exist_ok=True, parents=True)
            
            summary_csv = summary_dir / f"summary_getis_ord_analysis.csv"
            
            # Create summary data for DataFrame
            summary_data = {
                'variable': variable,
                'total_neighborhoods': total_count,
                'hot_spots': hot_spots,
                'cold_spots': cold_spots,
                'not_significant': not_sig,
                'significant_percent': significant_pct,
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
            # Add counts for each hotspot type
            for hotspot_type in ['Hot Spot (99% confidence)', 'Hot Spot (95% confidence)', 'Hot Spot (90% confidence)',
                              'Not Significant', 'Cold Spot (90% confidence)', 'Cold Spot (95% confidence)', 
                              'Cold Spot (99% confidence)']:
                count = hotspot_summary.get(hotspot_type, 0)
                summary_data[hotspot_type.replace(' ', '_')] = count
            
            summary_df = pd.DataFrame([summary_data])
            
            # Append to existing CSV or create new one
            if summary_csv.exists():
                existing_df = pd.read_csv(summary_csv)
                # Update if variable already exists, otherwise append
                if variable in existing_df['variable'].values:
                    existing_df.loc[existing_df['variable'] == variable] = summary_df.iloc[0]
                else:
                    existing_df = pd.concat([existing_df, summary_df], ignore_index=True)
                existing_df.to_csv(summary_csv, index=False)
            else:
                summary_df.to_csv(summary_csv, index=False)
            
            print(f"Saved summary statistics to {summary_csv}")
            
            return output_path
        
        except Exception as e:
            print(f"Error visualizing Getis-Ord G*: {e}")
            traceback.print_exc()
            return None
    
    def visualize_dbscan(self, variable):
        """
        Create visualization of DBSCAN clustering results.
        
        Args:
            variable: Name of the variable to analyze
        
        Returns:
            Path to the saved visualization
        """
        if variable not in self.dbscan_results:
            print(f"Running DBSCAN clustering for {variable} first...")
            self.run_dbscan_clustering(variable)
        
        if variable not in self.dbscan_results:
            print(f"Error: No DBSCAN results for {variable}")
            return None
        
        try:
            # Get results
            results = self.dbscan_results[variable]
            gdf = results['results_gdf']
            
            # Create a folium map centered on Miami-Dade County
            miami_center = [25.7617, -80.1918]  # Latitude, Longitude of Miami
            m = folium.Map(location=miami_center, zoom_start=10, 
                        tiles='CartoDB positron', control_scale=True)
            
            # Create choropleth maps for different results
            # 1. Value
            if variable in gdf.columns:
                # Define color map for values
                value_colormap = cm.LinearColormap(
                    colors=['blue', 'white', 'red'],
                    vmin=gdf[variable].quantile(0.1),
                    vmax=gdf[variable].quantile(0.9),
                    caption=f"{variable.replace('_', ' ').title()}"
                )
                
                # Add the variable choropleth
                folium.GeoJson(
                    data=gdf.__geo_interface__,
                    name=f"{variable}",
                    style_function=lambda feature: {
                        'fillColor': value_colormap(feature['properties'][variable]) 
                                    if feature['properties'][variable] is not None else 'gray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(fields=[variable], aliases=[variable.replace('_', ' ').title()]),
                    highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.9}
                ).add_to(m)
                
                # Add the colormap to the map
                value_colormap.add_to(m)
            
            # 2. DBSCAN Cluster Map
            if 'dbscan_cluster' in gdf.columns:
                # Generate a colormap for clusters
                n_clusters = results['n_clusters']
                cluster_colors = {}
                
                # Noise points are black
                cluster_colors[-1] = '#000000'
                
                # Generate colors for clusters
                if n_clusters > 0:
                    cmap = plt.cm.get_cmap('tab20', n_clusters)
                    for i in range(n_clusters):
                        cluster_colors[i] = plt.matplotlib.colors.rgb2hex(cmap(i))
                
                # Add the cluster choropleth
                folium.GeoJson(
                    data=gdf.__geo_interface__,
                    name="DBSCAN Clusters",
                    style_function=lambda feature: {
                        'fillColor': cluster_colors.get(feature['properties']['dbscan_cluster'], '#cccccc'),
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7 if feature['properties']['dbscan_cluster'] != -1 else 0.3
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['dbscan_cluster', variable],
                        aliases=['Cluster', variable.replace('_', ' ').title()],
                        localize=True
                    ),
                    highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.9}
                ).add_to(m)
                
                # Add a legend for cluster types
                legend_html = """
                <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; padding: 10px; 
                background-color: white; border: 2px solid grey; border-radius: 5px">
                <p><b>DBSCAN Clustering</b></p>
                <p><i>Parameters: eps={}, min_samples={}</i></p>
                <p><!-- Noise --><span style="background-color: {}; opacity: 0.3; display: inline-block; 
                width: 20px; height: 20px; margin-right: 5px;"></span> Noise Points ({} points)</p>
                """.format(
                    results['params']['eps'],
                    results['params']['min_samples'],
                    cluster_colors[-1],
                    results['n_noise']
                )
                
                # Add a legend entry for each cluster
                for i in range(n_clusters):
                    count = (gdf['dbscan_cluster'] == i).sum()
                    legend_html += """
                    <p><!-- Cluster {} --><span style="background-color: {}; opacity: 0.7; display: inline-block; 
                    width: 20px; height: 20px; margin-right: 5px;"></span> Cluster {} ({} points)</p>
                    """.format(i, cluster_colors[i], i, count)
                
                legend_html += "</div>"
                m.get_root().html.add_child(folium.Element(legend_html))
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Save map
            output_path = self.output_dir / f"dbscan_{variable}.html"
            m.save(str(output_path))
            print(f"Saved DBSCAN visualization to {output_path}")
            
            # Also create a static plot for reports
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))
            
            # Variable value map
            gdf.plot(column=variable, ax=axes[0], legend=True, cmap='RdBu_r',
                   legend_kwds={'shrink': 0.8, 'label': variable.replace('_', ' ').title()})
            axes[0].set_title(f"{variable.replace('_', ' ').title()} by Neighborhood")
            axes[0].axis('off')
            
            # Cluster map
            if 'dbscan_cluster' in gdf.columns:
                # Plot the map with custom colors based on dbscan_cluster
                gdf.plot(column='dbscan_cluster', ax=axes[1], categorical=True,
                       legend=True, cmap='tab20',
                       legend_kwds={'title': 'DBSCAN Cluster'})
                axes[1].set_title(f"DBSCAN Clusters for {variable.replace('_', ' ').title()}")
                axes[1].axis('off')
            
            # Save the static plot
            static_path = self.output_dir / f"dbscan_{variable}_static.png"
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            print(f"Saved static visualization to {static_path}")
            plt.close()
            
            # Create a summary file with key statistics
            summary_file = self.output_dir / f"dbscan_summary_{variable}.md"
            with open(summary_file, 'w') as f:
                f.write(f"# DBSCAN Clustering Analysis for {variable.replace('_', ' ').title()}\n\n")
                f.write(f"## Clustering Results\n\n")
                f.write(f"### Parameters:\n")
                f.write(f"- **Epsilon (eps)**: {results['params']['eps']}\n")
                f.write(f"- **Minimum Samples**: {results['params']['min_samples']}\n\n")
                
                f.write(f"### Results:\n")
                f.write(f"- **Number of Clusters**: {results['n_clusters']}\n")
                f.write(f"- **Number of Noise Points**: {results['n_noise']}\n")
                
                # Add counts for each cluster
                cluster_counts = gdf['dbscan_cluster'].value_counts().sort_index()
                f.write("\n### Cluster Distribution:\n")
                for cluster, count in cluster_counts.items():
                    if cluster == -1:
                        f.write(f"- **Noise Points**: {count} neighborhoods\n")
                    else:
                        f.write(f"- **Cluster {cluster}**: {count} neighborhoods\n")
                
                # Write interpretation
                f.write("\n### Interpretation:\n\n")
                f.write(f"DBSCAN clustering identifies dense regions of neighborhoods with similar {variable} values, ")
                f.write("separated by regions of lower density. Unlike other clustering methods, DBSCAN does not require ")
                f.write("specifying the number of clusters in advance and can identify clusters of arbitrary shape.\n\n")
                
                f.write("**Clusters**: Groups of neighborhoods with similar characteristics that are densely packed together.\n")
                f.write("**Noise Points**: Neighborhoods that don't belong to any cluster, often representing outliers or areas in transition.\n\n")
                
                # Write conclusion based on results
                f.write("### Conclusion:\n\n")
                if results['n_clusters'] > 0:
                    f.write(f"The DBSCAN analysis identified {results['n_clusters']} distinct clusters of neighborhoods ")
                    f.write(f"based on their {variable} values and spatial proximity. This suggests that there are ")
                    f.write(f"clear geographical patterns in how {variable} is distributed across Miami-Dade County.\n\n")
                    
                    if results['n_noise'] > 0:
                        noise_pct = (results['n_noise'] / (results['n_noise'] + gdf.shape[0])) * 100
                        f.write(f"Additionally, {results['n_noise']} neighborhoods ({noise_pct:.1f}%) were classified as noise points, ")
                        f.write("indicating they are spatial outliers that don't fit into any clear cluster pattern. ")
                        f.write("These areas may warrant individual attention in further analysis or targeted policies.")
                    else:
                        f.write("All neighborhoods were successfully assigned to a cluster, suggesting a very strong ")
                        f.write("spatial structure with no significant outliers.")
                else:
                    f.write(f"The DBSCAN analysis did not identify any significant clusters with the current parameters. ")
                    f.write(f"This suggests that {variable} may be relatively uniformly distributed across neighborhoods, ")
                    f.write(f"or that the clustering parameters need to be adjusted to better capture the underlying patterns.")
            
            # Create CSV summary
            summary_dir = self.project_root / "data" / "processed" / "geospatial"
            summary_dir.mkdir(exist_ok=True, parents=True)
            
            summary_csv = summary_dir / f"summary_dbscan_analysis.csv"
            
            # Create cluster size data
            cluster_sizes = gdf['dbscan_cluster'].value_counts().to_dict()
            
            # Create summary data for DataFrame
            summary_data = {
                'variable': variable,
                'n_clusters': results['n_clusters'],
                'n_noise': results['n_noise'],
                'eps': results['params']['eps'],
                'min_samples': results['params']['min_samples'],
                'noise_percent': (results['n_noise'] / gdf.shape[0]) * 100 if gdf.shape[0] > 0 else 0,
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            }
            
            # Add counts for each cluster (up to 10 clusters)
            for i in range(10):
                count = cluster_sizes.get(i, 0) if i < results['n_clusters'] else 0
                summary_data[f'cluster_{i}_size'] = count
            
            summary_df = pd.DataFrame([summary_data])
            
            # Append to existing CSV or create new one
            if summary_csv.exists():
                existing_df = pd.read_csv(summary_csv)
                # Check for existing entry with same variable and parameters
                mask = ((existing_df['variable'] == variable) & 
                       (existing_df['eps'] == results['params']['eps']) & 
                       (existing_df['min_samples'] == results['params']['min_samples']))
                if mask.any():
                    existing_df.loc[mask] = summary_df.iloc[0]
                else:
                    existing_df = pd.concat([existing_df, summary_df], ignore_index=True)
                existing_df.to_csv(summary_csv, index=False)
            else:
                summary_df.to_csv(summary_csv, index=False)
            
            print(f"Saved summary statistics to {summary_csv}")
            
            return output_path
        
        except Exception as e:
            print(f"Error visualizing DBSCAN clusters: {e}")
            traceback.print_exc()
            return None
    
    def run_full_analysis(self, variables=None, run_dbscan=True):
        """
        Run the complete geospatial hotspot analysis pipeline on specified variables.
        
        Args:
            variables: List of variables to analyze (if None, uses built-in default variables)
            run_dbscan: Whether to run DBSCAN clustering (default: True)
        
        Returns:
            Dictionary with analysis results and paths to visualizations
        """
        # If no variables provided, use some common housing-related variables
        if variables is None:
            print("No variables specified, using default housing-related variables")
            variables = self.default_params['default_variables']
        
        # Check if we have data
        if not self.load_data():
            print("Error: Failed to load data for analysis")
            return None
        
        # Create spatial weights matrix
        if not self.create_spatial_weights():
            print("Error: Failed to create spatial weights")
            return None
        
        results = {
            'global_moran': {},
            'local_moran': {},
            'getis_ord': {},
            'dbscan': {} if run_dbscan else None
        }
        
        # Create main report folder
        report_dir = self.output_dir / "geospatial_report"
        report_dir.mkdir(exist_ok=True, parents=True)
        
        # Main report file
        report_file = report_dir / "geospatial_hotspot_analysis_report.md"
        
        # Starting the report
        with open(report_file, 'w') as f:
            f.write("# Geospatial Hotspot Analysis Report\n\n")
            f.write("## Miami-Dade County Housing Market Spatial Analysis\n\n")
            f.write("This report provides a comprehensive analysis of spatial patterns ")
            f.write("in housing and Airbnb data across Miami-Dade County neighborhoods. ")
            f.write("The analysis employs spatial autocorrelation techniques and clustering ")
            f.write("algorithms to identify hotspots, coldspots, and spatial clusters.\n\n")
            
            f.write("### Analysis Methods\n\n")
            f.write("1. **Global Moran's I**: Measures overall spatial autocorrelation to determine if values are clustered, dispersed, or random.\n")
            f.write("2. **Local Moran's I (LISA)**: Identifies local clusters and spatial outliers.\n")
            f.write("3. **Getis-Ord G***: Identifies statistically significant hot spots and cold spots.\n")
            if run_dbscan:
                f.write("4. **DBSCAN Clustering**: Groups neighborhoods based on their attributes and spatial proximity.\n\n")
            
            f.write("### Variables Analyzed\n\n")
            for variable in variables:
                f.write(f"- **{variable.replace('_', ' ').title()}**\n")
            
            f.write("\n## Summary of Findings\n\n")
        
        # Analyze each variable
        for variable in variables:
            print(f"\n{'='*80}\nAnalyzing variable: {variable}\n{'='*80}")
            
            with open(report_file, 'a') as f:
                f.write(f"\n## Analysis of {variable.replace('_', ' ').title()}\n\n")
            
            # 1. Global Moran's I
            print(f"\nComputing Global Moran's I for {variable}...")
            global_moran_result = self.compute_global_moran(variable)
            if global_moran_result:
                global_moran_viz = self.visualize_global_moran(variable)
                results['global_moran'][variable] = {
                    'result': global_moran_result,
                    'visualization': global_moran_viz
                }
                
                with open(report_file, 'a') as f:
                    f.write(f"### Global Spatial Autocorrelation\n\n")
                    f.write(f"Moran's I statistic: **{global_moran_result['I']:.4f}** (p-value: {global_moran_result['p_norm']:.4f})\n\n")
                    
                    if global_moran_result['p_norm'] < 0.05:
                        if global_moran_result['I'] > 0:
                            f.write(f"There is statistically significant positive spatial autocorrelation in {variable}. ")
                            f.write("This indicates that similar values tend to cluster together spatially.\n\n")
                        else:
                            f.write(f"There is statistically significant negative spatial autocorrelation in {variable}. ")
                            f.write("This indicates that dissimilar values tend to be located near each other.\n\n")
                    else:
                        f.write(f"There is no statistically significant spatial autocorrelation in {variable}. ")
                        f.write("The values appear to be randomly distributed in space.\n\n")
                    
                    # Add visualization to the report
                    if global_moran_viz:
                        rel_path = os.path.relpath(global_moran_viz, report_dir.parent)
                        f.write(f"![Global Moran's I for {variable}]({rel_path})\n\n")
            
            # 2. Local Moran's I
            print(f"\nComputing Local Moran's I for {variable}...")
            local_moran_result = self.compute_local_moran(variable)
            if local_moran_result:
                local_moran_viz = self.visualize_local_moran(variable)
                results['local_moran'][variable] = {
                    'result': local_moran_result,
                    'visualization': local_moran_viz
                }
                
                with open(report_file, 'a') as f:
                    f.write(f"### Local Spatial Clusters and Outliers\n\n")
                    
                    # Count significant clusters by type
                    gdf = local_moran_result['results_gdf']
                    cluster_summary = gdf['sig_cluster'].value_counts().to_dict()
                    # Replace None with 'Not Significant' in the summary
                    if None in cluster_summary:
                        cluster_summary['Not Significant'] = cluster_summary.pop(None)
                    
                    f.write("Local Moran's I identified:\n\n")
                    for cluster_type, count in cluster_summary.items():
                        if cluster_type == 'HH':
                            f.write(f"- **{count}** High-High clusters (Hot spots)\n")
                        elif cluster_type == 'LL':
                            f.write(f"- **{count}** Low-Low clusters (Cold spots)\n")
                        elif cluster_type == 'HL':
                            f.write(f"- **{count}** High-Low spatial outliers\n")
                        elif cluster_type == 'LH':
                            f.write(f"- **{count}** Low-High spatial outliers\n")
                        elif cluster_type == 'Not Significant':
                            f.write(f"- **{count}** Not significant areas\n")
                    
                    f.write("\n")
                    
                    # Add static visualization to the report
                    static_viz_path = self.output_dir / f"local_moran_{variable}_static.png"
                    if static_viz_path.exists():
                        rel_path = os.path.relpath(static_viz_path, report_dir.parent)
                        f.write(f"![Local Moran's I Clusters for {variable}]({rel_path})\n\n")
                    
                    # Add link to interactive map
                    if local_moran_viz:
                        rel_path = os.path.relpath(local_moran_viz, report_dir.parent)
                        f.write(f"[View Interactive LISA Cluster Map for {variable}]({rel_path})\n\n")
            
            # 3. Getis-Ord G*
            print(f"\nComputing Getis-Ord G* for {variable}...")
            getis_result = self.compute_getis_ord(variable)
            if getis_result:
                getis_viz = self.visualize_getis_ord(variable)
                results['getis_ord'][variable] = {
                    'result': getis_result,
                    'visualization': getis_viz
                }
                
                with open(report_file, 'a') as f:
                    f.write(f"### Hot and Cold Spots\n\n")
                    
                    # Count hotspots by type
                    gdf = getis_result['results_gdf']
                    hotspot_summary = gdf['hotspot_type'].value_counts().to_dict()
                    
                    # Calculate hot and cold spot totals
                    hot_spots = sum(count for hotspot_type, count in hotspot_summary.items() 
                                  if 'Hot Spot' in hotspot_type)
                    cold_spots = sum(count for hotspot_type, count in hotspot_summary.items() 
                                   if 'Cold Spot' in hotspot_type)
                    not_sig = hotspot_summary.get('Not Significant', 0)
                    
                    f.write(f"Getis-Ord G* analysis identified **{hot_spots}** hot spots and **{cold_spots}** cold spots ")
                    f.write(f"for {variable}.\n\n")
                    
                    f.write("Breakdown by confidence level:\n\n")
                    for hotspot_type in ['Hot Spot (99% confidence)', 'Hot Spot (95% confidence)', 'Hot Spot (90% confidence)',
                                      'Cold Spot (90% confidence)', 'Cold Spot (95% confidence)', 'Cold Spot (99% confidence)']:
                        if hotspot_type in hotspot_summary:
                            f.write(f"- **{hotspot_summary[hotspot_type]}** {hotspot_type}\n")
                    
                    f.write(f"- **{not_sig}** Not significant areas\n\n")
                    
                    # Add static visualization to the report
                    static_viz_path = self.output_dir / f"getis_ord_{variable}_static.png"
                    if static_viz_path.exists():
                        rel_path = os.path.relpath(static_viz_path, report_dir.parent)
                        f.write(f"![Getis-Ord G* Hot/Cold Spots for {variable}]({rel_path})\n\n")
                    
                    # Add link to interactive map
                    if getis_viz:
                        rel_path = os.path.relpath(getis_viz, report_dir.parent)
                        f.write(f"[View Interactive Hot/Cold Spot Map for {variable}]({rel_path})\n\n")
            
            # 4. DBSCAN Clustering (optional)
            if run_dbscan:
                print(f"\nRunning DBSCAN clustering for {variable}...")
                dbscan_result = self.run_dbscan_clustering(variable)
                if dbscan_result:
                    dbscan_viz = self.visualize_dbscan(variable)
                    results['dbscan'][variable] = {
                        'result': dbscan_result,
                        'visualization': dbscan_viz
                    }
                    
                    with open(report_file, 'a') as f:
                        f.write(f"### Spatial Clustering\n\n")
                        f.write(f"DBSCAN clustering identified **{dbscan_result['n_clusters']}** distinct clusters ")
                        f.write(f"and **{dbscan_result['n_noise']}** noise points for {variable}.\n\n")
                        
                        if dbscan_result['n_clusters'] > 0:
                            # Add cluster sizes
                            gdf = dbscan_result['results_gdf']
                            cluster_sizes = gdf[gdf['dbscan_cluster'] >= 0]['dbscan_cluster'].value_counts().sort_index()
                            
                            f.write("Cluster sizes:\n\n")
                            for cluster, count in cluster_sizes.items():
                                f.write(f"- Cluster {cluster}: **{count}** neighborhoods\n")
                            
                            f.write("\n")
                        
                        # Add static visualization to the report
                        static_viz_path = self.output_dir / f"dbscan_{variable}_static.png"
                        if static_viz_path.exists():
                            rel_path = os.path.relpath(static_viz_path, report_dir.parent)
                            f.write(f"![DBSCAN Clusters for {variable}]({rel_path})\n\n")
                        
                        # Add link to interactive map
                        if dbscan_viz:
                            rel_path = os.path.relpath(dbscan_viz, report_dir.parent)
                            f.write(f"[View Interactive DBSCAN Cluster Map for {variable}]({rel_path})\n\n")
        
        # Add overall conclusions to the report
        with open(report_file, 'a') as f:
            f.write("## Overall Conclusions\n\n")
            f.write("This geospatial analysis reveals several important patterns in the Miami-Dade housing market:\n\n")
            
            # Generate conclusions based on results
            significant_autocorrelation = []
            hotspot_areas = []
            coldspot_areas = []
            
            for variable in variables:
                if variable in results['global_moran'] and results['global_moran'][variable]['result']['p_norm'] < 0.05:
                    if results['global_moran'][variable]['result']['I'] > 0:
                        significant_autocorrelation.append(f"{variable.replace('_', ' ').title()} (positive)")
                    else:
                        significant_autocorrelation.append(f"{variable.replace('_', ' ').title()} (negative)")
                
                if variable in results['getis_ord']:
                    gdf = results['getis_ord'][variable]['result']['results_gdf']
                    # Count high confidence hot spots (95% and 99%)
                    high_conf_hot = gdf['hotspot_type'].isin(['Hot Spot (99% confidence)', 'Hot Spot (95% confidence)']).sum()
                    high_conf_cold = gdf['hotspot_type'].isin(['Cold Spot (99% confidence)', 'Cold Spot (95% confidence)']).sum()
                    
                    if high_conf_hot > 0:
                        hotspot_areas.append(f"{variable.replace('_', ' ').title()} ({high_conf_hot} areas)")
                    
                    if high_conf_cold > 0:
                        coldspot_areas.append(f"{variable.replace('_', ' ').title()} ({high_conf_cold} areas)")
            
            # Write spatial autocorrelation summary
            if significant_autocorrelation:
                f.write("1. **Spatial Autocorrelation**: Significant spatial patterns were detected for:\n")
                for var in significant_autocorrelation:
                    f.write(f"   - {var}\n")
                f.write("   This indicates that these housing variables exhibit clear geographic clustering rather than random distribution.\n\n")
            else:
                f.write("1. **Spatial Autocorrelation**: No significant spatial patterns were detected for the analyzed variables, ")
                f.write("suggesting relatively uniform or random distribution across the region.\n\n")
            
            # Write hotspot summary
            if hotspot_areas:
                f.write("2. **Hot Spots**: Statistically significant hot spots (areas with high values) were identified for:\n")
                for var in hotspot_areas:
                    f.write(f"   - {var}\n")
                f.write("   These areas represent potential investment opportunities or areas of concern depending on the variable.\n\n")
            
            # Write coldspot summary
            if coldspot_areas:
                f.write("3. **Cold Spots**: Statistically significant cold spots (areas with low values) were identified for:\n")
                for var in coldspot_areas:
                    f.write(f"   - {var}\n")
                f.write("   These areas may represent undervalued markets or areas requiring economic development.\n\n")
            
            # Write implications
            f.write("### Market Implications\n\n")
            f.write("This spatial analysis provides valuable insights for various stakeholders:\n\n")
            f.write("- **Investors**: Can identify neighborhoods with strong growth potential and avoid overheated markets\n")
            f.write("- **Policymakers**: Can target interventions to address housing affordability in specific areas\n")
            f.write("- **Developers**: Can identify gaps in the market and potential areas for new housing projects\n")
            f.write("- **Residents**: Can make more informed decisions about home purchases or rentals based on neighborhood trends\n\n")
            
            f.write("### Next Steps\n\n")
            f.write("To build on this analysis, consider:\n\n")
            f.write("1. Incorporating temporal data to analyze how spatial patterns change over time\n")
            f.write("2. Conducting detailed analysis of specific high-interest neighborhoods identified in this report\n")
            f.write("3. Integrating additional datasets such as transit accessibility, school quality, or crime rates\n")
            f.write("4. Developing predictive models that incorporate both spatial and non-spatial factors\n")
        
        print(f"\nAnalysis complete! Report saved to {report_file}")
        return {
            'results': results,
            'report_file': report_file
        }
