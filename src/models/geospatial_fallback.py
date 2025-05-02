"""
Lightweight Geospatial Analysis Module

This module provides a fallback implementation of geospatial hotspot analysis
techniques for the Miami Housing Impact Hub that doesn't rely on the pysal library.
Instead, it uses more commonly available libraries like pandas, numpy, and sklearn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class LightweightGeospatialAnalysis:
    """
    A lightweight implementation of geospatial hotspot analysis techniques
    that doesn't rely on heavy dependencies like pysal.
    
    This class uses a combination of heatmaps, DBSCAN clustering, and
    basic spatial statistics to identify spatial patterns in housing data.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the lightweight geospatial analysis module.
        
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
        self.analysis_results = {}
        self.hotspot_map = None
        self.cluster_results = None
        self.current_variable = None
    
    def load_data(self, file_path=None):
        """
        Load data for geospatial analysis, either from a specific file
        or from the processed data directory.
        
        Args:
            file_path: Path to the data file (optional)
            
        Returns:
            pd.DataFrame: The loaded data
        """
        try:
            if file_path is not None and os.path.exists(file_path):
                self.data = pd.read_csv(file_path)
                logger.info(f"Loaded data from {file_path}: {self.data.shape[0]} records")
            else:
                # Try to find and load a suitable file from the processed directory
                potential_files = [
                    self.data_dir / "miami_dade_merged_data.csv",
                    self.data_dir / "airbnb" / "airbnb_listings_processed.csv",
                    self.data_dir / "airbnb" / "airbnb_by_zipcode.csv"
                ]
                
                for file_path in potential_files:
                    if os.path.exists(file_path):
                        self.data = pd.read_csv(file_path)
                        logger.info(f"Loaded data from {file_path}: {self.data.shape[0]} records")
                        break
                
                if self.data is None:
                    logger.error("No suitable data file found in the processed directory")
                    return None
            
            # Check if latitude and longitude columns exist for mapping
            lat_columns = [col for col in self.data.columns if 'lat' in col.lower()]
            lng_columns = [col for col in self.data.columns if any(term in col.lower() for term in ['lng', 'lon', 'long'])]
            
            if not lat_columns or not lng_columns:
                logger.warning("Latitude and/or longitude columns not found in the data")
                return self.data
            
            # Standardize column names
            self.data = self.data.rename(columns={
                lat_columns[0]: 'latitude',
                lng_columns[0]: 'longitude'
            })
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def create_heatmap(self, variable=None, normalize=True, radius=15, map_zoom=11):
        """
        Create a heatmap visualization of a variable's spatial distribution.
        
        Args:
            variable: Column name to visualize (defaults to density or count if available)
            normalize: Whether to normalize the variable before mapping
            radius: Radius for the heatmap points
            map_zoom: Initial zoom level for the map
            
        Returns:
            folium.Map: Interactive heatmap
        """
        try:
            if self.data is None:
                logger.error("No data loaded. Call load_data() first.")
                return None
            
            # Check if latitude and longitude columns exist
            if 'latitude' not in self.data.columns or 'longitude' not in self.data.columns:
                logger.error("Latitude and longitude columns required for heatmap creation")
                return None
            
            # If no variable specified, try to find a suitable one
            if variable is None:
                # Look for density or count columns
                density_cols = [col for col in self.data.columns if any(term in col.lower() for term in ['density', 'count', 'frequency', 'number'])]
                if density_cols:
                    variable = density_cols[0]
                    logger.info(f"Using {variable} for heatmap")
                else:
                    # Default to count of 1 per row if no suitable column found
                    self.data['count'] = 1
                    variable = 'count'
                    logger.info("No density/count column found, using row count")
            
            # Prepare data for heatmap
            heat_data = self.data.dropna(subset=['latitude', 'longitude'])
            
            if variable not in heat_data.columns:
                logger.error(f"Variable {variable} not found in data columns")
                return None
            
            # Normalize the variable for better visualization if needed
            if normalize:
                heat_data['heat_value'] = (heat_data[variable] - heat_data[variable].min()) / (heat_data[variable].max() - heat_data[variable].min())
            else:
                heat_data['heat_value'] = heat_data[variable]
            
            # Calculate center of the map
            avg_lat = heat_data['latitude'].mean()
            avg_lon = heat_data['longitude'].mean()
            
            # Create the map
            heatmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=map_zoom, 
                                 tiles='CartoDB positron', width='100%', height='100%')
            
            # Prepare heatmap data
            heat_matrix = [[row['latitude'], row['longitude'], row['heat_value']] for _, row in heat_data.iterrows()]
            
            # Add the heatmap layer
            HeatMap(heat_matrix, radius=radius, blur=10, max_zoom=map_zoom+2, gradient={
                0.0: 'blue', 0.3: 'lime', 0.5: 'yellow', 0.7: 'orange', 1.0: 'red'
            }).add_to(heatmap)
            
            # Add a title and explanation
            title_html = f'''
                <h3 align="center" style="font-size:16px"><b>Heatmap of {variable}</b></h3>
                <p align="center">Darker red areas indicate higher values of {variable}.</p>
            '''
            heatmap.get_root().html.add_child(folium.Element(title_html))
            
            # Save the map to file for easy access
            output_path = self.output_dir / f"heatmap_{variable}.html"
            heatmap.save(str(output_path))
            logger.info(f"Heatmap saved to {output_path}")
            
            # Store the map for later use
            self.hotspot_map = heatmap
            self.current_variable = variable
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return None
    
    def detect_hotspots_dbscan(self, variable=None, eps=0.01, min_samples=5, hotspot_percentile=75):
        """
        Detect hotspots using DBSCAN clustering algorithm.
        
        Args:
            variable: Column name to use for hotspot detection (defaults to previous or suitable column)
            eps: DBSCAN epsilon parameter (spatial distance threshold)
            min_samples: DBSCAN min_samples parameter (minimum points to form a cluster)
            hotspot_percentile: Percentile threshold to classify clusters as hotspots
            
        Returns:
            pd.DataFrame: Results with cluster assignments and hotspot classifications
        """
        try:
            if self.data is None:
                logger.error("No data loaded. Call load_data() first.")
                return None
            
            # If no variable specified, use previously used one or find a suitable one
            if variable is None:
                if self.current_variable is not None:
                    variable = self.current_variable
                else:
                    # Look for density or count columns
                    density_cols = [col for col in self.data.columns if any(term in col.lower() for term in ['density', 'count', 'frequency', 'number'])]
                    if density_cols:
                        variable = density_cols[0]
                        logger.info(f"Using {variable} for hotspot detection")
                    else:
                        # Default to count of 1 per row if no suitable column found
                        self.data['count'] = 1
                        variable = 'count'
                        logger.info("No density/count column found, using row count")
            
            # Prepare data for clustering
            cluster_data = self.data.dropna(subset=['latitude', 'longitude', variable]).copy()
            
            if len(cluster_data) == 0:
                logger.error("No valid data for clustering after dropping NAs")
                return None
            
            # Standardize coordinates for clustering
            coords = cluster_data[['latitude', 'longitude']].values
            coords_scaled = StandardScaler().fit_transform(coords)
            
            # Perform DBSCAN clustering
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)
            cluster_data['cluster_label'] = db.labels_
            
            # Calculate statistics per cluster
            cluster_stats = cluster_data.groupby('cluster_label').agg({
                variable: ['mean', 'median', 'std', 'count'],
                'latitude': 'mean',
                'longitude': 'mean'
            })
            
            # Flatten the multi-index columns
            cluster_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cluster_stats.columns]
            
            # Identify hotspots (high value clusters)
            value_threshold = cluster_data[variable].quantile(hotspot_percentile/100)
            cluster_stats['is_hotspot'] = cluster_stats[f"{variable}_mean"] > value_threshold
            
            # Add cluster information back to the original data
            cluster_data['is_hotspot'] = cluster_data['cluster_label'].map(
                cluster_stats['is_hotspot'].to_dict()).fillna(False)
            
            # Generate cluster descriptions
            cluster_descriptions = {}
            for idx, row in cluster_stats.iterrows():
                if idx == -1:  # Noise points
                    cluster_descriptions[idx] = "Noise (unclustered points)"
                elif row['is_hotspot']:
                    intensity = "Very high" if row[f"{variable}_mean"] > cluster_data[variable].quantile(0.9) else "High"
                    cluster_descriptions[idx] = f"{intensity} {variable} hotspot"
                else:
                    intensity = "Low" if row[f"{variable}_mean"] < cluster_data[variable].quantile(0.25) else "Moderate"
                    cluster_descriptions[idx] = f"{intensity} {variable} area"
            
            cluster_data['cluster_description'] = cluster_data['cluster_label'].map(cluster_descriptions)
            
            # Store results
            self.cluster_results = {
                'data': cluster_data,
                'stats': cluster_stats,
                'variable': variable,
                'descriptions': cluster_descriptions
            }
            
            # Return the clustered data with hotspot information
            return cluster_data
            
        except Exception as e:
            logger.error(f"Error detecting hotspots: {e}")
            return None
    
    def visualize_hotspots(self, include_all_points=True, map_zoom=11):
        """
        Create an interactive map visualization of detected hotspots.
        
        Args:
            include_all_points: Whether to include all points or just cluster centers
            map_zoom: Initial zoom level for the map
            
        Returns:
            folium.Map: Interactive map with hotspots highlighted
        """
        try:
            if self.cluster_results is None:
                logger.error("No cluster results available. Call detect_hotspots_dbscan() first.")
                return None
            
            # Extract data from cluster results
            cluster_data = self.cluster_results['data']
            cluster_stats = self.cluster_results['stats']
            variable = self.cluster_results['variable']
            
            # Calculate center of the map
            avg_lat = cluster_data['latitude'].mean()
            avg_lon = cluster_data['longitude'].mean()
            
            # Create the map
            hotspot_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=map_zoom, 
                                     tiles='CartoDB positron', width='100%', height='100%')
            
            # Add a legend
            legend_html = '''
                <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                            padding: 10px; border: 2px solid grey; border-radius: 5px;">
                    <p><strong>Cluster Legend:</strong></p>
                    <div><i class="fa fa-circle" style="color:red"></i> High Value Hotspot</div>
                    <div><i class="fa fa-circle" style="color:blue"></i> Regular Cluster</div>
                    <div><i class="fa fa-circle" style="color:gray"></i> Unclustered Points</div>
                </div>
            '''
            hotspot_map.get_root().html.add_child(folium.Element(legend_html))
            
            # Add markers for cluster centers
            for idx, row in cluster_stats.iterrows():
                if idx != -1:  # Skip noise points
                    # Choose color based on hotspot status
                    color = 'red' if row['is_hotspot'] else 'blue'
                    
                    # Create a CircleMarker for the cluster center
                    folium.CircleMarker(
                        location=[row['latitude_mean'], row['longitude_mean']],
                        radius=10,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        popup=f"Cluster {idx}<br>Mean {variable}: {row[f'{variable}_mean']:.2f}<br>Count: {row[f'{variable}_count']:.0f}"
                    ).add_to(hotspot_map)
            
            # Add all points if requested
            if include_all_points:
                # Create marker clusters for better performance
                markers = MarkerCluster(name="All Points").add_to(hotspot_map)
                
                # Add each point
                for _, row in cluster_data.iterrows():
                    # Choose color based on cluster type
                    if row['cluster_label'] == -1:
                        color = 'gray'  # Noise points
                    elif row['is_hotspot']:
                        color = 'red'   # Hotspot
                    else:
                        color = 'blue'  # Regular cluster
                    
                    # Create a small circle marker for each point
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.5,
                        popup=f"{variable}: {row[variable]}<br>Cluster: {row['cluster_description']}"
                    ).add_to(markers)
            
            # Add layer control
            folium.LayerControl().add_to(hotspot_map)
            
            # Add a title and explanation
            title_html = f'''
                <h3 align="center" style="font-size:16px"><b>Hotspot Analysis: {variable}</b></h3>
                <p align="center">Red clusters indicate hotspots (high values of {variable}).</p>
            '''
            hotspot_map.get_root().html.add_child(folium.Element(title_html))
            
            # Save the map to file for easy access
            output_path = self.output_dir / f"hotspot_map_{variable}.html"
            hotspot_map.save(str(output_path))
            logger.info(f"Hotspot map saved to {output_path}")
            
            return hotspot_map
            
        except Exception as e:
            logger.error(f"Error visualizing hotspots: {e}")
            return None
    
    def generate_hotspot_report(self, variable=None, num_hotspots=5):
        """
        Generate a report of the most significant hotspots and their characteristics.
        
        Args:
            variable: Column name to analyze (defaults to previously used variable)
            num_hotspots: Number of top hotspots to include in the report
            
        Returns:
            dict: Report data with hotspot information
        """
        try:
            if self.cluster_results is None:
                if variable is not None:
                    # Try to run the hotspot detection with the provided variable
                    self.detect_hotspots_dbscan(variable=variable)
                else:
                    logger.error("No cluster results available. Call detect_hotspots_dbscan() first.")
                    return None
            
            # Extract data from cluster results
            cluster_data = self.cluster_results['data']
            cluster_stats = self.cluster_results['stats']
            variable = self.cluster_results['variable']
            
            # Filter to just the hotspot clusters
            hotspot_clusters = cluster_stats[cluster_stats['is_hotspot'] & (cluster_stats.index != -1)]
            
            # Sort by the variable mean to get top hotspots
            top_hotspots = hotspot_clusters.sort_values(f"{variable}_mean", ascending=False).head(num_hotspots)
            
            # Create report data
            report = {
                'variable': variable,
                'total_hotspots': sum(hotspot_clusters[f"{variable}_count"]),
                'percent_in_hotspots': sum(hotspot_clusters[f"{variable}_count"]) / len(cluster_data) * 100,
                'num_hotspot_clusters': len(hotspot_clusters),
                'hotspot_details': []
            }
            
            # Add details for each top hotspot
            for idx, row in top_hotspots.iterrows():
                # Get points in this cluster
                cluster_points = cluster_data[cluster_data['cluster_label'] == idx]
                
                # Try to get location names if available
                location_cols = [col for col in cluster_points.columns if any(term in col.lower() for term in ['neighborhood', 'zipcode', 'zip_code', 'zip', 'area', 'city'])]
                location_name = "Unknown location"
                if location_cols:
                    # Use most common value in the first location column
                    most_common = cluster_points[location_cols[0]].value_counts().idxmax()
                    location_name = f"{most_common}"
                
                # Calculate distance to downtown Miami (approx coordinates)
                downtown_lat, downtown_lon = 25.7617, -80.1918
                distance = np.sqrt((row['latitude_mean'] - downtown_lat)**2 + (row['longitude_mean'] - downtown_lon)**2) * 111  # rough km conversion
                
                # Create hotspot details
                hotspot_detail = {
                    'cluster_id': idx,
                    'location': location_name,
                    'center_lat': row['latitude_mean'],
                    'center_lon': row['longitude_mean'],
                    'mean_value': row[f"{variable}_mean"],
                    'point_count': row[f"{variable}_count"],
                    'std_dev': row.get(f"{variable}_std", 0),
                    'distance_to_downtown_km': distance,
                    'relative_intensity': row[f"{variable}_mean"] / cluster_stats[f"{variable}_mean"].mean()
                }
                
                report['hotspot_details'].append(hotspot_detail)
            
            # Generate summary statistics
            if report['hotspot_details']:
                report['avg_hotspot_size'] = sum(h['point_count'] for h in report['hotspot_details']) / len(report['hotspot_details'])
                report['avg_distance_to_downtown'] = sum(h['distance_to_downtown_km'] for h in report['hotspot_details']) / len(report['hotspot_details'])
                report['highest_intensity'] = max(h['relative_intensity'] for h in report['hotspot_details'])
            
            # Save report as CSV for reference
            report_df = pd.DataFrame(report['hotspot_details'])
            output_path = self.output_dir / f"hotspot_report_{variable}.csv"
            report_df.to_csv(output_path, index=False)
            logger.info(f"Hotspot report saved to {output_path}")
            
            # Store in analysis results
            self.analysis_results[f'hotspot_report_{variable}'] = report
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating hotspot report: {e}")
            return None
    
    def plot_hotspot_distribution(self, variable=None, figsize=(12, 8)):
        """
        Create statistical plots of the hotspot distribution compared to non-hotspots.
        
        Args:
            variable: Column name to analyze (defaults to previously used variable)
            figsize: Size of the figure (width, height)
            
        Returns:
            matplotlib.figure.Figure: Figure with statistical plots
        """
        try:
            if self.cluster_results is None:
                if variable is not None:
                    # Try to run the hotspot detection with the provided variable
                    self.detect_hotspots_dbscan(variable=variable)
                else:
                    logger.error("No cluster results available. Call detect_hotspots_dbscan() first.")
                    return None
            
            # Extract data from cluster results
            cluster_data = self.cluster_results['data']
            variable = self.cluster_results['variable']
            
            # Create a figure with subplots
            fig, axs = plt.subplots(2, 1, figsize=figsize)
            
            # Plot 1: Distribution of variable in hotspots vs non-hotspots
            ax1 = axs[0]
            sns.histplot(data=cluster_data, x=variable, hue='is_hotspot', 
                        kde=True, ax=ax1, bins=20, element='step',
                        hue_order=[True, False], palette=['red', 'blue'])
            ax1.set_title(f'Distribution of {variable} in Hotspots vs Non-Hotspots')
            ax1.set_xlabel(variable)
            ax1.set_ylabel('Frequency')
            ax1.legend(['Hotspots', 'Non-Hotspots'])
            
            # Plot 2: Boxplot of variable by cluster type
            ax2 = axs[1]
            sns.boxplot(data=cluster_data, x='is_hotspot', y=variable, ax=ax2,
                       order=[True, False], palette=['red', 'blue'])
            ax2.set_title(f'Boxplot of {variable} by Cluster Type')
            ax2.set_xlabel('Is Hotspot')
            ax2.set_ylabel(variable)
            ax2.set_xticklabels(['Hotspots', 'Non-Hotspots'])
            
            # Add statistics annotations to the boxplot
            hotspot_mean = cluster_data[cluster_data['is_hotspot']][variable].mean()
            non_hotspot_mean = cluster_data[~cluster_data['is_hotspot']][variable].mean()
            ax2.annotate(f"Mean: {hotspot_mean:.2f}", xy=(0, hotspot_mean), 
                        xytext=(0.15, hotspot_mean), 
                        arrowprops=dict(facecolor='black', shrink=0.05))
            ax2.annotate(f"Mean: {non_hotspot_mean:.2f}", xy=(1, non_hotspot_mean), 
                        xytext=(1.15, non_hotspot_mean), 
                        arrowprops=dict(facecolor='black', shrink=0.05))
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.output_dir / f"hotspot_distribution_{variable}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hotspot distribution plot saved to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting hotspot distribution: {e}")
            return None
    
    def run_full_analysis(self, variable=None, include_all_plots=True):
        """
        Run a complete geospatial hotspot analysis for a given variable.
        
        Args:
            variable: Column name to analyze
            include_all_plots: Whether to generate all plots
            
        Returns:
            dict: Analysis results including maps and reports
        """
        try:
            # Step 1: Load data
            if self.data is None:
                self.load_data()
            
            if self.data is None or len(self.data) == 0:
                logger.error("No data available for analysis")
                return None
            
            # Select a suitable variable if none provided
            if variable is None:
                # First check density columns
                density_cols = [col for col in self.data.columns if any(term in col.lower() for term in ['density', 'count', 'price', 'value', 'rent'])]
                
                if density_cols:
                    variable = density_cols[0]
                    logger.info(f"Selected {variable} for analysis")
                else:
                    # Default to row count
                    self.data['count'] = 1
                    variable = 'count'
                    logger.info("No suitable column found, using row count")
            
            # Step 2: Create heat map
            heatmap = self.create_heatmap(variable=variable)
            
            # Step 3: Detect hotspots
            hotspot_data = self.detect_hotspots_dbscan(variable=variable)
            
            # Step 4: Visualize hotspots
            hotspot_map = self.visualize_hotspots()
            
            # Step 5: Generate hotspot report
            report = self.generate_hotspot_report(variable=variable)
            
            # Step 6: Create additional plots if requested
            if include_all_plots:
                distribution_plot = self.plot_hotspot_distribution(variable=variable)
            
            # Compile results
            results = {
                'variable': variable,
                'heatmap_path': str(self.output_dir / f"heatmap_{variable}.html"),
                'hotspot_map_path': str(self.output_dir / f"hotspot_map_{variable}.html"),
                'report_path': str(self.output_dir / f"hotspot_report_{variable}.csv"),
                'distribution_plot_path': str(self.output_dir / f"hotspot_distribution_{variable}.png") if include_all_plots else None,
                'report': report
            }
            
            logger.info(f"Completed full geospatial hotspot analysis for {variable}")
            return results
            
        except Exception as e:
            logger.error(f"Error running full analysis: {e}")
            return None


# Example usage
if __name__ == "__main__":
    analysis = LightweightGeospatialAnalysis()
    analysis.load_data()
    results = analysis.run_full_analysis(variable='price')
    print(f"Analysis results saved to: {analysis.output_dir}")
