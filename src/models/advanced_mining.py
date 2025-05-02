#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Data Mining Techniques Module

This module implements advanced data mining and machine learning techniques
for the Miami Housing Impact Hub, using real data from Miami-Dade County.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from kneed import KneeLocator

class NeighborhoodSegmentation:
    """
    K-means clustering implementation for Miami-Dade neighborhoods.
    
    This class segments neighborhoods based on housing metrics to identify
    distinct market segments within Miami-Dade County.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize with path to processed data directory.
        
        Args:
            data_dir: Path to processed data directory (optional)
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[2]
            self.data_dir = self.project_root / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        # Initialize attributes
        self.data = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.optimal_k = None
        self.cluster_data = None
        self.feature_columns = None
        self.features = [
            'airbnb_count',
            'airbnb_density',
            'median_property_price',
            'median_airbnb_price',
            'median_income',
            'rent_to_income_ratio',
            'price_to_income_ratio'
        ]
    
    def load_data(self):
        """Load the processed data for clustering analysis."""
        try:
            # Try to load the main merged dataset
            data_path = self.data_dir / "miami_dade_merged_data.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                print(f"Loaded {len(self.data)} neighborhoods from {data_path}")
                
                # Also check for zipcode summary which might have additional metrics
                zip_summary_path = self.data_dir / "merged" / "zipcode_summary.csv"
                if zip_summary_path.exists():
                    zip_data = pd.read_csv(zip_summary_path)
                    # If we have more data in the zipcode summary, use that instead
                    if len(zip_data) > len(self.data):
                        self.data = zip_data
                        print(f"Using more comprehensive zipcode data with {len(self.data)} areas")
                
                return True
            else:
                print(f"Warning: Could not find dataset at {data_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare and scale features for clustering."""
        if self.data is None:
            if not self.load_data():
                return False
        
        # Identify available features from our desired list
        self.feature_columns = [col for col in self.features if col in self.data.columns]
        
        if len(self.feature_columns) < 2:
            print("Warning: Not enough features available for meaningful clustering")
            # Try to identify any numeric columns we could use
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove any ID columns or other non-meaningful numerics
            exclude_terms = ['id', 'code', 'zip', 'latitude', 'longitude']
            numeric_cols = [col for col in numeric_cols if not any(term in col.lower() for term in exclude_terms)]
            
            if len(numeric_cols) >= 2:
                self.feature_columns = numeric_cols[:4]  # Use up to 4 numeric columns
                print(f"Using available numeric columns for clustering: {self.feature_columns}")
            else:
                print("Error: Insufficient numeric data for clustering")
                return False
        
        # Drop rows with missing values in feature columns
        complete_data = self.data.dropna(subset=self.feature_columns)
        
        if len(complete_data) < 5:
            print("Error: Too few complete data points for clustering")
            return False
        
        # Scale the features
        features_scaled = self.scaler.fit_transform(complete_data[self.feature_columns])
        
        # Store data for clustering
        self.cluster_data = pd.DataFrame(
            features_scaled, 
            columns=self.feature_columns,
            index=complete_data.index
        )
        
        # Keep track of original data with same indices
        self.prepared_data = complete_data
        
        print(f"Prepared {len(self.cluster_data)} neighborhoods with {len(self.feature_columns)} features for clustering")
        return True
    
    def find_optimal_k(self, max_k=10, plot=True):
        """
        Find the optimal number of clusters using the Elbow method.
        
        Args:
            max_k: Maximum number of clusters to consider
            plot: Whether to generate and save an elbow plot
        
        Returns:
            Optimal K value
        """
        if self.cluster_data is None:
            if not self.prepare_features():
                return None
        
        # Limit max_k to half the number of data points 
        max_k = min(max_k, len(self.cluster_data) // 2)
        max_k = max(max_k, 2)  # Ensure min of 2 clusters
        
        print(f"Finding optimal K from 2 to {max_k} clusters...")
        
        # Compute sum of squared distances for range of K values
        distortions = []
        K_range = range(1, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.cluster_data)
            distortions.append(kmeans.inertia_)
        
        # Use KneeLocator to find the elbow point
        try:
            kneedle = KneeLocator(
                K_range, distortions, curve='convex', direction='decreasing'
            )
            optimal_k = kneedle.elbow
            
            if optimal_k is None:
                # Default to 3 clusters if no clear elbow is found
                optimal_k = 3
                print("No clear elbow found, defaulting to 3 clusters")
            else:
                print(f"Optimal number of clusters: {optimal_k}")
        except Exception as e:
            print(f"Error finding optimal K: {e}")
            optimal_k = 3
        
        self.optimal_k = optimal_k
        
        # Generate elbow plot if requested
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(K_range, distortions, 'bx-')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Distortion (Sum of squared distances)')
            plt.title('Elbow Method For Optimal k')
            
            if optimal_k:
                plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], 
                          linestyles='dashed', colors='r')
                plt.text(optimal_k + 0.2, distortions[optimal_k-1], 
                        f'Optimal k={optimal_k}', 
                        fontsize=12, color='r')
            
            # Save the plot
            os.makedirs(self.project_root / "visualizations" / "clustering", exist_ok=True)
            plt.savefig(self.project_root / "visualizations" / "clustering" / "elbow_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        return optimal_k
    
    def perform_clustering(self, k=None, force_recalculate=False):
        """
        Perform K-means clustering on the prepared data.
        
        Args:
            k: Number of clusters (if None, will use optimal_k or calculate it)
            force_recalculate: If True, will recalculate optimal k even if already calculated
            
        Returns:
            DataFrame with original data and cluster assignments
        """
        if self.cluster_data is None:
            if not self.prepare_features():
                return None
        
        # Determine number of clusters
        if k is not None:
            self.optimal_k = k
        elif self.optimal_k is None or force_recalculate:
            self.find_optimal_k()
        
        # Run K-means with optimal k
        print(f"Performing K-means clustering with {self.optimal_k} clusters...")
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        self.kmeans.fit(self.cluster_data)
        
        # Add cluster labels to original data
        self.prepared_data['cluster'] = self.kmeans.labels_
        
        # Create cluster profiles
        self.create_cluster_profiles()
        
        return self.prepared_data
    
    def create_cluster_profiles(self):
        """
        Create profiles for each cluster with key characteristics.
        
        Returns:
            DataFrame with cluster profiles
        """
        if 'cluster' not in self.prepared_data.columns:
            print("Error: Clustering has not been performed yet")
            return None
        
        # Compute cluster profiles based on mean values of features
        profiles = self.prepared_data.groupby('cluster')[self.feature_columns].mean()
        
        # Count neighborhoods in each cluster
        counts = self.prepared_data['cluster'].value_counts().sort_index()
        profiles['neighborhood_count'] = counts
        
        # Calculate percentile ranks for each feature within clusters
        for feature in self.feature_columns:
            profiles[f"{feature}_percentile"] = profiles[feature].rank(pct=True) * 100
        
        # Compute cluster centers (for radar charts)
        centers = self.kmeans.cluster_centers_
        
        # Map back from scaled to original values for interpretability
        feature_mins = self.prepared_data[self.feature_columns].min()
        feature_maxs = self.prepared_data[self.feature_columns].max()
        
        # Create descriptive labels for clusters
        labels = []
        
        for idx, row in profiles.iterrows():
            # Find the top 2 distinctive features for this cluster
            percentile_cols = [col for col in profiles.columns if col.endswith('_percentile')]
            top_features = profiles.loc[idx, percentile_cols].nlargest(2)
            
            # Strip _percentile suffix
            top_feature_names = [name.replace('_percentile', '') for name in top_features.index]
            
            # Create a descriptive label
            if 'airbnb_density' in top_feature_names and 'price_to_income_ratio' in top_feature_names:
                label = "High Impact Tourism Area"
            elif 'airbnb_density' in top_feature_names:
                label = "Tourism Hotspot"
            elif 'median_property_price' in top_feature_names:
                label = "Luxury Real Estate Market"
            elif 'median_income' in top_feature_names:
                label = "Affluent Residential Area"
            elif 'price_to_income_ratio' in top_feature_names:
                label = "Housing Affordability Challenge Area"
            else:
                # Use the top feature as fallback
                feature_name = top_feature_names[0].replace('_', ' ').title()
                label = f"High {feature_name} Area"
            
            labels.append(label)
        
        profiles['cluster_label'] = labels
        
        # Store the profiles
        self.cluster_profiles = profiles
        
        # Create a detailed summary with neighborhood listings
        self.detailed_profiles = {}
        
        for cluster_id in range(self.optimal_k):
            cluster_neighborhoods = self.prepared_data[self.prepared_data['cluster'] == cluster_id]
            
            if 'neighborhood' in cluster_neighborhoods.columns:
                neighborhood_list = cluster_neighborhoods['neighborhood'].tolist()
            elif 'zipcode' in cluster_neighborhoods.columns:
                neighborhood_list = cluster_neighborhoods['zipcode'].tolist()
            else:
                neighborhood_list = [f"Area {i}" for i in cluster_neighborhoods.index]
            
            self.detailed_profiles[cluster_id] = {
                'label': profiles.loc[cluster_id, 'cluster_label'],
                'count': profiles.loc[cluster_id, 'neighborhood_count'],
                'neighborhoods': neighborhood_list,
                'profile': {feature: profiles.loc[cluster_id, feature] for feature in self.feature_columns}
            }
        
        # Save profiles to CSV
        os.makedirs(self.project_root / "data" / "processed" / "clustering", exist_ok=True)
        profiles.to_csv(self.project_root / "data" / "processed" / "clustering" / "cluster_profiles.csv")
        
        return profiles
    
    def visualize_clusters(self):
        """
        Create visualizations of the clustering results.
        
        Returns:
            Path to the generated visualization files
        """
        if 'cluster' not in self.prepared_data.columns:
            print("Error: Clustering has not been performed yet")
            return None
        
        # Create output directory
        output_dir = self.project_root / "visualizations" / "clustering"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Scatter Plot of 2 Most Important Features
        plt.figure(figsize=(12, 8))
        
        # Choose the 2 most discriminative features if we have enough features
        if len(self.feature_columns) >= 2:
            # Use the features with highest variance between clusters
            feature_variances = []
            for feature in self.feature_columns:
                variance = self.prepared_data.groupby('cluster')[feature].mean().var()
                feature_variances.append((feature, variance))
            
            # Sort features by between-cluster variance
            sorted_features = sorted(feature_variances, key=lambda x: x[1], reverse=True)
            x_feature, y_feature = sorted_features[0][0], sorted_features[1][0]
        else:
            # Fallback to first two features
            x_feature, y_feature = self.feature_columns[0], self.feature_columns[1]
        
        # Create scatter plot with cluster colors
        sns.scatterplot(
            data=self.prepared_data,
            x=x_feature,
            y=y_feature,
            hue='cluster',
            palette='viridis',
            s=100,
            alpha=0.7
        )
        
        # Add cluster centers
        centers = self.kmeans.cluster_centers_
        center_df = pd.DataFrame(
            self.scaler.inverse_transform(centers),
            columns=self.feature_columns
        )
        
        plt.scatter(
            center_df[x_feature],
            center_df[y_feature],
            s=200,
            marker='X',
            c='red',
            label='Cluster Centers'
        )
        
        # Add neighborhood labels if available
        if 'neighborhood' in self.prepared_data.columns:
            for idx, row in self.prepared_data.iterrows():
                plt.annotate(
                    row['neighborhood'],
                    (row[x_feature], row[y_feature]),
                    fontsize=8,
                    alpha=0.7,
                    ha='right'
                )
        
        # Formatting
        plt.title(f'Neighborhood Segments by {x_feature.replace("_", " ").title()} and {y_feature.replace("_", " ").title()}')
        plt.xlabel(x_feature.replace('_', ' ').title())
        plt.ylabel(y_feature.replace('_', ' ').title())
        plt.legend(title='Cluster')
        plt.grid(alpha=0.3)
        
        # Save the plot
        scatter_path = output_dir / "cluster_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Radar Chart for Cluster Profiles
        # Normalize features to 0-1 scale for radar chart
        features_scaled = self.cluster_data.copy()
        
        # Create a radar chart for each cluster
        for cluster_id in range(self.optimal_k):
            # Get the profile for this cluster
            profile = self.cluster_profiles.loc[cluster_id]
            
            # Create the plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Get values for this cluster (already scaled)
            center = self.kmeans.cluster_centers_[cluster_id]
            
            # Number of variables
            N = len(self.feature_columns)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Draw the plot
            ax.plot(angles, np.append(center, center[0]), 'o-', linewidth=2, label=f'Cluster {cluster_id}')
            ax.fill(angles, np.append(center, center[0]), alpha=0.25)
            
            # Fix axis to go in the right order and start at 12 o'clock
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines for each angle and label
            plt.xticks(angles[:-1], [f.replace('_', ' ').title() for f in self.feature_columns])
            
            # Add cluster label and stats
            plt.title(f"Cluster {cluster_id}: {profile['cluster_label']} (n={int(profile['neighborhood_count'])})")
            
            # Save the radar chart
            radar_path = output_dir / f"cluster_{cluster_id}_radar.png"
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Neighborhood Distribution Bar Chart
        plt.figure(figsize=(14, 6))
        cluster_counts = self.prepared_data['cluster'].value_counts().sort_index()
        
        # Add cluster labels to the chart
        cluster_labels = [f"{i}: {self.cluster_profiles.loc[i, 'cluster_label']}" 
                          for i in range(self.optimal_k)]
        
        # Create bar chart with cluster labels
        bars = plt.bar(cluster_labels, cluster_counts, color='skyblue')
        
        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Number of Neighborhoods in Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Neighborhoods')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the distribution chart
        dist_path = output_dir / "cluster_distribution.png"
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap of Cluster Profiles
        plt.figure(figsize=(14, 8))
        
        # Create a heatmap of the cluster profiles for each feature
        profile_heatmap = self.cluster_profiles[self.feature_columns].copy()
        
        # Normalize each feature column to 0-1 scale for better visualization
        for feature in self.feature_columns:
            profile_heatmap[feature] = (profile_heatmap[feature] - profile_heatmap[feature].min()) / \
                                      (profile_heatmap[feature].max() - profile_heatmap[feature].min())
        
        # Create cluster labels for y-axis
        cluster_labels = [f"{i}: {self.cluster_profiles.loc[i, 'cluster_label']}" 
                          for i in range(self.optimal_k)]
        
        # Plot heatmap
        sns.heatmap(
            profile_heatmap,
            annot=True,
            cmap='YlGnBu',
            fmt='.2f',
            linewidths=0.5,
            yticklabels=cluster_labels
        )
        
        plt.title('Cluster Profiles (Normalized Feature Values)')
        plt.ylabel('Cluster')
        plt.xlabel('Features')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the heatmap
        heatmap_path = output_dir / "cluster_profiles_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Write markdown explanation of clusters
        explanation_md = f"""# Miami Housing Market Neighborhood Segmentation

## Methodology
This analysis uses K-means clustering to segment Miami-Dade neighborhoods into {self.optimal_k} distinct market types based on real housing and Airbnb data.

The clustering algorithm identified patterns across multiple metrics:
- Airbnb density and counts
- Property prices and rental rates
- Income levels and affordability ratios

## Identified Neighborhood Segments

"""
        # Add details for each cluster
        for cluster_id in range(self.optimal_k):
            profile = self.detailed_profiles[cluster_id]
            explanation_md += f"""### Cluster {cluster_id}: {profile['label']}
- **Neighborhoods**: {len(profile['neighborhoods'])} areas, including {', '.join(str(n) for n in profile['neighborhoods'][:5])}{'...' if len(profile['neighborhoods']) > 5 else ''}
- **Key Characteristics**:
"""
            # Add top 3 distinctive features
            top_features = [(f, profile['profile'][f]) for f in self.feature_columns]
            sorted_features = sorted(top_features, key=lambda x: x[1], reverse=True)
            
            for feature, value in sorted_features[:3]:
                feature_name = feature.replace('_', ' ').title()
                explanation_md += f"  - {feature_name}: {value:.2f}\n"
            
            explanation_md += "\n"
        
        # Add business implications
        explanation_md += """## Business Implications

### For Policymakers
- Different neighborhood segments require tailored policy approaches
- High-density Airbnb clusters may need specific regulations to prevent affordability issues
- Low-income, high-price-to-income ratio areas need affordable housing interventions

### For Investors
- Each segment represents a distinct investment opportunity with different risk profiles
- Tourism hotspots offer higher Airbnb revenue potential but may face stricter regulations
- Luxury markets provide stability but require higher capital investment

### For Residents
- Understand how your neighborhood compares to others on affordability metrics
- Identify areas with similar characteristics that might offer better value
- Recognize trends that might affect future housing costs in your area
"""
        
        # Save the markdown explanation
        with open(output_dir / "neighborhood_segmentation_explanation.md", "w") as f:
            f.write(explanation_md)
        
        return output_dir
    
    def get_cluster_detail_for_area(self, area_name):
        """
        Get detailed clustering information for a specific neighborhood/area.
        
        Args:
            area_name: Name of neighborhood or zipcode to look up
            
        Returns:
            Dictionary with cluster information for the area
        """
        if 'cluster' not in self.prepared_data.columns:
            print("Error: Clustering has not been performed yet")
            return None
        
        # Try to find the area in neighborhood column
        if 'neighborhood' in self.prepared_data.columns:
            area_data = self.prepared_data[self.prepared_data['neighborhood'] == area_name]
            if len(area_data) > 0:
                area_row = area_data.iloc[0]
                cluster_id = area_row['cluster']
                return self._format_area_cluster_details(area_row, cluster_id)
        
        # Try zipcode if neighborhood didn't work
        if 'zipcode' in self.prepared_data.columns:
            # Convert area_name to string for comparison with zipcode
            area_name_str = str(area_name)
            area_data = self.prepared_data[self.prepared_data['zipcode'] == area_name_str]
            if len(area_data) > 0:
                area_row = area_data.iloc[0]
                cluster_id = area_row['cluster']
                return self._format_area_cluster_details(area_row, cluster_id)
        
        print(f"Area '{area_name}' not found in clustering data")
        return None
    
    def _format_area_cluster_details(self, area_row, cluster_id):
        """Helper to format area-specific cluster details."""
        # Get cluster profile
        profile = self.cluster_profiles.loc[cluster_id]
        
        # Determine distinctive features of this area within its cluster
        area_features = {}
        for feature in self.feature_columns:
            # Skip features not in area_row
            if feature not in area_row:
                continue
                
            # Calculate how this area compares to its cluster average
            cluster_avg = profile[feature]
            area_value = area_row[feature]
            
            # Percentage difference from cluster average
            if cluster_avg != 0:
                pct_diff = (area_value - cluster_avg) / cluster_avg * 100
                area_features[feature] = {
                    'value': area_value,
                    'cluster_avg': cluster_avg,
                    'pct_diff': pct_diff
                }
        
        # Find similar areas (in same cluster)
        similar_areas = []
        if 'neighborhood' in self.prepared_data.columns:
            similar_areas = self.prepared_data[
                (self.prepared_data['cluster'] == cluster_id) & 
                (self.prepared_data['neighborhood'] != area_row.get('neighborhood', ''))
            ]['neighborhood'].tolist()[:5]  # Top 5 similar areas
        elif 'zipcode' in self.prepared_data.columns:
            similar_areas = self.prepared_data[
                (self.prepared_data['cluster'] == cluster_id) & 
                (self.prepared_data['zipcode'] != area_row.get('zipcode', ''))
            ]['zipcode'].tolist()[:5]  # Top 5 similar areas
        
        # Format the result
        result = {
            'area_name': area_row.get('neighborhood', area_row.get('zipcode', 'Unknown')),
            'cluster_id': int(cluster_id),
            'cluster_label': profile['cluster_label'],
            'features': area_features,
            'similar_areas': similar_areas,
            'cluster_size': int(profile['neighborhood_count'])
        }
        
        return result

    def run_full_analysis(self):
        """
        Run the complete clustering analysis pipeline.
        
        Returns:
            Path to output visualizations
        """
        print("Running full K-means clustering analysis...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Prepare features
        if not self.prepare_features():
            return None
        
        # Find optimal number of clusters
        self.find_optimal_k()
        
        # Perform clustering
        self.perform_clustering()
        
        # Generate visualizations
        output_dir = self.visualize_clusters()
        
        print(f"Analysis complete! Visualizations saved to {output_dir}")
        return output_dir


class RandomForestPredictor:
    """
    Random Forest regression implementation for property price prediction.
    
    This class builds a predictive model to estimate property values based on
    neighborhood characteristics, Airbnb metrics, and other features.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize with path to processed data directory.
        
        Args:
            data_dir: Path to processed data directory (optional)
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[2]
            self.data_dir = self.project_root / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        # Initialize attributes
        self.data = None
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.predictions = None
        self.model_metrics = {}
        self.prediction_data = None
        self.feature_columns = [
            'airbnb_count',
            'airbnb_density',
            'median_airbnb_price',
            'median_income',
            'population',
            '1_bedroom_rent',
            '2_bedroom_rent',
            '3_bedroom_rent'
        ]
        self.target_column = 'median_property_price'
    
    def load_data(self):
        """Load the processed data for price prediction modeling."""
        try:
            # Try to load the main merged dataset
            data_path = self.data_dir / "miami_dade_merged_data.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                print(f"Loaded {len(self.data)} neighborhoods from {data_path}")
                
                # Check for zipcode summary which might have additional metrics
                zip_summary_path = self.data_dir / "merged" / "zipcode_summary.csv"
                if zip_summary_path.exists():
                    zip_data = pd.read_csv(zip_summary_path)
                    # If we have more data in the zipcode summary, use that instead
                    if len(zip_data) > len(self.data):
                        self.data = zip_data
                        print(f"Using more comprehensive zipcode data with {len(self.data)} areas")
                
                return True
            else:
                print(f"Warning: Could not find dataset at {data_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_features(self):
        """Prepare and scale features for prediction."""
        if self.data is None:
            if not self.load_data():
                return False
        
        # Check if target column exists
        if self.target_column not in self.data.columns:
            print(f"Error: Target column '{self.target_column}' not found in data")
            return False
        
        # Identify available features from our desired list
        self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]
        
        if len(self.feature_columns) < 2:
            print("Warning: Not enough features available for meaningful prediction")
            # Try to identify any numeric columns we could use
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove ID columns, target column or other non-meaningful numerics
            exclude_terms = ['id', 'code', 'zip', 'latitude', 'longitude', self.target_column]
            numeric_cols = [col for col in numeric_cols if not any(term in col.lower() for term in exclude_terms)]
            
            if len(numeric_cols) >= 2:
                self.feature_columns = numeric_cols
                print(f"Using available numeric columns for prediction: {self.feature_columns}")
            else:
                print("Error: Insufficient numeric data for prediction")
                return False
        
        # Drop rows with missing values in feature columns or target column
        columns_to_check = self.feature_columns + [self.target_column]
        complete_data = self.data.dropna(subset=columns_to_check)
        
        if len(complete_data) < 5:
            print("Error: Too few complete data points for prediction model")
            return False
        
        print(f"Prepared {len(complete_data)} neighborhoods with {len(self.feature_columns)} features for prediction")
        
        # Split data into features and target
        X = complete_data[self.feature_columns]
        y = complete_data[self.target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Store original data with same indices for later reference
        self.prediction_data = complete_data
        
        print(f"Training set: {self.X_train.shape[0]} samples, Test set: {self.X_test.shape[0]} samples")
        return True
    
    def train_model(self, n_estimators=100, max_depth=None):
        """Train a Random Forest regression model for price prediction."""
        if self.X_train is None or self.y_train is None:
            if not self.prepare_features():
                return False
        
        print(f"Training Random Forest model with {n_estimators} trees...")
        
        # Initialize and train the model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Model training complete. Top 3 important features:")
        print(self.feature_importance.head(3))
        
        return True
    
    def evaluate_model(self):
        """Evaluate the model performance on the test set."""
        if self.model is None:
            print("Error: Model has not been trained yet")
            return False
        
        # Make predictions on the test set
        self.predictions = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, self.predictions)
        
        # Store metrics
        self.model_metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mean_absolute_error': np.mean(np.abs(self.predictions - self.y_test)),
            'mean_absolute_percentage_error': np.mean(np.abs((self.y_test - self.predictions) / self.y_test)) * 100
        }
        
        print(f"Model Evaluation Metrics:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  Mean Absolute Error: ${self.model_metrics['mean_absolute_error']:,.2f}")
        print(f"  Mean Absolute Percentage Error: {self.model_metrics['mean_absolute_percentage_error']:.2f}%")
        
        return self.model_metrics
    
    def predict_price(self, feature_values):
        """Make a property price prediction for a given set of feature values."""
        if self.model is None:
            print("Error: Model has not been trained yet")
            return None
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(feature_values, dict):
            feature_values = pd.DataFrame([feature_values])
        
        # Ensure we have all required features
        missing_features = [f for f in self.feature_columns if f not in feature_values.columns]
        if missing_features:
            print(f"Error: Missing required features: {missing_features}")
            return None
        
        # Extract only the features we need and in the right order
        X_pred = feature_values[self.feature_columns]
        
        # Scale the features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make the prediction
        predicted_price = self.model.predict(X_pred_scaled)[0]
        
        return predicted_price
    
    def visualize_model(self):
        """Create visualizations of the model results and performance."""
        if self.model is None or self.predictions is None:
            print("Error: Model has not been trained or evaluated yet")
            return None
        
        # Create output directory
        output_dir = self.project_root / "visualizations" / "prediction"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Feature Importance Plot
        plt.figure(figsize=(12, 8))
        
        # Sort by importance
        sorted_idx = self.model.feature_importances_.argsort()
        sorted_features = [self.feature_columns[i] for i in sorted_idx]
        
        # Create horizontal bar chart
        plt.barh(range(len(sorted_idx)), self.model.feature_importances_[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [f.replace('_', ' ').title() for f in sorted_features])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance for Property Price Prediction')
        plt.tight_layout()
        
        # Save the plot
        feature_importance_path = output_dir / "feature_importance.png"
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Actual vs Predicted Scatter Plot
        plt.figure(figsize=(10, 8))
        
        # Scatter plot with ideal prediction line
        plt.scatter(self.y_test, self.predictions, alpha=0.6, color='blue')
        max_val = max(max(self.y_test), max(self.predictions))
        min_val = min(min(self.y_test), min(self.predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Property Price ($)')
        plt.ylabel('Predicted Property Price ($)')
        plt.title('Actual vs Predicted Property Prices')
        
        # Add R² value annotation
        plt.annotate(f"R² = {self.model_metrics['r2']:.4f}", 
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save the plot
        predicted_actual_path = output_dir / "predicted_vs_actual.png"
        plt.savefig(predicted_actual_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Error Distribution
        plt.figure(figsize=(10, 6))
        
        # Calculate errors
        errors = self.y_test - self.predictions
        
        # Create histogram
        plt.hist(errors, bins=20, alpha=0.6, color='teal')
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        
        # Add mean error line
        plt.axvline(errors.mean(), color='r', linestyle='--', linewidth=2)
        plt.annotate(f'Mean Error: ${errors.mean():,.2f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Save the plot
        error_dist_path = output_dir / "error_distribution.png"
        plt.savefig(error_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create Markdown explanation of the model
        explanation_md = f"""# Property Price Prediction Model

## Methodology
This analysis uses Random Forest regression to predict property prices in Miami-Dade neighborhoods based on 
real housing metrics, Airbnb data, and demographic information.

## Model Performance
- **R² Score**: {self.model_metrics['r2']:.4f} (higher is better, 1.0 is perfect)
- **Root Mean Squared Error**: ${self.model_metrics['rmse']:,.2f}
- **Mean Absolute Error**: ${self.model_metrics['mean_absolute_error']:,.2f}
- **Mean Absolute Percentage Error**: {self.model_metrics['mean_absolute_percentage_error']:.2f}%

## Key Price Drivers
"""
        
        # Add top features and their importance
        for idx, row in self.feature_importance.head(5).iterrows():
            feature_name = row['feature'].replace('_', ' ').title()
            importance = row['importance'] * 100  # Convert to percentage
            explanation_md += f"- **{feature_name}**: {importance:.2f}% importance\n"
        
        explanation_md += "\n## Implications\n"
        explanation_md += """### For Policymakers
- Focus regulations on the features that most strongly impact property prices
- Use this model to predict how policy changes might affect neighborhood affordability
- Identify areas where prices are significantly higher than model predictions for further investigation

### For Investors
- Target properties with characteristics that the model identifies as undervalued
- Use predicted price vs. actual price to find potential investment opportunities
- Prioritize neighborhoods where features suggest prices will appreciate

### For Residents
- Understand how different features affect the value of your property
- When house hunting, identify properties that might be undervalued
- Recognize which amenities or neighborhood features provide the best value
"""
        
        # Save the markdown explanation
        with open(output_dir / "property_price_model_explanation.md", "w") as f:
            f.write(explanation_md)
        
        return output_dir
    
    def save_model(self):
        """Save the trained model for future use."""
        if self.model is None:
            print("Error: No trained model to save")
            return False
        
        # Create directory for models
        models_dir = self.project_root / "models" / "saved"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model using pickle
        import pickle
        
        model_path = models_dir / "property_price_predictor.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance,
                'metrics': self.model_metrics,
                'target_column': self.target_column
            }, f)
        
        print(f"Model saved to {model_path}")
        return True
    
    def load_model(self):
        """Load a previously saved model."""
        # Create directory path for models
        models_dir = self.project_root / "models" / "saved"
        model_path = models_dir / "property_price_predictor.pkl"
        
        if not model_path.exists():
            print(f"Error: No saved model found at {model_path}")
            return False
        
        # Load model using pickle
        import pickle
        
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_columns = saved_data['feature_columns']
                self.feature_importance = saved_data['feature_importance']
                self.model_metrics = saved_data['metrics']
                self.target_column = saved_data['target_column']
            
            print(f"Model loaded from {model_path}")
            print(f"Model features: {self.feature_columns}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def run_full_analysis(self):
        """Run the complete price prediction pipeline."""
        print("Running property price prediction analysis...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Prepare features and split data
        if not self.prepare_features():
            return None
        
        # Train the model
        if not self.train_model():
            return None
        
        # Evaluate the model
        self.evaluate_model()
        
        # Create visualizations
        output_dir = self.visualize_model()
        
        # Save the model for future use
        self.save_model()
        
        print(f"Analysis complete! Visualizations saved to {output_dir}")
        return output_dir


class MonteCarloInvestmentSimulator:
    """
    Monte Carlo simulation for real estate investment risk assessment.
    
    This class models potential returns and risks for real estate investments
    in different Miami-Dade neighborhoods using historical data and market parameters.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize with path to processed data directory.
        
        Args:
            data_dir: Path to processed data directory (optional)
        """
        if data_dir is None:
            self.project_root = Path(__file__).resolve().parents[2]
            self.data_dir = self.project_root / "data" / "processed"
        else:
            self.data_dir = Path(data_dir)
        
        # Initialize attributes
        self.data = None
        self.property_data = None
        self.airbnb_data = None
        self.neighborhood_data = None
        self.simulation_results = {}
        self.current_simulation = {}
        
        # Default simulation parameters
        self.default_params = {
            'num_simulations': 1000,
            'investment_horizon': 5,
            'initial_investment': 500000,
            'property_type': 'residential',
            'rental_strategy': 'long_term',  # or 'short_term' for Airbnb
            'interest_rate_mean': 0.05,
            'interest_rate_std': 0.01,
            'property_appreciation_mean': 0.04,
            'property_appreciation_std': 0.03,
            'vacancy_rate_mean': 0.08,
            'vacancy_rate_std': 0.04,
            'maintenance_cost_pct': 0.01,
            'property_tax_rate': 0.02,
            'insurance_rate': 0.005,
            'management_fee_pct': 0.1,
            'closing_costs_pct': 0.03,
            'down_payment_pct': 0.25,
            'loan_term_years': 30
        }
    
    def load_data(self):
        """Load the processed data for simulation."""
        try:
            # Load Miami-Dade merged dataset for neighborhood metrics
            data_path = self.data_dir / "miami_dade_merged_data.csv"
            if data_path.exists():
                self.data = pd.read_csv(data_path)
                print(f"Loaded {len(self.data)} neighborhoods for simulation")
                
                # Also check for more specific property and rental data
                property_path = self.data_dir / "property" / "processed_property_data.csv"
                if property_path.exists():
                    self.property_data = pd.read_csv(property_path)
                    print(f"Loaded {len(self.property_data)} property records for detailed modeling")
                
                # Load Airbnb data for short-term rental modeling
                airbnb_path = self.data_dir / "airbnb" / "processed_airbnb_data.csv"
                if airbnb_path.exists():
                    self.airbnb_data = pd.read_csv(airbnb_path)
                    print(f"Loaded {len(self.airbnb_data)} Airbnb listings for short-term rental modeling")
                
                return True
            else:
                print(f"Warning: Could not find dataset at {data_path}")
                return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _get_neighborhood_params(self, neighborhood=None, zipcode=None):
        """Extract specific parameters for a neighborhood or zipcode."""
        if self.data is None:
            if not self.load_data():
                return None
        
        # Filter by neighborhood or zipcode
        if neighborhood is not None:
            # Try exact match first
            mask = self.data['neighborhood'].str.lower() == neighborhood.lower()
            if not mask.any():
                # Try partial match
                mask = self.data['neighborhood'].str.lower().str.contains(neighborhood.lower())
            
            if not mask.any():
                print(f"Warning: Neighborhood '{neighborhood}' not found. Using aggregate data.")
                # Use average across neighborhoods
                neighborhood_data = self.data.mean(numeric_only=True).to_dict()
                neighborhood_data['neighborhood'] = 'Miami-Dade Average'
                return neighborhood_data
            else:
                neighborhood_data = self.data.loc[mask].iloc[0].to_dict()
                print(f"Using data for {neighborhood_data['neighborhood']}")
                return neighborhood_data
        
        elif zipcode is not None:
            # Check for zipcode column
            if 'zipcode' in self.data.columns:
                mask = self.data['zipcode'] == zipcode
                if mask.any():
                    neighborhood_data = self.data.loc[mask].iloc[0].to_dict()
                    print(f"Using data for zipcode {zipcode}")
                    return neighborhood_data
            
            print(f"Warning: Zipcode {zipcode} not found. Using aggregate data.")
            # Use average across neighborhoods
            neighborhood_data = self.data.mean(numeric_only=True).to_dict()
            neighborhood_data['zipcode'] = zipcode
            neighborhood_data['neighborhood'] = f'Zipcode {zipcode} Average'
            return neighborhood_data
        
        # If no specific location provided, use average
        neighborhood_data = self.data.mean(numeric_only=True).to_dict()
        neighborhood_data['neighborhood'] = 'Miami-Dade Average'
        return neighborhood_data
    
    def run_simulation(self, neighborhood=None, zipcode=None, **params):
        """Run Monte Carlo simulation for a specific neighborhood or zipcode.
        
        Args:
            neighborhood: Name of neighborhood (optional)
            zipcode: Zipcode as string or int (optional)
            **params: Optional parameters to override defaults:
                - num_simulations: Number of Monte Carlo runs
                - investment_horizon: Years to simulate
                - initial_investment: Purchase price
                - property_type: 'residential', 'condo', etc.
                - rental_strategy: 'long_term' or 'short_term'
                - interest_rate_mean: Mean interest rate
                - interest_rate_std: Standard deviation of interest rate
                - property_appreciation_mean: Mean annual appreciation
                - property_appreciation_std: Std dev of appreciation
                - vacancy_rate_mean: Mean vacancy rate
                - vacancy_rate_std: Std dev of vacancy rate
                - And others...
        
        Returns:
            Dictionary of simulation results
        """
        # Update default parameters with any provided
        sim_params = self.default_params.copy()
        sim_params.update(params)
        
        # Get neighborhood-specific data
        neighborhood_data = self._get_neighborhood_params(neighborhood, zipcode)
        
        if neighborhood_data is None:
            print("Error: Cannot run simulation without neighborhood data")
            return None
        
        # Extract neighborhood metrics for the simulation
        sim_params['location'] = neighborhood_data.get('neighborhood', 'Unknown')
        
        # Use real neighborhood data to adjust simulation parameters where available
        if 'median_property_price' in neighborhood_data:
            if 'initial_investment' not in params:  # Only override if not explicitly provided
                sim_params['initial_investment'] = neighborhood_data['median_property_price']
        
        # Use real rental data if available
        if 'median_rent' in neighborhood_data:
            sim_params['monthly_rent_initial'] = neighborhood_data['median_rent']
        elif '2_bedroom_rent' in neighborhood_data:
            sim_params['monthly_rent_initial'] = neighborhood_data['2_bedroom_rent']
        else:
            # Estimate rent as percentage of property value
            sim_params['monthly_rent_initial'] = sim_params['initial_investment'] * 0.007  # 0.7% monthly
        
        # If short-term rental strategy, use Airbnb data
        if sim_params['rental_strategy'] == 'short_term' and 'median_airbnb_price' in neighborhood_data:
            # Adjust for average occupancy
            if 'airbnb_occupancy_rate' in neighborhood_data:
                occupancy = neighborhood_data['airbnb_occupancy_rate']
            else:
                occupancy = 0.65  # Typical Airbnb occupancy
                
            # Calculate monthly revenue: daily rate * days per month * occupancy
            sim_params['monthly_rent_initial'] = neighborhood_data['median_airbnb_price'] * 30 * occupancy
            
            # Short-term rentals have higher vacancy and management costs
            if 'vacancy_rate_mean' not in params:
                sim_params['vacancy_rate_mean'] = 0.35  # Higher vacancy for short-term
            if 'management_fee_pct' not in params:
                sim_params['management_fee_pct'] = 0.2  # Higher management fees for short-term
        
        # Adjust appreciation rates based on neighborhood trends if available
        if 'historical_appreciation' in neighborhood_data:
            sim_params['property_appreciation_mean'] = neighborhood_data['historical_appreciation']
        elif 'price_growth_rate' in neighborhood_data:
            sim_params['property_appreciation_mean'] = neighborhood_data['price_growth_rate']
        
        # Run the simulations
        num_sims = sim_params['num_simulations']
        years = sim_params['investment_horizon']
        
        # Arrays to store results
        roi_results = np.zeros(num_sims)
        npv_results = np.zeros(num_sims)
        irr_results = np.zeros(num_sims)
        exit_values = np.zeros(num_sims)
        cashflows = np.zeros((num_sims, years+1))  # +1 for initial investment
        
        print(f"Running {num_sims} simulations for {sim_params['location']} over {years} years...")
        
        for i in range(num_sims):
            # Generate random variables for this simulation
            interest_rate = np.random.normal(
                sim_params['interest_rate_mean'],
                sim_params['interest_rate_std']
            )
            
            # Ensure reasonable bounds
            interest_rate = max(0.01, min(0.12, interest_rate))
            
            # Initial investment is downpayment + closing costs
            initial_investment = sim_params['initial_investment']
            down_payment = initial_investment * sim_params['down_payment_pct']
            closing_costs = initial_investment * sim_params['closing_costs_pct']
            initial_cash_outflow = down_payment + closing_costs
            
            # Loan amount and monthly payment
            loan_amount = initial_investment - down_payment
            monthly_rate = interest_rate / 12
            loan_term_months = sim_params['loan_term_years'] * 12
            monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term_months) / \
                             ((1 + monthly_rate) ** loan_term_months - 1) if monthly_rate > 0 else \
                             loan_amount / loan_term_months
            
            # Initial monthly rent
            monthly_rent = sim_params['monthly_rent_initial']
            
            # Simulate for each year
            yearly_cashflows = [-initial_cash_outflow]  # Initial investment (down payment + closing costs)
            cumulative_cashflow = -initial_cash_outflow
            property_value = initial_investment
            remaining_loan = loan_amount
            
            for year in range(1, years+1):
                # Random property appreciation for this year
                appreciation_rate = np.random.normal(
                    sim_params['property_appreciation_mean'],
                    sim_params['property_appreciation_std']
                )
                # Ensure reasonable bounds
                appreciation_rate = max(-0.1, min(0.2, appreciation_rate))
                
                # Update property value
                property_value *= (1 + appreciation_rate)
                
                # Random vacancy rate for this year
                vacancy_rate = np.random.normal(
                    sim_params['vacancy_rate_mean'],
                    sim_params['vacancy_rate_std']
                )
                # Ensure reasonable bounds
                vacancy_rate = max(0, min(0.5, vacancy_rate))
                
                # Increase rent annually (assume rent increases with inflation)
                monthly_rent *= (1 + sim_params.get('rent_growth_rate', 0.03))
                
                # Calculate annual revenue
                annual_revenue = monthly_rent * 12 * (1 - vacancy_rate)
                
                # Calculate annual expenses
                annual_mortgage = monthly_payment * 12
                annual_maintenance = property_value * sim_params['maintenance_cost_pct']
                annual_property_tax = property_value * sim_params['property_tax_rate']
                annual_insurance = property_value * sim_params['insurance_rate']
                annual_management = annual_revenue * sim_params['management_fee_pct']
                
                # Calculate loan principal reduction for the year (rough estimate)
                interest_portion = remaining_loan * interest_rate
                principal_reduction = annual_mortgage - interest_portion
                remaining_loan -= principal_reduction
                
                # Total expenses
                annual_expenses = annual_mortgage + annual_maintenance + \
                                annual_property_tax + annual_insurance + annual_management
                
                # Net cashflow for the year
                annual_cashflow = annual_revenue - annual_expenses
                
                # Add to yearly cashflows array
                yearly_cashflows.append(annual_cashflow)
                cumulative_cashflow += annual_cashflow
            
            # Calculate exit value (property value minus remaining loan and selling costs)
            selling_costs = property_value * 0.06  # Typical selling costs
            exit_value = property_value - remaining_loan - selling_costs
            
            # Add final exit value to last year's cashflow
            yearly_cashflows[-1] += exit_value
            
            # Calculate ROI
            total_return = exit_value + cumulative_cashflow - (-yearly_cashflows[0])  # Subtract negative initial investment
            roi = total_return / (-yearly_cashflows[0]) * 100  # Convert to percentage
            
            # Calculate NPV
            npv = np.npv(sim_params.get('discount_rate', 0.07), yearly_cashflows)
            
            # Calculate IRR
            try:
                irr = np.irr(yearly_cashflows)
            except:
                irr = np.nan  # If IRR calculation fails
            
            # Store results
            roi_results[i] = roi
            npv_results[i] = npv
            irr_results[i] = irr * 100 if not np.isnan(irr) else 0  # Convert to percentage
            exit_values[i] = exit_value
            cashflows[i, :] = yearly_cashflows
        
        # Store simulation results
        self.current_simulation = {
            'params': sim_params,
            'location': sim_params['location'],
            'roi': {
                'mean': np.mean(roi_results),
                'median': np.median(roi_results),
                'std': np.std(roi_results),
                'min': np.min(roi_results),
                'max': np.max(roi_results),
                'percentiles': {
                    '10': np.percentile(roi_results, 10),
                    '25': np.percentile(roi_results, 25),
                    '75': np.percentile(roi_results, 75),
                    '90': np.percentile(roi_results, 90)
                },
                'values': roi_results
            },
            'npv': {
                'mean': np.mean(npv_results),
                'median': np.median(npv_results),
                'std': np.std(npv_results),
                'min': np.min(npv_results),
                'max': np.max(npv_results),
                'percentiles': {
                    '10': np.percentile(npv_results, 10),
                    '25': np.percentile(npv_results, 25),
                    '75': np.percentile(npv_results, 75),
                    '90': np.percentile(npv_results, 90)
                },
                'values': npv_results
            },
            'irr': {
                'mean': np.mean(irr_results),
                'median': np.median(irr_results),
                'std': np.std(irr_results),
                'min': np.min(irr_results),
                'max': np.max(irr_results),
                'percentiles': {
                    '10': np.percentile(irr_results, 10),
                    '25': np.percentile(irr_results, 25),
                    '75': np.percentile(irr_results, 75),
                    '90': np.percentile(irr_results, 90)
                },
                'values': irr_results
            },
            'exit_value': {
                'mean': np.mean(exit_values),
                'median': np.median(exit_values),
                'std': np.std(exit_values),
                'min': np.min(exit_values),
                'max': np.max(exit_values),
                'percentiles': {
                    '10': np.percentile(exit_values, 10),
                    '25': np.percentile(exit_values, 25),
                    '75': np.percentile(exit_values, 75),
                    '90': np.percentile(exit_values, 90)
                },
                'values': exit_values
            },
            'cashflows': {
                'mean': np.mean(cashflows, axis=0),
                'median': np.median(cashflows, axis=0),
                'std': np.std(cashflows, axis=0),
                'all': cashflows
            },
            'risk_metrics': {
                'probability_negative_roi': (roi_results < 0).mean() * 100,
                'probability_positive_npv': (npv_results > 0).mean() * 100,
                'sharpe_ratio': (np.mean(roi_results) - sim_params.get('risk_free_rate', 0.03)) / np.std(roi_results) if np.std(roi_results) > 0 else 0,
                'downside_deviation': np.sqrt(np.mean(np.minimum(roi_results - sim_params.get('target_return', 8), 0) ** 2)),
                'sortino_ratio': (np.mean(roi_results) - sim_params.get('risk_free_rate', 0.03)) / np.sqrt(np.mean(np.minimum(roi_results - sim_params.get('target_return', 8), 0) ** 2)) if np.mean(np.minimum(roi_results - sim_params.get('target_return', 8), 0) ** 2) > 0 else 0
            }
        }
        
        # Store in simulation results dictionary
        if sim_params['location'] not in self.simulation_results:
            self.simulation_results[sim_params['location']] = []
        
        self.simulation_results[sim_params['location']].append(self.current_simulation)
        
        print(f"Simulation complete for {sim_params['location']}.")
        print(f"Mean ROI: {self.current_simulation['roi']['mean']:.2f}%")
        print(f"Mean IRR: {self.current_simulation['irr']['mean']:.2f}%")
        print(f"Mean Exit Value: ${self.current_simulation['exit_value']['mean']:,.2f}")
        print(f"Probability of Negative ROI: {self.current_simulation['risk_metrics']['probability_negative_roi']:.2f}%")
        
        return self.current_simulation
    
    def visualize_simulations(self):
        """Create visualizations for the most recent simulation results."""
        if not self.current_simulation:
            print("Error: No simulation results to visualize")
            return None
        
        # Create output directory
        output_dir = self.project_root / "visualizations" / "investment"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating visualizations for {self.current_simulation['location']}...")
        
        # Extract data for plots
        roi_values = self.current_simulation['roi']['values']
        irr_values = self.current_simulation['irr']['values']
        npv_values = self.current_simulation['npv']['values']
        exit_values = self.current_simulation['exit_value']['values']
        cashflows = self.current_simulation['cashflows']
        location = self.current_simulation['location']
        params = self.current_simulation['params']
        years = params['investment_horizon']
        
        # 1. ROI Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(roi_values, kde=True, color='teal', bins=30)
        plt.axvline(np.percentile(roi_values, 50), color='red', linestyle='--', label=f'Median: {np.percentile(roi_values, 50):.2f}%')
        plt.axvline(np.percentile(roi_values, 10), color='orange', linestyle='--', label=f'10th Percentile: {np.percentile(roi_values, 10):.2f}%')
        plt.axvline(np.percentile(roi_values, 90), color='green', linestyle='--', label=f'90th Percentile: {np.percentile(roi_values, 90):.2f}%')
        plt.xlabel('Return on Investment (%)')
        plt.ylabel('Frequency')
        plt.title(f'ROI Distribution for {location} ({years}-Year Horizon)')
        plt.legend()
        plt.tight_layout()
        roi_dist_path = output_dir / f"{location.replace(' ', '_')}_roi_distribution.png"
        plt.savefig(roi_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cashflow Projection with Confidence Intervals
        plt.figure(figsize=(12, 6))
        year_labels = ['Initial'] + [f'Year {y}' for y in range(1, years+1)]
        x = np.arange(len(year_labels))
        
        # Mean and median cash flows
        plt.plot(x, cashflows['mean'], marker='o', linestyle='-', color='blue', label='Mean Cashflow')
        plt.plot(x, cashflows['median'], marker='s', linestyle='--', color='green', label='Median Cashflow')
        
        # 25th-75th percentile range
        all_cashflows = cashflows['all']
        lower_percentile = np.percentile(all_cashflows, 25, axis=0)
        upper_percentile = np.percentile(all_cashflows, 75, axis=0)
        plt.fill_between(x, lower_percentile, upper_percentile, alpha=0.2, color='blue', label='25th-75th Percentile')
        
        # Add annotations for initial investment and exit value
        plt.annotate(f"Initial: ${cashflows['mean'][0]:,.0f}", 
                    xy=(0, cashflows['mean'][0]), 
                    xytext=(0, cashflows['mean'][0] - 20000),
                    arrowprops=dict(arrowstyle="->", color='black'))
        
        plt.annotate(f"Exit: ${cashflows['mean'][-1]:,.0f}", 
                    xy=(len(year_labels)-1, cashflows['mean'][-1]), 
                    xytext=(len(year_labels)-1.5, cashflows['mean'][-1] + 20000),
                    arrowprops=dict(arrowstyle="->", color='black'))
        
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xticks(x, year_labels, rotation=45)
        plt.ylabel('Cashflow ($)')
        plt.title(f'Projected Cashflows for {location} Investment')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        cashflow_path = output_dir / f"{location.replace(' ', '_')}_cashflow_projection.png"
        plt.savefig(cashflow_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. IRR vs ROI Scatter Plot (Risk vs Return)
        plt.figure(figsize=(10, 6))
        plt.scatter(roi_values, irr_values, alpha=0.5, c=exit_values, cmap='viridis')
        plt.colorbar(label='Exit Value ($)')
        plt.xlabel('Return on Investment (%)')
        plt.ylabel('Internal Rate of Return (%)')
        plt.title(f'Risk-Return Profile for {location} Investment')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        risk_return_path = output_dir / f"{location.replace(' ', '_')}_risk_return.png"
        plt.savefig(risk_return_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Risk Metrics Comparison Radar Chart
        plt.figure(figsize=(8, 8))
        
        # Define risk metrics for radar chart
        risk_metrics = [
            # Convert ROI to 0-1 scale for the radar chart
            max(0, min(1, self.current_simulation['roi']['mean'] / 100)),
            # Normalize IRR to 0-1 scale
            max(0, min(1, self.current_simulation['irr']['mean'] / 20)),  # Assuming 20% IRR is excellent
            # Normalize cash-on-cash return
            max(0, min(1, abs(cashflows['mean'][1:].mean() / cashflows['mean'][0]))),
            # Probability of positive NPV (already in 0-100 scale, convert to 0-1)
            self.current_simulation['risk_metrics']['probability_positive_npv'] / 100,
            # Invert probability of negative ROI (so higher is better)
            1 - (self.current_simulation['risk_metrics']['probability_negative_roi'] / 100),
            # Normalize Sharpe ratio
            max(0, min(1, self.current_simulation['risk_metrics']['sharpe_ratio'] / 3))  # Assuming 3 is excellent
        ]
        
        # Labels for the risk metrics
        categories = ['ROI', 'IRR', 'Cash-on-Cash', 'NPV Probability', 'Risk Avoidance', 'Risk-Adjusted Return']
        
        # Create radar chart
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, size=12)
        
        # Plot risk metrics
        risk_metrics += risk_metrics[:1]  # Close the loop
        ax.plot(angles, risk_metrics, linewidth=2, linestyle='solid', label=location)
        ax.fill(angles, risk_metrics, alpha=0.25)
        
        # Add value annotations
        for i in range(N):
            ax.annotate(f"{risk_metrics[i]:.2f}", 
                      xy=(angles[i], risk_metrics[i]),
                      xytext=(angles[i], risk_metrics[i] + 0.1),
                      ha='center')
        
        plt.title(f'Investment Quality Metrics for {location}')
        plt.tight_layout()
        radar_path = output_dir / f"{location.replace(' ', '_')}_risk_radar.png"
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Create Markdown explanation of the simulation
        explanation_md = f"""# Real Estate Investment Simulation for {location}

## Simulation Parameters
- **Investment Amount**: ${params['initial_investment']:,.2f}
- **Rental Strategy**: {params['rental_strategy'].replace('_', ' ').title()}
- **Down Payment**: {params['down_payment_pct']*100:.0f}%
- **Investment Horizon**: {years} years
- **Loan Term**: {params['loan_term_years']} years
- **Interest Rate**: {params['interest_rate_mean']*100:.2f}% ± {params['interest_rate_std']*100:.2f}%
- **Appreciation Rate**: {params['property_appreciation_mean']*100:.2f}% ± {params['property_appreciation_std']*100:.2f}%

## Return Metrics (Mean Values)
- **Return on Investment (ROI)**: {self.current_simulation['roi']['mean']:.2f}%
- **Internal Rate of Return (IRR)**: {self.current_simulation['irr']['mean']:.2f}%
- **Net Present Value (NPV)**: ${self.current_simulation['npv']['mean']:,.2f}
- **Exit Value**: ${self.current_simulation['exit_value']['mean']:,.2f}

## Risk Metrics
- **Probability of Negative ROI**: {self.current_simulation['risk_metrics']['probability_negative_roi']:.2f}%
- **Probability of Positive NPV**: {self.current_simulation['risk_metrics']['probability_positive_npv']:.2f}%
- **Sharpe Ratio**: {self.current_simulation['risk_metrics']['sharpe_ratio']:.2f}
- **Sortino Ratio**: {self.current_simulation['risk_metrics']['sortino_ratio']:.2f}

## Return Distribution
- **ROI Percentiles**:
  - 10th Percentile: {self.current_simulation['roi']['percentiles']['10']:.2f}%
  - 25th Percentile: {self.current_simulation['roi']['percentiles']['25']:.2f}%
  - 50th Percentile (Median): {self.current_simulation['roi']['median']:.2f}%
  - 75th Percentile: {self.current_simulation['roi']['percentiles']['75']:.2f}%
  - 90th Percentile: {self.current_simulation['roi']['percentiles']['90']:.2f}%

## Investment Implications

### For Investors
- **Risk Profile**: {self._risk_profile()}
- **Investment Horizon**: {self._horizon_recommendation()}
- **Strategy Fit**: {self._strategy_fit()}

### Market Insights
- This simulation incorporates real data from Miami-Dade County housing and rental markets.
- Property values and rent prices are based on actual neighborhood metrics.
- Risk metrics account for market volatility and property-specific factors.
"""
        
        # Save the markdown explanation
        with open(output_dir / f"{location.replace(' ', '_')}_investment_simulation.md", "w") as f:
            f.write(explanation_md)
        
        print(f"Visualizations completed and saved to {output_dir}")
        return output_dir
    
    def _risk_profile(self):
        """Generate risk profile description based on simulation results."""
        roi_std = self.current_simulation['roi']['std']
        prob_negative = self.current_simulation['risk_metrics']['probability_negative_roi']
        mean_roi = self.current_simulation['roi']['mean']
        
        if prob_negative < 5 and roi_std < 10:
            return "Conservative - Low volatility with minimal risk of loss"
        elif prob_negative < 15 and roi_std < 20 and mean_roi > 30:
            return "Balanced - Moderate risk with good potential returns"
        elif prob_negative > 25 or roi_std > 30:
            return "Aggressive - Higher volatility with potential for significant gains or losses"
        else:
            return "Moderate - Average risk level for real estate investment"
    
    def _horizon_recommendation(self):
        """Generate investment horizon recommendation."""
        years = self.current_simulation['params']['investment_horizon']
        cashflows = self.current_simulation['cashflows']['mean']
        break_even_year = 0
        
        # Find break-even point
        cumulative = cashflows[0]
        for i in range(1, len(cashflows)):
            cumulative += cashflows[i]
            if cumulative >= 0 and break_even_year == 0:
                break_even_year = i
        
        if break_even_year == 0:
            return f"Long-term hold recommended (> {years} years) as ROI improves with longer horizons"
        elif break_even_year < years / 2:
            return f"Mid-term investment potential with break-even around year {break_even_year}"
        else:
            return f"Long-term investment with break-even projected in year {break_even_year}"
    
    def _strategy_fit(self):
        """Generate rental strategy recommendation."""
        strategy = self.current_simulation['params']['rental_strategy']
        
        if strategy == 'long_term':
            irr = self.current_simulation['irr']['mean']
            if irr > 12:
                return "Long-term rental strategy is optimal for this location"
            else:
                return "Consider exploring short-term rental options for potentially higher returns"
        else:  # short_term
            roi = self.current_simulation['roi']['mean']
            risk = self.current_simulation['roi']['std']
            if roi > 40 and risk < 25:
                return "Short-term rental strategy appears optimal for this location"
            else:
                return "Consider long-term rentals for more stable returns with lower management overhead"
    
    def run_full_analysis(self, neighborhood=None, rental_strategy='long_term'):
        """Run a complete simulation and visualization pipeline."""
        print(f"Running investment simulation analysis for {neighborhood or 'Miami-Dade County'}...")
        
        # Load data
        if not self.load_data():
            return None
        
        # Run the simulation
        self.run_simulation(neighborhood=neighborhood, rental_strategy=rental_strategy)
        
        # Generate visualizations
        output_dir = self.visualize_simulations()
        
        print(f"Analysis complete! Visualizations saved to {output_dir}")
        return output_dir
    
    def compare_neighborhoods(self, neighborhoods, **params):
        """Run simulations for multiple neighborhoods and compare results."""
        if not neighborhoods:
            print("Error: Please provide a list of neighborhoods to compare")
            return None
        
        # Create output directory
        output_dir = self.project_root / "visualizations" / "investment" / "comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        comparison_results = {}
        roi_means = []
        roi_stds = []
        irr_means = []
        prob_neg = []
        locations = []
        
        # Run simulations for each neighborhood
        for neighborhood in neighborhoods:
            # Run simulation for this neighborhood
            simulation = self.run_simulation(neighborhood=neighborhood, **params)
            
            if simulation:
                # Store results
                comparison_results[neighborhood] = simulation
                roi_means.append(simulation['roi']['mean'])
                roi_stds.append(simulation['roi']['std'])
                irr_means.append(simulation['irr']['mean'])
                prob_neg.append(simulation['risk_metrics']['probability_negative_roi'])
                locations.append(simulation['location'])
            else:
                print(f"Warning: Simulation failed for {neighborhood}")
        
        if not comparison_results:
            print("Error: All simulations failed")
            return None
        
        print(f"Comparing investment outcomes across {len(comparison_results)} neighborhoods...")
        
        # Create comparison visualizations
        
        # 1. ROI Comparison Bar Chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(locations, roi_means, yerr=roi_stds, capsize=5, alpha=0.7, color='skyblue')
        
        # Add data labels
        for bar, roi in zip(bars, roi_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                   f"{roi:.1f}%", ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Neighborhood')
        plt.ylabel('Mean ROI (%)')
        plt.title('Return on Investment Comparison by Neighborhood')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(min(0, min(roi_means) - 10), max(roi_means) + max(roi_stds) + 10)
        plt.tight_layout()
        roi_comp_path = output_dir / "roi_neighborhood_comparison.png"
        plt.savefig(roi_comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Risk vs Return Scatter Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(roi_stds, roi_means, s=100, alpha=0.7, c=irr_means, cmap='viridis')
        
        # Add neighborhood labels
        for i, location in enumerate(locations):
            plt.annotate(location, (roi_stds[i], roi_means[i]), 
                      xytext=(7, 0), textcoords='offset points')
        
        cbar = plt.colorbar()
        cbar.set_label('IRR (%)')
        plt.axhline(y=np.mean(roi_means), color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=np.mean(roi_stds), color='red', linestyle='--', alpha=0.3)
        plt.xlabel('Risk (ROI Standard Deviation)')
        plt.ylabel('Mean ROI (%)')
        plt.title('Risk vs. Return by Neighborhood')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        risk_return_comp_path = output_dir / "risk_return_comparison.png"
        plt.savefig(risk_return_comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Risk Metrics Summary Table
        plt.figure(figsize=(14, len(locations) * 0.5 + 2))
        plt.axis('tight')
        plt.axis('off')
        
        # Prepare table data
        headers = ['Neighborhood', 'ROI (%)', 'IRR (%)', 'Risk (%)', 'Probability of Loss (%)', 'Risk-Adjusted Return']
        table_data = []
        
        for location, roi, irr, std, prob in zip(locations, roi_means, irr_means, roi_stds, prob_neg):
            # Calculate risk-adjusted return (simplified Sharpe ratio)
            risk_adj = roi / std if std > 0 else float('inf')
            table_data.append([location, f"{roi:.2f}", f"{irr:.2f}", f"{std:.2f}", f"{prob:.2f}", f"{risk_adj:.2f}"])
        
        # Create table
        table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Highlight best options
        best_roi_idx = np.argmax(roi_means)
        lowest_risk_idx = np.argmin(roi_stds)
        lowest_prob_idx = np.argmin(prob_neg)
        best_risk_adj_idx = np.argmax([r/s if s > 0 else float('inf') for r, s in zip(roi_means, roi_stds)])
        
        # Color cells for best options
        for idx, col in [(best_roi_idx, 1), (lowest_risk_idx, 3), (lowest_prob_idx, 4), (best_risk_adj_idx, 5)]:
            cell = table[(idx+1, col)]
            cell.set_facecolor('#a8d08d')  # Light green
        
        plt.title('Investment Metrics Comparison by Neighborhood', fontsize=16, pad=20)
        plt.tight_layout()
        metrics_table_path = output_dir / "metrics_comparison_table.png"
        plt.savefig(metrics_table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Markdown comparison summary
        comparison_md = f"""# Neighborhood Investment Comparison

## Summary of Investment Outcomes

This analysis compares potential real estate investments across {len(locations)} Miami-Dade neighborhoods
 using Monte Carlo simulation with {params.get('num_simulations', self.default_params['num_simulations'])} runs.

## Key Findings

### Best Return on Investment
**{locations[best_roi_idx]}** offers the highest mean ROI at **{roi_means[best_roi_idx]:.2f}%**

### Lowest Risk
**{locations[lowest_risk_idx]}** has the lowest volatility with standard deviation of **{roi_stds[lowest_risk_idx]:.2f}%**

### Lowest Probability of Loss
**{locations[lowest_prob_idx]}** has the lowest probability of negative returns at **{prob_neg[lowest_prob_idx]:.2f}%**

### Best Risk-Adjusted Return
**{locations[best_risk_adj_idx]}** provides the best balance of risk and return

## Investment Recommendations

1. **Conservative Investors**: Consider {locations[lowest_risk_idx]} for stable returns with minimal downside risk

2. **Balanced Investors**: {locations[best_risk_adj_idx]} offers optimal risk-adjusted returns

3. **Growth-Focused Investors**: {locations[best_roi_idx]} has highest potential upside but with corresponding risk

## Simulation Parameters
"""
        
        # Add simulation parameters
        for key, value in params.items():
            if key in self.default_params:
                formatted_value = f"{value*100:.2f}%" if "rate" in key or "pct" in key else f"{value}"
                comparison_md += f"- **{key.replace('_', ' ').title()}**: {formatted_value}\n"
        
        # Save the markdown summary
        with open(output_dir / "neighborhood_comparison.md", "w") as f:
            f.write(comparison_md)
        
        print(f"Neighborhood comparison completed and saved to {output_dir}")
        return comparison_results, output_dir

# Example usage if run directly
if __name__ == "__main__":
    # Run neighborhood segmentation
    print("Running K-means neighborhood segmentation...")
    segmentation = NeighborhoodSegmentation()
    segmentation.run_full_analysis()
    
    # Run property price prediction
    print("\nRunning Random Forest property price prediction...")
    predictor = RandomForestPredictor()
    predictor.run_full_analysis()
