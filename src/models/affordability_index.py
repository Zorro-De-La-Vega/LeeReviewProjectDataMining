"""
Affordability Index Module

This module provides functionality to create and analyze an affordability index
using Principal Component Analysis (PCA) and other advanced techniques.
The affordability index combines multiple housing and economic indicators
into a composite score that measures overall housing affordability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AffordabilityIndexCreator:
    """Creates and analyzes an affordability index using PCA and other techniques."""
    
    def __init__(self, data=None):
        """
        Initialize the AffordabilityIndexCreator with data.
        
        Args:
            data (pd.DataFrame, optional): DataFrame containing housing metrics
        """
        self.data = data
        self.affordability_data = None
        self.pca_model = None
        self.scaler = None
        self.components = None
        self.explained_variance = None
        self.affordability_index = None
        self.cluster_model = None
        self.cluster_labels = None
    
    def preprocess_data(self, columns_to_use=None, dropna=True):
        """
        Preprocess the data for affordability index creation.
        
        Args:
            columns_to_use (list): List of column names to include in the index
            dropna (bool): Whether to drop rows with missing values
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            logger.error("No data provided for preprocessing")
            return None
        
        try:
            # Select columns if specified
            if columns_to_use:
                # Make sure all specified columns exist in the data
                missing_columns = [col for col in columns_to_use if col not in self.data.columns]
                if missing_columns:
                    logger.warning(f"The following columns are not in the dataset: {missing_columns}")
                    # Use only columns that exist
                    columns_to_use = [col for col in columns_to_use if col in self.data.columns]
                
                if not columns_to_use:
                    logger.error("No valid columns to use for affordability index")
                    return None
                
                data_subset = self.data[columns_to_use].copy()
            else:
                # Auto-select relevant columns based on names
                affordability_keywords = [
                    'price', 'rent', 'income', 'value', 'cost', 'afford', 
                    'mortgage', 'median', 'mean', 'average', 'ratio'
                ]
                
                numeric_columns = self.data.select_dtypes(include=np.number).columns
                selected_columns = []
                
                for col in numeric_columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in affordability_keywords):
                        selected_columns.append(col)
                
                if not selected_columns:
                    logger.warning("No columns automatically identified as affordability metrics. Using all numeric columns.")
                    selected_columns = numeric_columns.tolist()
                
                data_subset = self.data[selected_columns].copy()
            
            # Drop rows with missing values if required
            if dropna:
                data_subset = data_subset.dropna()
            
            # Store processed data
            self.affordability_data = data_subset
            
            logger.info(f"Preprocessed data: {data_subset.shape[0]} rows, {data_subset.shape[1]} columns")
            return data_subset
        
        except Exception as e:
            logger.error(f"Error in preprocessing data: {e}", exc_info=True)
            return None
    
    def create_affordability_index(self, n_components=1, columns_to_use=None):
        """
        Create an affordability index using PCA.
        
        Args:
            n_components (int): Number of principal components to use
            columns_to_use (list): List of column names to include in the index
            
        Returns:
            pd.DataFrame: DataFrame with the affordability index
        """
        try:
            # Preprocess data if not already done
            if self.affordability_data is None:
                self.preprocess_data(columns_to_use=columns_to_use)
                
            if self.affordability_data is None or self.affordability_data.empty:
                logger.error("No data available for creating affordability index")
                return None
            
            # Standardize the data
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(self.affordability_data)
            
            # Apply PCA
            self.pca_model = PCA(n_components=n_components)
            principal_components = self.pca_model.fit_transform(scaled_data)
            
            # Store PCA results
            self.components = self.pca_model.components_
            self.explained_variance = self.pca_model.explained_variance_ratio_
            
            # Create affordability index DataFrame
            index_df = pd.DataFrame()
            
            # Add identifier columns (zip code, neighborhood, etc.) if available
            id_columns = ['zip_code', 'zipcode', 'zip', 'neighborhood', 'location', 'area']
            available_id_cols = [col for col in id_columns if col in self.data.columns]
            
            if available_id_cols:
                for col in available_id_cols:
                    index_df[col] = self.data.loc[self.affordability_data.index, col].values
            
            # Add principal components
            for i in range(n_components):
                index_df[f'PC{i+1}'] = principal_components[:, i]
            
            # Calculate affordability index as the first principal component
            # Flip sign if necessary so that higher values indicate better affordability
            correlation_with_price = self._check_price_correlation(principal_components[:, 0])
            index_df['affordability_index'] = principal_components[:, 0] * (-1 if correlation_with_price > 0 else 1)
            
            # Normalize to 0-100 scale for easier interpretation
            index_df['affordability_score'] = self._normalize_to_scale(index_df['affordability_index'], 0, 100)
            
            self.affordability_index = index_df
            logger.info(f"Created affordability index with {n_components} components. Explained variance: {self.explained_variance}")
            
            # Generate summary metrics
            self._generate_affordability_summary()
            
            return index_df
        
        except Exception as e:
            logger.error(f"Error creating affordability index: {e}", exc_info=True)
            return None
    
    def analyze_component_contributions(self):
        """
        Analyze the contribution of each original variable to the principal components.
        
        Returns:
            pd.DataFrame: DataFrame with component loadings
        """
        if self.pca_model is None or self.affordability_data is None:
            logger.error("PCA model not created yet. Run create_affordability_index first.")
            return None
        
        try:
            # Create DataFrame of component loadings
            loadings = pd.DataFrame(
                self.pca_model.components_.T,
                columns=[f'PC{i+1}' for i in range(self.pca_model.n_components_)],
                index=self.affordability_data.columns
            )
            
            # Add the absolute values for easier interpretation
            for col in loadings.columns:
                loadings[f'{col}_abs'] = loadings[col].abs()
            
            # Sort by contribution to PC1
            loadings = loadings.sort_values(by='PC1_abs', ascending=False)
            
            return loadings
        
        except Exception as e:
            logger.error(f"Error analyzing component contributions: {e}", exc_info=True)
            return None
    
    def cluster_by_affordability(self, n_clusters=4):
        """
        Cluster areas by their affordability metrics.
        
        Args:
            n_clusters (int): Number of clusters to create
            
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        if self.affordability_index is None:
            logger.error("Affordability index not created yet. Run create_affordability_index first.")
            return None
        
        try:
            # Create KMeans clustering model
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            
            # Use principal components for clustering
            pc_columns = [col for col in self.affordability_index.columns if col.startswith('PC')]
            cluster_data = self.affordability_index[pc_columns].values
            
            # Fit model and predict clusters
            self.cluster_labels = self.cluster_model.fit_predict(cluster_data)
            
            # Add cluster labels to affordability index
            result_df = self.affordability_index.copy()
            result_df['affordability_cluster'] = self.cluster_labels
            
            # Calculate cluster metrics
            silhouette = silhouette_score(cluster_data, self.cluster_labels)
            logger.info(f"Created {n_clusters} affordability clusters. Silhouette score: {silhouette:.3f}")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error clustering by affordability: {e}", exc_info=True)
            return None
    
    def analyze_clusters(self):
        """
        Analyze the characteristics of each affordability cluster.
        
        Returns:
            pd.DataFrame: DataFrame with cluster profiles
        """
        if self.cluster_labels is None or self.affordability_index is None:
            logger.error("Clusters not created yet. Run cluster_by_affordability first.")
            return None
        
        try:
            # Create cluster profile DataFrame
            cluster_df = self.affordability_index.copy()
            cluster_df['cluster'] = self.cluster_labels
            
            # Calculate mean values for each cluster
            cluster_profiles = cluster_df.groupby('cluster').mean()
            
            # Add count and percentage of areas in each cluster
            counts = cluster_df['cluster'].value_counts().sort_index()
            cluster_profiles['count'] = counts.values
            cluster_profiles['percentage'] = (counts.values / counts.sum() * 100).round(1)
            
            # Sort by affordability score
            cluster_profiles = cluster_profiles.sort_values(by='affordability_score', ascending=False)
            
            # Add cluster labels for easier interpretation
            affordability_labels = ['Very High', 'High', 'Moderate', 'Low', 'Very Low']
            if len(cluster_profiles) <= len(affordability_labels):
                cluster_profiles['affordability_level'] = affordability_labels[:len(cluster_profiles)]
            
            return cluster_profiles
        
        except Exception as e:
            logger.error(f"Error analyzing clusters: {e}", exc_info=True)
            return None
    
    def generate_visualizations(self, output_dir='visualizations/affordability_index'):
        """
        Generate visualizations of affordability index results.
        
        Args:
            output_dir (str): Directory to save visualizations
            
        Returns:
            list: Paths to saved visualization files
        """
        if self.affordability_index is None:
            logger.error("Affordability index not created yet. Run create_affordability_index first.")
            return []
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            visualization_paths = []
            
            # 1. Component loadings plot
            plt.figure(figsize=(12, 8))
            loadings = self.analyze_component_contributions()
            components_to_plot = min(3, len(loadings.columns) // 2)
            
            plt.subplot(2, 1, 1)
            loadings.sort_values(f'PC1_abs', ascending=False)[f'PC1'].head(10).plot(kind='bar')
            plt.title('Top Features Contributing to Affordability Index (PC1)', fontsize=14)
            plt.ylabel('Loading Value', fontsize=12)
            plt.xlabel('Features', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3)
            
            if components_to_plot > 1:
                plt.subplot(2, 1, 2)
                loadings.sort_values(f'PC2_abs', ascending=False)[f'PC2'].head(10).plot(kind='bar')
                plt.title('Top Features Contributing to PC2', fontsize=14)
                plt.ylabel('Loading Value', fontsize=12)
                plt.xlabel('Features', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(alpha=0.3)
                
            plt.tight_layout()
            
            file_path = os.path.join(output_dir, 'component_loadings.png')
            plt.savefig(file_path, bbox_inches='tight')
            visualization_paths.append(file_path)
            plt.close()
            
            # 2. Affordability score distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(self.affordability_index['affordability_score'], bins=20, kde=True)
            plt.title('Distribution of Affordability Scores', fontsize=15)
            plt.xlabel('Affordability Score (0-100)', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.grid(alpha=0.3)
            
            file_path = os.path.join(output_dir, 'affordability_distribution.png')
            plt.savefig(file_path, bbox_inches='tight')
            visualization_paths.append(file_path)
            plt.close()
            
            # 3. Cluster analysis (if clusters exist)
            if self.cluster_labels is not None:
                # Cluster scatterplot
                plt.figure(figsize=(10, 6))
                
                # Plot the first two principal components
                pc_columns = [col for col in self.affordability_index.columns if col.startswith('PC')]
                if len(pc_columns) >= 2:
                    cluster_data = self.affordability_index.copy()
                    cluster_data['cluster'] = self.cluster_labels
                    
                    sns.scatterplot(
                        x=pc_columns[0], 
                        y=pc_columns[1], 
                        hue='cluster',
                        palette='viridis',
                        data=cluster_data,
                        s=80,
                        alpha=0.7
                    )
                    
                    plt.title('Affordability Clusters', fontsize=15)
                    plt.xlabel('Principal Component 1', fontsize=12)
                    plt.ylabel('Principal Component 2', fontsize=12)
                    plt.grid(alpha=0.3)
                    
                    file_path = os.path.join(output_dir, 'affordability_clusters.png')
                    plt.savefig(file_path, bbox_inches='tight')
                    visualization_paths.append(file_path)
                    plt.close()
                    
                # Cluster profiles
                cluster_profiles = self.analyze_clusters()
                if cluster_profiles is not None:
                    plt.figure(figsize=(12, 6))
                    
                    # Bar chart of affordability scores by cluster
                    sns.barplot(
                        x=cluster_profiles.index, 
                        y='affordability_score',
                        palette='viridis',
                        data=cluster_profiles
                    )
                    
                    # Add count labels
                    for i, (idx, row) in enumerate(cluster_profiles.iterrows()):
                        plt.text(
                            i, 
                            row['affordability_score'] + 2,
                            f"n={int(row['count'])}\n({row['percentage']}%)",
                            ha='center'
                        )
                    
                    plt.title('Affordability Scores by Cluster', fontsize=15)
                    plt.xlabel('Cluster', fontsize=12)
                    plt.ylabel('Average Affordability Score', fontsize=12)
                    plt.grid(axis='y', alpha=0.3)
                    
                    file_path = os.path.join(output_dir, 'cluster_affordability_scores.png')
                    plt.savefig(file_path, bbox_inches='tight')
                    visualization_paths.append(file_path)
                    plt.close()
            
            logger.info(f"Generated {len(visualization_paths)} affordability index visualizations")
            
            return visualization_paths
        
        except Exception as e:
            logger.error(f"Error generating affordability visualizations: {e}", exc_info=True)
            return []
    
    def _check_price_correlation(self, component):
        """
        Check correlation of the component with price-related variables.
        
        Args:
            component (np.array): Principal component values
            
        Returns:
            float: Correlation coefficient
        """
        try:
            # Look for price-related columns
            price_columns = []
            for col in self.affordability_data.columns:
                col_lower = col.lower()
                if 'price' in col_lower or 'cost' in col_lower or 'value' in col_lower or 'rent' in col_lower:
                    price_columns.append(col)
            
            if not price_columns:
                # If no price columns found, assume we want higher values to mean better affordability
                return 0
            
            # Use the first price column found
            price_col = price_columns[0]
            price_data = self.affordability_data[price_col].values
            
            # Calculate correlation
            correlation = np.corrcoef(component, price_data)[0, 1]
            
            return correlation
        
        except Exception as e:
            logger.error(f"Error checking price correlation: {e}", exc_info=True)
            return 0
    
    def _normalize_to_scale(self, data, min_val=0, max_val=100):
        """
        Normalize data to a specified scale.
        
        Args:
            data (np.array): Data to normalize
            min_val (float): Minimum value of the new scale
            max_val (float): Maximum value of the new scale
            
        Returns:
            np.array: Normalized data
        """
        try:
            data_min = np.min(data)
            data_max = np.max(data)
            
            if data_max == data_min:
                return np.full_like(data, (max_val + min_val) / 2)
            
            normalized = (data - data_min) / (data_max - data_min) * (max_val - min_val) + min_val
            
            return normalized
        
        except Exception as e:
            logger.error(f"Error normalizing data: {e}", exc_info=True)
            return data
    
    def _generate_affordability_summary(self):
        """
        Generate summary metrics of affordability index results and save to CSV.
        """
        if self.affordability_index is None:
            return
        
        try:
            # Create summary directory if it doesn't exist
            summary_dir = os.path.join('data', 'processed', 'affordability')
            os.makedirs(summary_dir, exist_ok=True)
            
            # Generate summary statistics
            summary = pd.DataFrame()
            
            # Basic statistics for affordability scores
            score_stats = self.affordability_index['affordability_score'].describe()
            for stat in score_stats.index:
                summary.loc['affordability_score', stat] = score_stats[stat]
            
            # PCA explained variance
            for i, var in enumerate(self.explained_variance):
                summary.loc['explained_variance', f'PC{i+1}'] = var
            summary.loc['explained_variance', 'cumulative'] = np.sum(self.explained_variance)
            
            # Component loadings
            loadings = self.analyze_component_contributions()
            if loadings is not None:
                top_features = loadings.index[:5]  # Top 5 features
                for i, feature in enumerate(top_features):
                    summary.loc['top_features', f'feature_{i+1}'] = feature
                    summary.loc['top_features', f'loading_{i+1}'] = loadings.loc[feature, 'PC1']
            
            # Cluster statistics if available
            if self.cluster_labels is not None:
                cluster_profiles = self.analyze_clusters()
                if cluster_profiles is not None:
                    for cluster in cluster_profiles.index:
                        summary.loc['cluster_stats', f'cluster_{cluster}_count'] = cluster_profiles.loc[cluster, 'count']
                        summary.loc['cluster_stats', f'cluster_{cluster}_percentage'] = cluster_profiles.loc[cluster, 'percentage']
                        summary.loc['cluster_stats', f'cluster_{cluster}_affordability'] = cluster_profiles.loc[cluster, 'affordability_score']
            
            # Save summary to CSV
            summary_path = os.path.join(summary_dir, 'summary_affordability.csv')
            summary.to_csv(summary_path)
            logger.info(f"Saved affordability index summary to {summary_path}")
            
            # Also save full affordability index
            index_path = os.path.join(summary_dir, 'affordability_index.csv')
            self.affordability_index.to_csv(index_path, index=False)
            logger.info(f"Saved full affordability index to {index_path}")
            
            return summary
        
        except Exception as e:
            logger.error(f"Error generating affordability summary: {e}", exc_info=True)
            return None

# Example usage:
# affordability_creator = AffordabilityIndexCreator(housing_data)
# affordability_index = affordability_creator.create_affordability_index()
# cluster_results = affordability_creator.cluster_by_affordability(n_clusters=4)
# visualization_paths = affordability_creator.generate_visualizations()
