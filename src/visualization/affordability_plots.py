"""
Affordability Index Visualization Module

This module provides interactive visualization functions for the affordability index
analysis results using Plotly. These visualizations help stakeholders understand
housing affordability patterns across different Miami neighborhoods.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_affordability_distribution(affordability_data):
    """
    Create a histogram of affordability scores distribution.
    
    Args:
        affordability_data (pd.DataFrame): DataFrame containing affordability scores
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Create histogram with distribution curve
        fig = px.histogram(
            affordability_data, 
            x='affordability_score',
            nbins=20,
            histnorm='percent',
            title='Distribution of Housing Affordability Scores',
            labels={'affordability_score': 'Affordability Score (0-100)', 'percent': 'Percentage (%)'},
            color_discrete_sequence=['#3366CC'],
        )
        
        # Add KDE curve
        fig.update_traces(opacity=0.7)
        
        # Add median and mean markers
        median_score = affordability_data['affordability_score'].median()
        mean_score = affordability_data['affordability_score'].mean()
        
        # Add vertical lines for mean and median
        fig.add_vline(x=median_score, line_width=2, line_dash="dash", line_color="red",
                     annotation_text=f"Median: {median_score:.1f}", annotation_position="top right")
        fig.add_vline(x=mean_score, line_width=2, line_dash="dot", line_color="green",
                     annotation_text=f"Mean: {mean_score:.1f}", annotation_position="top left")
        
        # Improve layout
        fig.update_layout(
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            title_font=dict(size=18),
            template='plotly_white',
            bargap=0.1,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating affordability distribution plot: {e}", exc_info=True)
        return None

def plot_affordability_by_feature(affordability_data, feature_column, **kwargs):
    """
    Create a scatter plot of affordability scores vs. another feature.
    
    Args:
        affordability_data (pd.DataFrame): DataFrame containing affordability scores
        feature_column (str): Column name to plot against affordability score
        **kwargs: Additional keyword arguments for customization
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Get optional parameters
        color_column = kwargs.get('color_column', None)
        size_column = kwargs.get('size_column', None)
        hover_data = kwargs.get('hover_data', None)
        trend_line = kwargs.get('trend_line', True)
        custom_title = kwargs.get('title', f'Affordability Score vs. {feature_column}')
        
        # Create scatter plot
        if color_column:
            fig = px.scatter(
                affordability_data,
                x=feature_column,
                y='affordability_score',
                color=color_column,
                size=size_column,
                hover_name=hover_data,
                title=custom_title,
                trendline='ols' if trend_line else None,
                labels={'affordability_score': 'Affordability Score (0-100)'},
                color_continuous_scale=px.colors.sequential.Viridis
            )
        else:
            fig = px.scatter(
                affordability_data,
                x=feature_column,
                y='affordability_score',
                size=size_column,
                hover_name=hover_data,
                title=custom_title,
                trendline='ols' if trend_line else None,
                labels={'affordability_score': 'Affordability Score (0-100)'},
                color_discrete_sequence=['#3366CC']
            )
        
        # Improve layout
        fig.update_layout(
            xaxis_title_font=dict(size=14),
            yaxis_title_font=dict(size=14),
            title_font=dict(size=18),
            template='plotly_white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Customize marker appearance
        fig.update_traces(
            marker=dict(
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating affordability by feature plot: {e}", exc_info=True)
        return None

def plot_component_loadings(loadings_data, n_components=2, top_n=10):
    """
    Create a bar chart of component loadings to show feature importance.
    
    Args:
        loadings_data (pd.DataFrame): DataFrame containing loadings data
        n_components (int): Number of components to plot
        top_n (int): Number of top features to show
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Create subplot figure with n_components rows
        fig = make_subplots(
            rows=n_components,
            cols=1,
            subplot_titles=[f"Principal Component {i+1} Feature Contributions" for i in range(n_components)]
        )
        
        colors = px.colors.qualitative.Plotly
        
        # Loop through components and add bar chart for each
        for i in range(n_components):
            component_col = f'PC{i+1}'
            abs_col = f'{component_col}_abs'
            
            # Sort by absolute loading value and get top features
            sorted_loadings = loadings_data.sort_values(by=abs_col, ascending=False).head(top_n)
            
            # Create bar chart for this component
            fig.add_trace(
                go.Bar(
                    x=sorted_loadings.index,
                    y=sorted_loadings[component_col],
                    name=f'PC{i+1}',
                    marker_color=colors[i % len(colors)],
                    text=[f"{val:.3f}" for val in sorted_loadings[component_col]],
                    textposition='outside'
                ),
                row=i+1,
                col=1
            )
            
            # Update y-axis title
            fig.update_yaxes(title_text="Loading Value", row=i+1, col=1)
        
        # Update layout
        fig.update_layout(
            height=300 * n_components,
            title=f"Top {top_n} Features Contributing to Principal Components",
            title_font=dict(size=18),
            template='plotly_white',
            showlegend=False,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Update x-axis for all subplots
        fig.update_xaxes(
            tickangle=45,
            title_text="Features"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating component loadings plot: {e}", exc_info=True)
        return None

def plot_cluster_analysis(cluster_data, cluster_column='affordability_cluster'):
    """
    Create a scatter plot of clusters in PCA space.
    
    Args:
        cluster_data (pd.DataFrame): DataFrame containing cluster assignments
        cluster_column (str): Column name for cluster assignments
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Get available PC columns
        pc_columns = [col for col in cluster_data.columns if col.startswith('PC')]
        
        # Need at least 2 PCs to create a scatter plot
        if len(pc_columns) < 2:
            logger.error("Need at least 2 principal components for cluster visualization")
            return None
        
        # Create 3D scatter plot if 3 or more PCs are available
        if len(pc_columns) >= 3:
            fig = px.scatter_3d(
                cluster_data,
                x=pc_columns[0],
                y=pc_columns[1],
                z=pc_columns[2],
                color=cluster_column,
                symbol=cluster_column,
                title='Affordability Clusters in 3D Principal Component Space',
                labels={
                    pc_columns[0]: 'Principal Component 1',
                    pc_columns[1]: 'Principal Component 2',
                    pc_columns[2]: 'Principal Component 3',
                    cluster_column: 'Cluster'
                },
                hover_name='affordability_score',
                hover_data=['affordability_score'],
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
        # Otherwise create 2D scatter plot
        else:
            fig = px.scatter(
                cluster_data,
                x=pc_columns[0],
                y=pc_columns[1],
                color=cluster_column,
                symbol=cluster_column,
                title='Affordability Clusters in Principal Component Space',
                labels={
                    pc_columns[0]: 'Principal Component 1',
                    pc_columns[1]: 'Principal Component 2',
                    cluster_column: 'Cluster'
                },
                hover_name='affordability_score',
                hover_data=['affordability_score'],
                color_discrete_sequence=px.colors.qualitative.Bold
            )
        
        # Improve layout
        fig.update_layout(
            title_font=dict(size=18),
            template='plotly_white',
            legend_title_text='Affordability Cluster',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Customize marker appearance
        fig.update_traces(
            marker=dict(
                size=12,
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating cluster analysis plot: {e}", exc_info=True)
        return None

def plot_cluster_profiles(cluster_profiles):
    """
    Create a radar chart showing the profiles of each affordability cluster.
    
    Args:
        cluster_profiles (pd.DataFrame): DataFrame containing cluster profile statistics
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Identify numeric columns to include in radar chart (exclude count, percentage)
        exclude_cols = ['count', 'percentage', 'cluster', 'affordability_level']
        radar_cols = [col for col in cluster_profiles.columns 
                     if col not in exclude_cols and pd.api.types.is_numeric_dtype(cluster_profiles[col])]
        
        # Need at least 3 dimensions for a meaningful radar chart
        if len(radar_cols) < 3:
            logger.warning("Not enough numeric columns for radar chart, creating bar chart instead")
            
            # Create bar chart of affordability scores instead
            fig = px.bar(
                cluster_profiles,
                y='affordability_score',
                title='Affordability Score by Cluster',
                labels={'affordability_score': 'Average Affordability Score (0-100)'},
                color_discrete_sequence=px.colors.qualitative.Bold,
                text='percentage'
            )
            
            # Add count labels
            for i, row in enumerate(cluster_profiles.itertuples()):
                fig.add_annotation(
                    x=i,
                    y=row.affordability_score + 2,
                    text=f"n={int(row.count)}",
                    showarrow=False
                )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Cluster",
                title_font=dict(size=18),
                template='plotly_white',
                showlegend=False
            )
            
            # Format percentage text
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            
            return fig
        
        # Create radar chart with multiple traces (one per cluster)
        fig = go.Figure()
        
        colors = px.colors.qualitative.Bold
        
        # Normalize values for radar chart (0-1 scale)
        normalized_profiles = cluster_profiles.copy()
        for col in radar_cols:
            if normalized_profiles[col].max() > normalized_profiles[col].min():
                normalized_profiles[col] = (normalized_profiles[col] - normalized_profiles[col].min()) / \
                                          (normalized_profiles[col].max() - normalized_profiles[col].min())
        
        # Add a trace for each cluster
        for i, (cluster_idx, row) in enumerate(normalized_profiles.iterrows()):
            cluster_name = f"Cluster {cluster_idx}"
            if 'affordability_level' in cluster_profiles.columns:
                level = cluster_profiles.loc[cluster_idx, 'affordability_level']
                cluster_name = f"Cluster {cluster_idx}: {level}"
            
            fig.add_trace(go.Scatterpolar(
                r=row[radar_cols].values,
                theta=radar_cols,
                fill='toself',
                name=cluster_name,
                line_color=colors[i % len(colors)]
            ))
        
        # Improve layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='Affordability Cluster Profiles',
            title_font=dict(size=18),
            template='plotly_white',
            showlegend=True,
            legend_title_text='Cluster',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating cluster profiles plot: {e}", exc_info=True)
        return None

def plot_affordability_time_series(affordability_data, time_column):
    """
    Create a time series plot of affordability scores.
    
    Args:
        affordability_data (pd.DataFrame): DataFrame containing affordability scores
        time_column (str): Column name containing time/date information
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Check if time column exists
        if time_column not in affordability_data.columns:
            logger.error(f"Time column '{time_column}' not found in data")
            return None
        
        # Check if data has a time index and convert if needed
        if not pd.api.types.is_datetime64_any_dtype(affordability_data[time_column]):
            try:
                affordability_data[time_column] = pd.to_datetime(affordability_data[time_column])
            except:
                logger.error(f"Could not convert '{time_column}' to datetime")
                return None
        
        # Create time series plot
        fig = px.line(
            affordability_data.sort_values(time_column),
            x=time_column,
            y='affordability_score',
            title='Affordability Score Over Time',
            labels={'affordability_score': 'Affordability Score (0-100)'},
            markers=True
        )
        
        # Add trend line
        fig.add_traces(
            px.scatter(
                affordability_data.sort_values(time_column),
                x=time_column,
                y='affordability_score',
                trendline='ols'
            ).data[1]
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title=time_column,
            yaxis_title='Affordability Score (0-100)',
            title_font=dict(size=18),
            template='plotly_white',
            hovermode='closest',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Customize lines and markers
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8),
            selector=dict(mode='markers+lines')
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating affordability time series plot: {e}", exc_info=True)
        return None

def plot_affordability_heatmap(affordability_data, group_by_column):
    """
    Create a heatmap of affordability scores grouped by a categorical variable.
    
    Args:
        affordability_data (pd.DataFrame): DataFrame containing affordability scores
        group_by_column (str): Column name to group by
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    try:
        # Check if group_by column exists
        if group_by_column not in affordability_data.columns:
            logger.error(f"Group by column '{group_by_column}' not found in data")
            return None
        
        # Group data and calculate average affordability score
        grouped_data = affordability_data.groupby(group_by_column)['affordability_score'].agg(
            ['mean', 'median', 'count']
        ).reset_index().sort_values('mean', ascending=False)
        
        # Create a heatmap-style visualization
        fig = px.imshow(
            grouped_data[['mean']].T,
            x=grouped_data[group_by_column],
            y=['Mean Affordability Score'],
            color_continuous_scale='viridis',
            aspect='auto',
            title=f'Affordability Score by {group_by_column}'
        )
        
        # Add text annotations
        for i, row in enumerate(grouped_data.itertuples()):
            fig.add_annotation(
                x=i,
                y=0,
                text=f"{row.mean:.1f}<br>n={row.count}",
                showarrow=False,
                font=dict(color='white' if row.mean < 50 else 'black')
            )
        
        # Improve layout
        fig.update_layout(
            title_font=dict(size=18),
            template='plotly_white',
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Update colorbar
        fig.update_coloraxes(
            colorbar_title='Affordability<br>Score',
            colorbar_len=0.6
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating affordability heatmap: {e}", exc_info=True)
        return None

def create_affordability_dashboard(affordability_data, loadings=None, cluster_profiles=None):
    """
    Create a comprehensive dashboard of affordability visualizations.
    
    Args:
        affordability_data (pd.DataFrame): DataFrame containing affordability scores
        loadings (pd.DataFrame, optional): Component loadings data
        cluster_profiles (pd.DataFrame, optional): Cluster profiles data
        
    Returns:
        dict: Dictionary with multiple plotly figure objects
    """
    try:
        dashboard = {}
        
        # Generate all available visualizations
        dashboard['distribution'] = plot_affordability_distribution(affordability_data)
        
        # Add component loadings if available
        if loadings is not None:
            dashboard['loadings'] = plot_component_loadings(loadings)
        
        # Add cluster visualizations if available
        if 'affordability_cluster' in affordability_data.columns:
            dashboard['clusters'] = plot_cluster_analysis(affordability_data)
            
            # Add cluster profiles if available
            if cluster_profiles is not None:
                dashboard['cluster_profiles'] = plot_cluster_profiles(cluster_profiles)
        
        # Find numeric columns to create feature correlation plots
        numeric_cols = affordability_data.select_dtypes(include=np.number).columns
        # Exclude PC columns and affordability scores/indices
        feature_cols = [col for col in numeric_cols if not col.startswith('PC') and 
                       'affordability' not in col.lower() and 'cluster' not in col.lower()]
        
        # Add feature correlation plots if available
        if feature_cols:
            # Take the first feature for demonstration
            demo_feature = feature_cols[0]
            dashboard['feature_correlation'] = plot_affordability_by_feature(
                affordability_data, demo_feature
            )
        
        logger.info(f"Created affordability dashboard with {len(dashboard)} visualizations")
        
        return dashboard
    
    except Exception as e:
        logger.error(f"Error creating affordability dashboard: {e}", exc_info=True)
        return {}
