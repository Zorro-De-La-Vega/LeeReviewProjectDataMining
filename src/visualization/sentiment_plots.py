"""
Visualization module for sentiment analysis results.

This module provides functions to create interactive plots for sentiment analysis
results using Plotly, tailored for display in the Streamlit app.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import io
import base64
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_sentiment_distribution(sentiment_data):
    """
    Create an interactive bar chart showing the distribution of sentiment categories.
    
    Args:
        sentiment_data (pd.DataFrame): DataFrame containing sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot of sentiment distribution
    """
    try:
        # Count sentiment categories
        sentiment_counts = sentiment_data['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        # Create color map
        color_map = {'positive': '#2ecc71', 'neutral': '#3498db', 'negative': '#e74c3c'}
        
        # Create figure
        fig = px.bar(
            sentiment_counts, 
            x='Sentiment', 
            y='Count',
            color='Sentiment',
            color_discrete_map=color_map,
            title='Distribution of Sentiment in Airbnb Listing Descriptions',
            labels={'Sentiment': 'Sentiment Category', 'Count': 'Number of Listings'},
            text='Count'
        )
        
        # Update layout
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            legend_title_font_size=16,
            template='plotly_white'
        )
        
        # Add percentage labels
        total = sentiment_counts['Count'].sum()
        for i, row in sentiment_counts.iterrows():
            percentage = (row['Count'] / total) * 100
            fig.add_annotation(
                x=row['Sentiment'],
                y=row['Count'],
                text=f"{percentage:.1f}%",
                showarrow=False,
                yshift=10
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting sentiment distribution: {e}", exc_info=True)
        return go.Figure()

def plot_sentiment_subjectivity_scatter(sentiment_data):
    """
    Create an interactive scatter plot of sentiment vs. subjectivity.
    
    Args:
        sentiment_data (pd.DataFrame): DataFrame containing sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Interactive scatter plot
    """
    try:
        # Create figure
        fig = px.scatter(
            sentiment_data,
            x='sentiment_compound',
            y='subjectivity',
            color='sentiment_category',
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#3498db',
                'negative': '#e74c3c'
            },
            title='Sentiment vs. Subjectivity in Listing Descriptions',
            labels={
                'sentiment_compound': 'Sentiment Score',
                'subjectivity': 'Subjectivity Score',
                'sentiment_category': 'Sentiment Category'
            },
            opacity=0.7
        )
        
        # Add quadrant lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Add quadrant labels
        fig.add_annotation(x=0.5, y=0.75, text="Positive & Subjective", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=-0.5, y=0.75, text="Negative & Subjective", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=0.5, y=0.25, text="Positive & Objective", showarrow=False, font=dict(size=12))
        fig.add_annotation(x=-0.5, y=0.25, text="Negative & Objective", showarrow=False, font=dict(size=12))
        
        # Update layout
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            legend_title_font_size=16,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting sentiment vs. subjectivity: {e}", exc_info=True)
        return go.Figure()

def plot_sentiment_components(sentiment_data):
    """
    Create an interactive bar chart of average sentiment components.
    
    Args:
        sentiment_data (pd.DataFrame): DataFrame containing sentiment analysis results
        
    Returns:
        plotly.graph_objects.Figure: Interactive bar chart
    """
    try:
        # Calculate average sentiment components
        components = {
            'Positive': sentiment_data['sentiment_positive'].mean(),
            'Negative': sentiment_data['sentiment_negative'].mean(),
            'Neutral': sentiment_data['sentiment_neutral'].mean()
        }
        
        # Create DataFrame
        components_df = pd.DataFrame({
            'Component': list(components.keys()),
            'Score': list(components.values())
        })
        
        # Create color map
        color_map = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
        
        # Create figure
        fig = px.bar(
            components_df,
            x='Component',
            y='Score',
            color='Component',
            color_discrete_map=color_map,
            title='Average Sentiment Components in Listing Descriptions',
            labels={'Component': 'Sentiment Component', 'Score': 'Average Score'},
            text_auto='.3f'
        )
        
        # Update layout
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            legend_title_font_size=16,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting sentiment components: {e}", exc_info=True)
        return go.Figure()

def generate_wordcloud_image(topic_data):
    """
    Generate a WordCloud image from topic extraction results.
    
    Args:
        topic_data (pd.DataFrame): DataFrame containing topic extraction results
        
    Returns:
        str: Base64-encoded image for display in Streamlit
    """
    try:
        # Extract all keywords from listings
        all_keywords = []
        for keywords in topic_data['top_keywords']:
            if isinstance(keywords, list):
                all_keywords.extend(keywords)
        
        # Count keyword frequencies
        keyword_freq = {}
        for keyword in all_keywords:
            if keyword in keyword_freq:
                keyword_freq[keyword] += 1
            else:
                keyword_freq[keyword] = 1
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1
        ).generate_from_frequencies(keyword_freq)
        
        # Convert to image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        # Encode as base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        
        return img_str
    
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}", exc_info=True)
        return None

def plot_sentiment_by_price_range(sentiment_data, price_column='price'):
    """
    Create an interactive grouped bar chart of sentiment distribution by price range.
    
    Args:
        sentiment_data (pd.DataFrame): DataFrame containing sentiment analysis results and price
        price_column (str): Name of the price column
        
    Returns:
        plotly.graph_objects.Figure: Interactive grouped bar chart
    """
    try:
        if price_column not in sentiment_data.columns:
            logger.error(f"Price column '{price_column}' not found in sentiment data")
            return go.Figure()
        
        # Create price bins
        sentiment_data['price_range'] = pd.cut(
            sentiment_data[price_column],
            bins=[0, 50, 100, 150, 200, float('inf')],
            labels=['$0-50', '$51-100', '$101-150', '$151-200', '$200+']
        )
        
        # Count sentiments by price range
        sentiment_by_price = sentiment_data.groupby(['price_range', 'sentiment_category']).size().reset_index()
        sentiment_by_price.columns = ['Price Range', 'Sentiment', 'Count']
        
        # Create figure
        fig = px.bar(
            sentiment_by_price,
            x='Price Range',
            y='Count',
            color='Sentiment',
            barmode='group',
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#3498db',
                'negative': '#e74c3c'
            },
            title='Sentiment Distribution by Price Range',
            labels={
                'Price Range': 'Listing Price Range',
                'Count': 'Number of Listings',
                'Sentiment': 'Sentiment Category'
            }
        )
        
        # Update layout
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=16,
            yaxis_title_font_size=16,
            legend_title_font_size=16,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting sentiment by price range: {e}", exc_info=True)
        return go.Figure()
