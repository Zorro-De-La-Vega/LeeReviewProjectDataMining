"""
Sentiment Analysis Page for Miami Housing Impact Hub

This module provides the Streamlit interface for the sentiment analysis
of Airbnb listing descriptions, uncovering insights about how hosts
market their properties and potential correlations with pricing and popularity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import logging
from PIL import Image
import base64
import io

from ..utils.data_loader import DataLoader
from ..utils.helpers import ensure_data_loaded, load_config

# Import sentiment analysis model and visualization
from src.models.sentiment_analysis import ListingSentimentAnalyzer
from src.visualization.sentiment_plots import (
    plot_sentiment_distribution,
    plot_sentiment_subjectivity_scatter,
    plot_sentiment_components,
    generate_wordcloud_image,
    plot_sentiment_by_price_range
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
CONFIG = load_config()
DATA_PATH_PROCESSED = CONFIG['paths']['processed']

@ensure_data_loaded(datasets=['airbnb_cleaned'])
def show_sentiment_analysis(data_loader, agent=None):
    """
    Display the sentiment analysis page.
    
    Args:
        data_loader: DataLoader instance
        agent: Optional agent instance for proactive insights
    """
    st.subheader("🔍 Sentiment Analysis of Airbnb Listing Descriptions")
    
    # Introduction
    st.markdown("""
    This advanced analysis uses Natural Language Processing techniques to extract insights from 
    the language used in Airbnb listing descriptions. Understanding the sentiment and key topics 
    can reveal how hosts market their properties and potential correlations with pricing and popularity.
    """)
    
    # Sidebar controls
    st.sidebar.subheader("Sentiment Analysis Controls")
    
    # Option to run or load analysis
    sentiment_option = st.sidebar.radio(
        "Sentiment Analysis Data",
        ["Run New Analysis", "Load Previous Results"],
        index=1 if os.path.exists(os.path.join(DATA_PATH_PROCESSED, 'sentiment', 'summary_sentiment.csv')) else 0
    )
    
    # Get Airbnb data
    airbnb_data = data_loader.get_data('airbnb_cleaned')
    
    if airbnb_data is None:
        st.error("Airbnb data not loaded. Please load the data from the Home page first.")
        return
    
    # Progress placeholder
    progress_placeholder = st.empty()
    
    # Results container
    results_container = st.container()
    
    with results_container:
        try:
            # Check if we need to run a new analysis or load previous results
            if sentiment_option == "Run New Analysis":
                # Create analyzer
                analyzer = ListingSentimentAnalyzer(airbnb_data)
                
                # Show progress
                with progress_placeholder:
                    progress_bar = st.progress(0)
                    st.info("Running sentiment analysis on listing descriptions... This may take a few moments.")
                    
                    # Analyze sentiment
                    sentiment_results = analyzer.analyze_sentiment(description_column='description')
                    progress_bar.progress(50)
                    
                    # Extract topics
                    topic_results = analyzer.extract_key_topics()
                    progress_bar.progress(100)
                    
                    # Clear progress
                    time.sleep(1)
                    progress_placeholder.empty()
            else:
                # Load previous results
                try:
                    sentiment_path = os.path.join(DATA_PATH_PROCESSED, 'sentiment', 'summary_sentiment.csv')
                    
                    if not os.path.exists(sentiment_path):
                        st.warning("No previous sentiment analysis results found. Running new analysis...")
                        analyzer = ListingSentimentAnalyzer(airbnb_data)
                        sentiment_results = analyzer.analyze_sentiment(description_column='Listing Title')
                        topic_results = analyzer.extract_key_topics(text_column='Listing Title')
                    else:
                        with progress_placeholder:
                            st.info("Loading previous sentiment analysis results...")
                            
                            # We'll need to run a lightweight analysis to get the actual results
                            # The summary file just gives us metrics, not the full analysis
                            analyzer = ListingSentimentAnalyzer(airbnb_data)
                            sentiment_results = analyzer.analyze_sentiment(description_column='Listing Title')
                            topic_results = analyzer.extract_key_topics(text_column='Listing Title')
                            
                            progress_placeholder.empty()
                            
                except Exception as e:
                    logger.error(f"Error loading previous sentiment analysis: {e}", exc_info=True)
                    st.error(f"Error loading previous results: {str(e)}")
                    st.info("Running new analysis instead...")
                    
                    analyzer = ListingSentimentAnalyzer(airbnb_data)
                    sentiment_results = analyzer.analyze_sentiment(description_column='Listing Title')
                    topic_results = analyzer.extract_key_topics(text_column='Listing Title')
            
            # Display results
            if sentiment_results is not None:
                # Show summary metrics
                st.subheader("Sentiment Analysis Summary")
                
                # Create metrics for positive, neutral, negative counts
                metrics_cols = st.columns(3)
                
                sentiment_counts = sentiment_results['sentiment_category'].value_counts()
                total_listings = len(sentiment_results)
                
                pos_count = sentiment_counts.get('positive', 0)
                neu_count = sentiment_counts.get('neutral', 0)
                neg_count = sentiment_counts.get('negative', 0)
                
                metrics_cols[0].metric(
                    "Positive Listings", 
                    f"{pos_count} ({pos_count/total_listings:.1%})",
                    delta=None,
                    delta_color="normal"
                )
                
                metrics_cols[1].metric(
                    "Neutral Listings", 
                    f"{neu_count} ({neu_count/total_listings:.1%})",
                    delta=None,
                    delta_color="normal"
                )
                
                metrics_cols[2].metric(
                    "Negative Listings", 
                    f"{neg_count} ({neg_count/total_listings:.1%})",
                    delta=None,
                    delta_color="normal"
                )
                
                # Show sentiment distribution
                st.subheader("Distribution of Sentiment")
                try:
                    fig = plot_sentiment_distribution(sentiment_results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insight:** The distribution shows how hosts generally frame their listings, 
                    with a strong bias toward positive language. This reflects the marketing nature 
                    of these descriptions, designed to attract potential guests.
                    """)
                except Exception as e:
                    logger.error(f"Error plotting sentiment distribution: {e}", exc_info=True)
                    st.error(f"Could not generate sentiment distribution plot: {str(e)}")
                
                # Sentiment components
                st.subheader("Sentiment Components")
                try:
                    fig = plot_sentiment_components(sentiment_results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insight:** This breakdown shows the composition of sentiment in listing descriptions.
                    Higher positive scores indicate more enthusiastic and appealing descriptions,
                    while higher negative scores might indicate mentions of restrictions or policies.
                    """)
                except Exception as e:
                    logger.error(f"Error plotting sentiment components: {e}", exc_info=True)
                    st.error(f"Could not generate sentiment components plot: {str(e)}")
                
                # Sentiment vs subjectivity
                st.subheader("Sentiment vs. Subjectivity")
                try:
                    fig = plot_sentiment_subjectivity_scatter(sentiment_results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insight:** This scatter plot reveals the relationship between the emotional tone 
                    (sentiment) and the level of opinion (subjectivity) in listings. 
                    
                    - **High subjectivity + positive sentiment**: Enthusiastic, personal descriptions
                    - **Low subjectivity + positive sentiment**: Factual descriptions with positive attributes
                    - **High subjectivity + negative sentiment**: Opinions about limitations or policies
                    - **Low subjectivity + negative sentiment**: Factual restrictions or warnings
                    """)
                except Exception as e:
                    logger.error(f"Error plotting sentiment vs subjectivity: {e}", exc_info=True)
                    st.error(f"Could not generate sentiment vs subjectivity plot: {str(e)}")
                
                # Price analysis
                if 'price' in airbnb_data.columns:
                    # Merge price data with sentiment results
                    sentiment_with_price = sentiment_results.copy()
                    sentiment_with_price['price'] = airbnb_data['price'].values
                    
                    st.subheader("Sentiment Analysis by Price Range")
                    try:
                        fig = plot_sentiment_by_price_range(sentiment_with_price)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Insight:** This analysis reveals how sentiment in listing descriptions varies across
                        different price ranges. Higher-priced listings might use different language strategies
                        compared to budget options, reflecting different target markets and value propositions.
                        """)
                    except Exception as e:
                        logger.error(f"Error plotting sentiment by price: {e}", exc_info=True)
                        st.error(f"Could not generate sentiment by price range plot: {str(e)}")
                
                # Topic extraction
                if topic_results is not None:
                    st.subheader("Key Topics in Listing Descriptions")
                    
                    # Generate word cloud
                    try:
                        img_str = generate_wordcloud_image(topic_results)
                        if img_str:
                            st.markdown(f"""
                            <div style="text-align: center">
                                <img src="data:image/png;base64,{img_str}" width="100%">
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("""
                            **Insight:** This word cloud visualizes the most common descriptive terms used
                            in Airbnb listings. The size of each word represents its frequency across all listings.
                            This reveals the common selling points and amenities that hosts emphasize when marketing
                            their properties.
                            """)
                        else:
                            st.warning("Could not generate word cloud visualization.")
                    except Exception as e:
                        logger.error(f"Error generating wordcloud: {e}", exc_info=True)
                        st.error(f"Could not generate word cloud: {str(e)}")
                    
                    # Show global top topics
                    if len(topic_results) > 0 and 'global_top_topics' in topic_results.columns:
                        global_topics = topic_results['global_top_topics'].iloc[0]
                        if isinstance(global_topics, list) and len(global_topics) > 0:
                            st.subheader("Most Common Topics Across All Listings")
                            
                            # Create a table
                            topics_df = pd.DataFrame(global_topics, columns=['Term', 'Score'])
                            topics_df['Score'] = topics_df['Score'].round(4)
                            
                            st.table(topics_df)
                            
                            st.markdown("""
                            **Business Value:** Understanding the most common terms allows property managers and hosts to:
                            1. Identify key amenities and features that are standard in the market
                            2. Find opportunities for differentiation from common offerings
                            3. Optimize listing descriptions to include high-performing keywords
                            """)
                
                # Machine learning insights
                st.subheader("NLP Insights for Miami's Housing Market")
                st.markdown("""
                The sentiment analysis of Airbnb listings provides valuable insights for various stakeholders:
                
                **For Property Managers and Hosts:**
                - Optimize listing descriptions based on successful patterns
                - Identify language that correlates with higher booking rates
                - Target specific market segments with appropriate tone and content
                
                **For City Planners and Policymakers:**
                - Understand how different neighborhoods are marketed
                - Identify trends in property types and amenities across areas
                - Monitor changes in the short-term rental market over time
                
                **For Investors:**
                - Identify emerging neighborhoods based on description sentiment and amenities
                - Analyze property positioning strategies across different market segments
                - Compare listing quality across different property types and price points
                """)
                
                # Business recommendations
                st.subheader("Strategic Recommendations")
                
                recommendation_cols = st.columns(3)
                
                with recommendation_cols[0]:
                    st.markdown("#### For Hosts")
                    st.markdown("""
                    - **Emphasize unique features** that stand out from common terms
                    - **Match sentiment to target audience** - luxury listings benefit from more objective, fact-based descriptions
                    - **Balance positivity with authenticity** to set appropriate guest expectations
                    """)
                
                with recommendation_cols[1]:
                    st.markdown("#### For Investors")
                    st.markdown("""
                    - **Look beyond just numbers** to how properties are positioned in the market
                    - **Identify undervalued areas** where sentiment doesn't match price potential
                    - **Track emerging neighborhood terms** to spot early gentrification trends
                    """)
                
                with recommendation_cols[2]:
                    st.markdown("#### For Policymakers")
                    st.markdown("""
                    - **Monitor neighborhood marketing trends** to understand tourism impacts
                    - **Analyze amenity patterns** across different communities
                    - **Track changes in sentiment** as indicators of market shifts or policy impacts
                    """)
            
            else:
                st.warning("No sentiment analysis results available. Please try running the analysis again.")
        
        except Exception as e:
            logger.error(f"Error in sentiment analysis page: {e}", exc_info=True)
            st.error(f"An error occurred while displaying sentiment analysis: {str(e)}")
            st.info("Please try again or contact support.")
