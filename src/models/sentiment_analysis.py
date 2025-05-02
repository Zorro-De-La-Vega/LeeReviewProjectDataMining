"""
Sentiment Analysis Module for Airbnb Listing Descriptions

This module provides advanced NLP analysis capabilities to extract sentiment,
key topics, and language patterns from Airbnb listing descriptions.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ListingSentimentAnalyzer:
    """Analyzes sentiment and extracts key topics from Airbnb listing descriptions."""
    
    def __init__(self, data=None):
        """
        Initialize the ListingSentimentAnalyzer with data.
        
        Args:
            data (pd.DataFrame, optional): DataFrame containing listing descriptions
        """
        self.data = data
        self.sentiment_results = None
        self.topic_results = None
        
        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            logger.info("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('vader_lexicon')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.analyzer = SentimentIntensityAnalyzer()
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(cleaned_tokens)
    
    def analyze_sentiment(self, description_column='description'):
        """
        Analyze sentiment of listing descriptions.
        
        Args:
            description_column (str): Name of the column containing descriptions
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        if self.data is None:
            logger.error("No data provided for sentiment analysis")
            return None
            
        if description_column not in self.data.columns:
            logger.error(f"Column '{description_column}' not found in data")
            return None
            
        logger.info("Starting sentiment analysis of listing descriptions...")
        
        # Create results DataFrame
        results = pd.DataFrame()
        results['listing_id'] = self.data.index if 'id' not in self.data.columns else self.data['id']
        
        # Process descriptions
        descriptions = self.data[description_column].astype(str).fillna("")
        results['processed_text'] = descriptions.apply(self.preprocess_text)
        
        # VADER sentiment analysis
        results['sentiment_compound'] = results['processed_text'].apply(
            lambda x: self.analyzer.polarity_scores(x)['compound']
        )
        results['sentiment_positive'] = results['processed_text'].apply(
            lambda x: self.analyzer.polarity_scores(x)['pos']
        )
        results['sentiment_negative'] = results['processed_text'].apply(
            lambda x: self.analyzer.polarity_scores(x)['neg']
        )
        results['sentiment_neutral'] = results['processed_text'].apply(
            lambda x: self.analyzer.polarity_scores(x)['neu']
        )
        
        # TextBlob for subjectivity
        results['subjectivity'] = results['processed_text'].apply(
            lambda x: TextBlob(x).sentiment.subjectivity
        )
        
        # Categorize sentiment
        results['sentiment_category'] = results['sentiment_compound'].apply(
            lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
        )
        
        self.sentiment_results = results
        logger.info("Sentiment analysis completed")
        
        # Generate and save summary metrics
        self._generate_sentiment_summary()
        
        return results
    
    def extract_key_topics(self, text_column=None, processed_text_column='processed_text', n_topics=5, min_df=5):
        """
        Extract key topics from listing descriptions.
        
        Args:
            text_column (str, optional): Original text column to use if no processed text exists
            processed_text_column (str): Column name containing preprocessed text
            n_topics (int): Number of top topics to extract
            min_df (int): Minimum document frequency for TF-IDF
            
        Returns:
            pd.DataFrame: DataFrame with key topics for each listing
        """
        if self.sentiment_results is None:
            logger.error("Must run analyze_sentiment before extracting topics")
            return None
            
        logger.info("Extracting key topics from listing descriptions...")
        
        # Check if processed text column exists, otherwise use the original text column
        if processed_text_column not in self.sentiment_results.columns:
            if text_column and text_column in self.sentiment_results.columns:
                logger.warning(f"Processed column '{processed_text_column}' not found. Using '{text_column}' instead.")
                processed_text_column = text_column
            else:
                logger.error(f"Neither processed text column '{processed_text_column}' nor text column '{text_column}' found in data")
                return None
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(min_df=min_df, max_features=100)
        try:
            tfidf_matrix = vectorizer.fit_transform(self.sentiment_results[processed_text_column].fillna(''))
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            logger.error(f"Error during TF-IDF vectorization: {e}")
            return None
        
        # Extract top terms for each listing
        topic_results = pd.DataFrame(index=self.sentiment_results.index)
        topic_results['listing_id'] = self.sentiment_results['listing_id']
        
        # Get top keywords for each listing
        def get_top_keywords(text, n=n_topics):
            if not text:
                return []
                
            tokens = word_tokenize(text)
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            return [item[0] for item in Counter(filtered_tokens).most_common(n)]
        
        topic_results['top_keywords'] = self.sentiment_results[processed_text_column].apply(
            lambda x: get_top_keywords(x)
        )
        
        # Global topic extraction
        dense = tfidf_matrix.todense()
        sums = dense.sum(axis=0)
        global_scores = [(feature_names[i], sums[0, i]) for i in range(len(feature_names))]
        global_scores = sorted(global_scores, key=lambda x: x[1], reverse=True)
        topic_results['global_top_topics'] = [global_scores[:n_topics]] * len(topic_results)
        
        self.topic_results = topic_results
        logger.info("Topic extraction completed")
        
        return topic_results
    
    def generate_sentiment_visualizations(self, output_dir='visualizations/sentiment_analysis'):
        """
        Generate visualizations of sentiment analysis results.
        
        Args:
            output_dir (str): Directory to save visualizations
            
        Returns:
            list: Paths to saved visualization files
        """
        if self.sentiment_results is None:
            logger.error("Must run analyze_sentiment before generating visualizations")
            return []
            
        logger.info("Generating sentiment analysis visualizations...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        visualization_paths = []
        
        # 1. Sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment_category', data=self.sentiment_results, palette='viridis')
        plt.title('Distribution of Sentiment in Airbnb Listings', fontsize=15)
        plt.xlabel('Sentiment Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        file_path = os.path.join(output_dir, 'sentiment_distribution.png')
        plt.savefig(file_path, bbox_inches='tight')
        visualization_paths.append(file_path)
        plt.close()
        
        # 2. Sentiment vs Subjectivity scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.sentiment_results['sentiment_compound'], 
                  self.sentiment_results['subjectivity'],
                  alpha=0.5)
        plt.title('Sentiment vs. Subjectivity in Listing Descriptions', fontsize=15)
        plt.xlabel('Sentiment Compound Score', fontsize=12)
        plt.ylabel('Subjectivity Score', fontsize=12)
        plt.grid(alpha=0.3)
        
        file_path = os.path.join(output_dir, 'sentiment_subjectivity.png')
        plt.savefig(file_path, bbox_inches='tight')
        visualization_paths.append(file_path)
        plt.close()
        
        # 3. Sentiment Components (Positive, Negative, Neutral)
        plt.figure(figsize=(12, 6))
        sentiment_components = self.sentiment_results[['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']].mean()
        sns.barplot(x=sentiment_components.index, y=sentiment_components.values, palette='viridis')
        plt.title('Average Sentiment Components in Listing Descriptions', fontsize=15)
        plt.xlabel('Sentiment Component', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        
        file_path = os.path.join(output_dir, 'sentiment_components.png')
        plt.savefig(file_path, bbox_inches='tight')
        visualization_paths.append(file_path)
        plt.close()
        
        logger.info(f"Generated {len(visualization_paths)} sentiment visualizations")
        
        return visualization_paths
    
    def _generate_sentiment_summary(self):
        """
        Generate summary metrics of sentiment analysis results and save to CSV.
        """
        if self.sentiment_results is None:
            return
            
        # Create summary directory if it doesn't exist
        summary_dir = os.path.join('data', 'processed', 'sentiment')
        os.makedirs(summary_dir, exist_ok=True)
        
        # Generate summary statistics
        summary = pd.DataFrame()
        
        # Count by category
        category_counts = self.sentiment_results['sentiment_category'].value_counts()
        for category in category_counts.index:
            summary.loc['count', f'category_{category}'] = category_counts[category]
        
        # Sentiment score statistics
        for col in ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 
                    'sentiment_neutral', 'subjectivity']:
            summary.loc['mean', col] = self.sentiment_results[col].mean()
            summary.loc['median', col] = self.sentiment_results[col].median()
            summary.loc['std', col] = self.sentiment_results[col].std()
            summary.loc['min', col] = self.sentiment_results[col].min()
            summary.loc['max', col] = self.sentiment_results[col].max()
        
        # Save summary to CSV
        summary_path = os.path.join(summary_dir, 'summary_sentiment.csv')
        summary.to_csv(summary_path)
        logger.info(f"Saved sentiment analysis summary to {summary_path}")
        
        return summary

# Example usage:
# analyzer = ListingSentimentAnalyzer(airbnb_data)
# sentiment_results = analyzer.analyze_sentiment(description_column='description')
# topic_results = analyzer.extract_key_topics()
# visualization_paths = analyzer.generate_sentiment_visualizations()
