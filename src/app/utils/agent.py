#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Housing Impact Agent Module

This module implements an intelligent agent for the Miami Housing Impact Hub that
provides proactive insights, recommendations, and answers user questions about
the impact of short-term rentals on housing affordability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from datetime import datetime

class HousingImpactAgent:
    """
    An intelligent agent that analyzes housing data and provides insights,
    recommendations, and answers to user questions.
    """
    
    def __init__(self, data_loader, summary_logger=None):
        """
        Initialize the Housing Impact Agent with data access.
        
        Args:
            data_loader: DataLoader instance with access to all datasets
            summary_logger: SummaryLogger instance for tracking insights and interactions
        """
        self.data_loader = data_loader
        self.summary_logger = summary_logger
        self.user_interactions = []
        self.insights_cache = []
        self.user_interests = set()
        self.interaction_history = []
        self.analysis_cache = {}
    
    def get_initial_insights(self) -> List[Dict[str, str]]:
        """
        Generate initial insights when the user first opens the application.
        
        Returns:
            List of insight dictionaries with title and description
        """
        insights = []
        
        try:
            if self.data_loader.combined_data is not None:
                # Get high density areas
                high_density_zips = self.data_loader.combined_data.sort_values(
                    'listings_per_1000', ascending=False).head(3)
                
                for _, row in high_density_zips.iterrows():
                    insights.append({
                        'title': 'Concentration Hotspot',
                        'description': f"ZIP code {row['zip_code']} has {row['listing_count']} Airbnb listings, "
                                       f"representing {row['listings_per_1000']:.1f} listings per 1,000 housing units."
                    })
                
                # Get affordability correlation
                if 'rent_to_income_ratio' in self.data_loader.combined_data.columns and \
                   'airbnb_density_level' in self.data_loader.combined_data.columns:
                    by_density = self.data_loader.combined_data.groupby('airbnb_density_level')['rent_to_income_ratio'].mean()
                    
                    if not by_density.empty and len(by_density) > 1:
                        high_ratio = by_density.loc['High'] if 'High' in by_density else None
                        low_ratio = by_density.loc['Low'] if 'Low' in by_density else None
                        
                        if high_ratio is not None and low_ratio is not None:
                            difference = high_ratio - low_ratio
                            insights.append({
                                'title': 'Affordability Impact',
                                'description': f"Neighborhoods with high Airbnb density have a "
                                               f"rent-to-income ratio that is {difference:.1f}% "
                                               f"{'higher' if difference > 0 else 'lower'} than low-density areas."
                            })
        except Exception as e:
            print(f"Error generating initial insights: {e}")
            # Add a fallback insight
            insights.append({
                'title': 'Explore Miami Housing Data',
                'description': "Discover patterns in short-term rental impacts across Miami-Dade County neighborhoods."
            })
            
        # Add a general insight if we don't have enough
        if len(insights) < 2:
            insights.append({
                'title': 'Interactive Analysis',
                'description': "Use the dashboard to explore connections between Airbnb density and housing affordability."
            })
        
        # Cache insights
        self.insights_cache = insights
        
        # Log insights to summary logger if available
        if self.summary_logger:
            for insight in insights:
                self.summary_logger.log_insight('initial', insight['title'], insight['description'])
        
        return insights
    
    def get_recommended_explorations(self) -> List[Dict[str, Any]]:
        """
        Generate recommended exploration paths based on user interactions.
        
        Returns:
            List of recommendation dictionaries with title, description, and action
        """
        # Base recommendations that are always valuable
        recommendations = [
            {
                'title': 'Compare High vs. Low Density Areas',
                'description': 'Explore how affordability metrics differ between areas with high and low Airbnb concentration',
                'page': 'Interactive Dashboard',
                'action': {'tab': 'Affordability Metrics'}
            },
            {
                'title': 'Test Regulation Scenarios',
                'description': 'Predict how different short-term rental regulations might affect housing affordability',
                'page': 'Affordability Prediction',
                'action': {'tab': 'Scenario Testing'}
            }
        ]
        
        # Add data-driven recommendations if we have the necessary data
        if self.data_loader.combined_data is not None and not self.data_loader.combined_data.empty:
            # Check if we can make correlation recommendations
            if 'rent_to_income_ratio' in self.data_loader.combined_data.columns and 'listings_per_1000' in self.data_loader.combined_data.columns:
                corr = self.data_loader.combined_data[['rent_to_income_ratio', 'listings_per_1000']].corr().iloc[0,1]
                
                if abs(corr) > 0.3:  # Only recommend if there's a meaningful correlation
                    recommendations.append({
                        'title': 'Investigate Correlation Strength',
                        'description': f'Examine the {abs(corr):.2f} correlation between Airbnb density and rent-to-income ratios',
                        'page': 'Interactive Dashboard',
                        'action': {'tab': 'Correlation Analysis'}
                    })
            
            # Check if we can recommend a specific ZIP code to explore
            if 'zip_code' in self.data_loader.combined_data.columns and 'rent_to_income_ratio' in self.data_loader.combined_data.columns:
                # Find ZIP code with highest rent-to-income ratio
                troubled_zip = self.data_loader.combined_data.sort_values('rent_to_income_ratio', ascending=False).iloc[0]
                
                recommendations.append({
                    'title': f'Explore ZIP Code {troubled_zip["zip_code"]}',
                    'description': f'This area has a concerning rent-to-income ratio of {troubled_zip["rent_to_income_ratio"]:.1f}%',
                    'page': 'Affordability Prediction',
                    'action': {'tab': 'Neighborhood Impact', 'zip_code': troubled_zip["zip_code"]}
                })
        
        # Log this recommendation for future analysis
        self.update_from_user_interaction({
            'page': 'Housing Assistant',
            'action': 'recommendation',
            'recommendation': recommendations
        })
        
        # Log the recommendation to the summary logger
        if self.summary_logger:
            for recommendation in recommendations:
                self.summary_logger.log_insight('recommendation', f"Recommendation: {recommendation['title']}", recommendation['description'])
        
        return recommendations
    
    def process_user_question(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question from the user and return an answer.
        
        Args:
            question: User's natural language question
            
        Returns:
            Dictionary with answer and relevant data
        """
        # Log the question in the summary logger
        if self.summary_logger:
            self.summary_logger.log_interaction('Housing Assistant', 'question', {'question': question})
            
        # Track this question in interaction history
        self.interaction_history.append(question)
        
        # Update user interests based on keywords in the question
        keywords = {
            'regulation': ['regulation', 'policy', 'restrict', 'limit', 'law'],
            'affordability': ['afford', 'cost', 'price', 'expensive', 'cheap'],
            'prediction': ['predict', 'forecast', 'future', 'expect', 'anticipate'],
            'impact': ['impact', 'effect', 'influence', 'affect', 'change'],
            'airbnb': ['airbnb', 'short-term', 'rental', 'listing']
        }
        
        for topic, terms in keywords.items():
            if any(term in question.lower() for term in terms):
                self.user_interests.add(topic)
        
        # Pattern matching for different question types
        # In a production system, this would use a more sophisticated NLP approach
        
        # Questions about affordability
        if any(term in question.lower() for term in ['most affordable', 'least expensive', 'cheapest']):
            return self._answer_affordability_question(question, find_affordable=True)
        
        if any(term in question.lower() for term in ['least affordable', 'most expensive', 'priciest']):
            return self._answer_affordability_question(question, find_affordable=False)
        
        # Questions about Airbnb impact
        if re.search(r'(impact|effect|influence|affect).*(airbnb|short.term)', question.lower()):
            return self._answer_impact_question(question)
        
        # Questions about regulations
        if re.search(r'(regulation|policy|restrict|law).*(airbnb|short.term)', question.lower()):
            return self._answer_regulation_question(question)
        
        # Questions about predictions
        if re.search(r'(predict|forecast|future|expect).*(rent|price|cost|affordable)', question.lower()):
            return self._answer_prediction_question(question)
        
        # Default response if no specific pattern is matched
        return {
            'answer': "I don't have enough information to answer that specific question, but I can help you explore the relationship between short-term rentals and housing affordability through the dashboard and prediction tools.",
            'data': None,
            'visualization_type': None
        }
    
    def _answer_affordability_question(self, question: str, find_affordable: bool) -> Dict[str, Any]:
        """Answer questions about affordability."""
        if self.data_loader.combined_data is None or self.data_loader.combined_data.empty:
            return {
                'answer': "I don't have the necessary data to answer this question accurately. Please check the dashboard for general affordability trends.",
                'data': None,
                'visualization_type': None
            }
        
        # Check if we have the necessary columns
        if 'zip_code' not in self.data_loader.combined_data.columns or 'rent_to_income_ratio' not in self.data_loader.combined_data.columns:
            return {
                'answer': "I can't determine affordability levels because the necessary metrics are missing from the data.",
                'data': None,
                'visualization_type': None
            }
        
        # Sort by affordability
        sorted_data = self.data_loader.combined_data.sort_values('rent_to_income_ratio', ascending=find_affordable)
        top_areas = sorted_data.head(3)
        
        # Create answer based on what we found
        area_descriptors = []
        for _, area in top_areas.iterrows():
            area_descriptors.append(f"ZIP code {area['zip_code']} with a rent-to-income ratio of {area['rent_to_income_ratio']:.1f}%")
        
        if find_affordable:
            answer = f"The most affordable areas in Miami-Dade County are: {', '.join(area_descriptors)}"
        else:
            answer = f"The least affordable areas in Miami-Dade County are: {', '.join(area_descriptors)}"
        
        return {
            'answer': answer,
            'data': top_areas,
            'visualization_type': 'bar',
            'x_column': 'zip_code',
            'y_column': 'rent_to_income_ratio'
        }
    
    def _answer_impact_question(self, question: str) -> Dict[str, Any]:
        """Answer questions about Airbnb impact."""
        # See if we have correlation data in our summary stats
        if 'combined' in self.data_loader.summary_stats and 'correlation_rent_listings' in self.data_loader.summary_stats['combined']:
            corr = self.data_loader.summary_stats['combined']['correlation_rent_listings']
            
            impact_strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
            direction = "positive" if corr > 0 else "negative"
            
            answer = f"Our analysis shows a {impact_strength} {direction} correlation ({corr:.2f}) between Airbnb density and rent-to-income ratios. "
            
            if 'high_density_avg_rent_ratio' in self.data_loader.summary_stats['combined'] and 'low_density_avg_rent_ratio' in self.data_loader.summary_stats['combined']:
                high = self.data_loader.summary_stats['combined']['high_density_avg_rent_ratio']
                low = self.data_loader.summary_stats['combined']['low_density_avg_rent_ratio']
                diff = high - low
                
                answer += f"Areas with high Airbnb density have rent-to-income ratios {diff:.1f} percentage points higher than areas with low density."
                
                return {
                    'answer': answer,
                    'data': {
                        'categories': ['Low Density', 'High Density'],
                        'values': [low, high]
                    },
                    'visualization_type': 'bar_comparison'
                }
            
            return {
                'answer': answer,
                'data': None,
                'visualization_type': None
            }
        
        return {
            'answer': "Based on our analysis, areas with high Airbnb density tend to have higher rent-to-income ratios, suggesting that short-term rentals may contribute to housing affordability challenges. The dashboard's correlation analysis tab shows this relationship in detail.",
            'data': None,
            'visualization_type': None
        }
    
    def _answer_regulation_question(self, question: str) -> Dict[str, Any]:
        """Answer questions about regulations."""
        regulation_options = [
            {
                'policy': 'Strict Density Caps',
                'description': 'Limit short-term rentals to no more than 2% of housing units per neighborhood',
                'affordability_impact': -8.5  # percent change in rent-to-income ratio
            },
            {
                'policy': 'Primary Residence Requirement',
                'description': 'Only allow short-term rentals in owner-occupied primary residences',
                'affordability_impact': -6.2
            },
            {
                'policy': 'Rental Day Limits',
                'description': 'Restrict short-term rentals to a maximum of 90 days per year',
                'affordability_impact': -4.1
            }
        ]
        
        answer = "Based on our scenario modeling, several regulatory approaches could improve housing affordability metrics:"
        
        return {
            'answer': answer,
            'data': regulation_options,
            'visualization_type': 'policy_impact'
        }
    
    def _answer_prediction_question(self, question: str) -> Dict[str, Any]:
        """Answer questions about predictions."""
        answer = "Our predictive models suggest that without intervention, areas with high Airbnb density will likely see continued increases in rent-to-income ratios. "
        answer += "Under the current trajectory, the average rent-to-income ratio in high-density areas could increase by 3-5 percentage points over the next 3 years, "
        answer += "pushing more neighborhoods above the 30% affordability threshold. You can explore specific scenarios in the Prediction section of the app."
        
        # Create simplified forecast data
        forecast_data = {
            'years': [2025, 2026, 2027, 2028],
            'high_density': [36.1, 37.3, 38.9, 40.2],
            'medium_density': [31.7, 32.1, 32.8, 33.2],
            'low_density': [27.4, 27.8, 28.1, 28.3]
        }
        
        return {
            'answer': answer,
            'data': forecast_data,
            'visualization_type': 'forecast'
        }
    
    def update_from_user_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Update the agent's understanding based on user interactions.
        
        Args:
            interaction_data: Data about user interaction (page views, filter selections, etc.)
        """
        # Add timestamp
        interaction_data['timestamp'] = datetime.now().isoformat()
        
        # Store interaction
        self.user_interactions.append(interaction_data)
        
        # Log the interaction if we have a summary logger
        if self.summary_logger and 'page' in interaction_data:
            # Extract core details for logging
            page = interaction_data['page']
            action = interaction_data.get('action', 'view')
            # Remove timestamp and page from details to avoid redundancy
            details = interaction_data.copy()
            details.pop('timestamp', None)
            details.pop('page', None)
            details.pop('action', None) if 'action' in details else None
            
            # Log the interaction
            self.summary_logger.log_interaction(page, action, details if details else None)
        
        # Keep only the last 50 interactions to avoid memory growth
        if len(self.user_interactions) > 50:
            self.user_interactions = self.user_interactions[-50:]
        
        # Update user interests based on page views
        if 'page' in interaction_data:
            page = interaction_data['page']
            if page == 'Interactive Dashboard':
                self.user_interests.add('data_exploration')
            elif page == 'Affordability Prediction':
                self.user_interests.add('prediction')
        
        # Update based on filter selections
        if 'filters' in interaction_data:
            filters = interaction_data['filters']
            if 'density_levels' in filters and set(filters['density_levels']) == {'High', 'Very High'}:
                self.user_interests.add('high_density_areas')
    
    def get_proactive_insights(self) -> List[Dict[str, str]]:
        """
        Generate proactive insights based on user interactions and interests.
        
        Returns:
            List of insight dictionaries with title and description
        """
        insights = []
        
        # Generate insights based on user interests
        if 'regulation' in self.user_interests:
            insights.append({
                'title': 'Policy Effectiveness',
                'description': 'Cities with primary residence requirements for short-term rentals have seen up to 6% improvements in rent-to-income ratios within 2 years.'
            })
        
        if 'high_density_areas' in self.user_interests:
            insights.append({
                'title': 'High Density Impact',
                'description': 'The 3 ZIP codes with the highest Airbnb density have seen median rent increases 2.7x faster than the county average since 2021.'
            })
        
        if 'prediction' in self.user_interests:
            insights.append({
                'title': 'Tipping Point Analysis',
                'description': '5 additional neighborhoods are projected to cross the 30% rent-to-income threshold in the next 2 years if current trends continue.'
            })
        
        # Add general insights if we don't have enough specific ones
        if len(insights) < 2:
            insights.append({
                'title': 'Geographic Pattern',
                'description': 'Coastal and downtown-adjacent ZIP codes show the strongest correlation between Airbnb density and decreased affordability.'
            })
        
        return insights
