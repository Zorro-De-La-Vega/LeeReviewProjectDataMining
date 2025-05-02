# src/models/association_rules.py

import pandas as pd
import logging
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import ast  # For safely evaluating string representations of lists

logger = logging.getLogger(__name__)

class AmenityAssociationMiner:
    """
    Performs association rule mining on Airbnb listing amenities
    to find relationships with a target variable (e.g., price category).
    """

    def __init__(self, listings_df):
        """
        Initializes the miner with the listings DataFrame.

        Args:
            listings_df (pd.DataFrame): DataFrame containing listings data,
                                        including 'amenities' and target columns.
        """
        if 'amenities' not in listings_df.columns:
            raise ValueError("Listings DataFrame must contain an 'amenities' column.")
        self.listings_df = listings_df.copy()
        logger.info("AmenityAssociationMiner initialized.")

    def _parse_amenities(self, amenities_str):
        """Safely parses the string representation of the amenities list."""
        try:
            # Handle potential NaN or non-string inputs
            if pd.isna(amenities_str) or not isinstance(amenities_str, str):
                return []
            # Replace single quotes with double quotes for valid JSON-like format if necessary
            # Evaluate the string literal to get the list
            amenities_list = ast.literal_eval(amenities_str)
            if isinstance(amenities_list, list):
                 # Basic cleaning: remove empty strings and strip whitespace
                return [str(item).strip() for item in amenities_list if str(item).strip()]
            else:
                logger.warning(f"Parsed amenities is not a list: {amenities_list}. Returning empty list.")
                return []
        except (ValueError, SyntaxError, TypeError) as e:
            logger.warning(f"Could not parse amenities string: '{amenities_str}'. Error: {e}. Returning empty list.")
            return []

    def preprocess_data(self, target_col='price', n_bins=3, target_labels=None, min_amenity_freq=10):
        """
        Prepares the data for Apriori by parsing amenities, discretizing the target,
        and creating a one-hot encoded transaction matrix.

        Args:
            target_col (str): The column to use as the target outcome (e.g., 'price', 'review_scores_rating').
            n_bins (int): The number of bins to discretize the target variable into.
            target_labels (list, optional): Custom labels for the target bins. If None, default labels are generated.
            min_amenity_freq (int): Minimum number of listings an amenity must appear in to be included.

        Returns:
            pd.DataFrame: One-hot encoded DataFrame suitable for Apriori, or None if preprocessing fails.
        """
        logger.info(f"Starting data preprocessing for target '{target_col}' with {n_bins} bins and min amenity freq {min_amenity_freq}.")

        if target_col not in self.listings_df.columns:
            logger.error(f"Target column '{target_col}' not found in listings DataFrame.")
            return None

        # --- 1. Parse Amenities ---
        logger.debug("Parsing amenities...")
        parsed_amenities = self.listings_df['amenities'].apply(self._parse_amenities)
        self.listings_df['parsed_amenities'] = parsed_amenities
        logger.debug(f"Amenities parsed. Example: {parsed_amenities.iloc[0]}")

        # --- 2. Filter Amenities by Frequency ---
        logger.debug("Filtering amenities by frequency...")
        all_amenities = [amenity for sublist in parsed_amenities for amenity in sublist]
        amenity_counts = pd.Series(all_amenities).value_counts()
        frequent_amenities = amenity_counts[amenity_counts >= min_amenity_freq].index.tolist()
        if not frequent_amenities:
            logger.warning("No amenities meet the minimum frequency threshold.")
            return None
        logger.info(f"Keeping {len(frequent_amenities)} amenities (min freq: {min_amenity_freq}).")

        # Filter parsed amenities list to keep only frequent ones
        filtered_parsed_amenities = parsed_amenities.apply(
            lambda amenities: [a for a in amenities if a in frequent_amenities]
        )
        logger.debug("Amenities filtered.")

        # --- 3. Discretize Target Variable ---
        logger.debug(f"Discretizing target column '{target_col}'...")
        target_data = self.listings_df[target_col].dropna()
        if target_data.empty:
             logger.error(f"Target column '{target_col}' contains only NaN values.")
             return None

        try:
            # Use qcut for quantile-based binning, handling potential duplicate edges
            target_bins, bin_edges = pd.qcut(target_data, q=n_bins, labels=False, retbins=True, duplicates='drop')
            actual_n_bins = len(bin_edges) - 1 # Number of bins might be reduced due to duplicate edges
            logger.info(f"Target column '{target_col}' discretized into {actual_n_bins} bins using qcut.")

            if not target_labels:
                target_labels = [f"{target_col}_Bin{i+1}" for i in range(actual_n_bins)]
            elif len(target_labels) != actual_n_bins:
                 logger.warning(f"Provided target_labels count ({len(target_labels)}) "
                                f"doesn't match actual bins ({actual_n_bins}). Using default labels.")
                 target_labels = [f"{target_col}_Bin{i+1}" for i in range(actual_n_bins)]

            # Map numerical bins to labels
            target_bins = target_bins.map(dict(enumerate(target_labels)))
            self.listings_df['target_category'] = target_bins
            logger.debug(f"Target categories created: {target_labels}")
            logger.debug(f"Target category distribution:\n{self.listings_df['target_category'].value_counts(dropna=False)}")


        except Exception as e:
            logger.error(f"Failed to discretize target column '{target_col}': {e}")
            return None

        # --- 4. Create Transaction List ---
        logger.debug("Creating transaction list...")
        # Combine frequent amenities with the target category for each listing
        transactions = []
        for i, amenities in filtered_parsed_amenities.items():
            target_cat = self.listings_df.loc[i, 'target_category']
            if pd.notna(target_cat): # Only include listings with a valid target category
                transactions.append(list(amenities) + [target_cat])
        logger.debug(f"Created {len(transactions)} transactions.")

        if not transactions:
            logger.error("No transactions could be created. Check data and preprocessing steps.")
            return None

        # --- 5. One-Hot Encode Transactions ---
        logger.debug("Performing one-hot encoding...")
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        transaction_df = pd.DataFrame(te_ary, columns=te.columns_)
        logger.info(f"Transaction matrix created with shape: {transaction_df.shape}")
        logger.debug(f"Transaction matrix columns: {transaction_df.columns.tolist()}")

        return transaction_df

    def find_rules(self, transaction_df, min_support=0.01, min_confidence=0.1, metric='lift', min_lift=1.0):
        """
        Applies the Apriori algorithm and generates association rules.

        Args:
            transaction_df (pd.DataFrame): The one-hot encoded transaction data.
            min_support (float): The minimum support threshold for frequent itemsets.
            min_confidence (float): The minimum confidence threshold for rules.
            metric (str): The metric to evaluate rules ('confidence', 'lift', etc.).
            min_lift (float): The minimum lift threshold for rules (used with metric='lift').

        Returns:
            pd.DataFrame: DataFrame containing the discovered association rules,
                          sorted by the specified metric, or None if no rules found.
        """
        if transaction_df is None or transaction_df.empty:
            logger.error("Transaction DataFrame is empty or None. Cannot find rules.")
            return None

        logger.info(f"Finding frequent itemsets with min_support={min_support}...")
        try:
            frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)
            logger.info(f"Found {len(frequent_itemsets)} frequent itemsets.")

            if frequent_itemsets.empty:
                logger.warning("No frequent itemsets found with the given support threshold.")
                return pd.DataFrame() # Return empty DataFrame instead of None

            logger.info(f"Generating association rules with min_{metric}={min_confidence if metric=='confidence' else min_lift}...")
            # Determine threshold based on metric
            min_threshold = min_lift if metric == 'lift' else min_confidence

            rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
            logger.info(f"Generated {len(rules)} rules.")

            if rules.empty:
                logger.warning(f"No rules found meeting the {metric}>={min_threshold} threshold.")
                return pd.DataFrame()

            # Sort rules for better presentation
            rules = rules.sort_values(by=metric, ascending=False).reset_index(drop=True)
            logger.debug(f"Top 5 rules by {metric}:\n{rules.head().to_string()}")
            return rules

        except Exception as e:
            logger.error(f"Error during Apriori or rule generation: {e}", exc_info=True)
            return None
