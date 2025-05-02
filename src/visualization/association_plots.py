# src/visualization/association_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def plot_association_rules(rules_df, support_col='support', confidence_col='confidence', lift_col='lift', title='Association Rules: Support vs. Confidence'):
    """
    Creates a scatter plot to visualize association rules.

    Args:
        rules_df (pd.DataFrame): DataFrame containing association rules
                                 (output from mlxtend.association_rules).
                                 Must contain columns specified by support_col,
                                 confidence_col, and lift_col.
        support_col (str): Name of the support column.
        confidence_col (str): Name of the confidence column.
        lift_col (str): Name of the lift column (used for color intensity).
        title (str): Title for the plot.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object, or None if input is invalid.
    """
    if not isinstance(rules_df, pd.DataFrame) or rules_df.empty:
        logger.warning("Input rules_df is empty or not a DataFrame. Cannot generate plot.")
        return None
    if not all(col in rules_df.columns for col in [support_col, confidence_col, lift_col]):
        logger.error(f"Rules DataFrame missing required columns: {support_col}, {confidence_col}, {lift_col}")
        return None

    # Ensure numeric types for plotting
    try:
        rules_df[support_col] = pd.to_numeric(rules_df[support_col])
        rules_df[confidence_col] = pd.to_numeric(rules_df[confidence_col])
        rules_df[lift_col] = pd.to_numeric(rules_df[lift_col])
    except ValueError as e:
        logger.error(f"Could not convert plot columns to numeric: {e}")
        return None


    logger.info(f"Generating association rules plot for {len(rules_df)} rules.")

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        # Create the scatter plot
        scatter = sns.scatterplot(
            data=rules_df,
            x=support_col,
            y=confidence_col,
            size=lift_col,  # Optional: Use lift for size
            hue=lift_col,   # Use lift for color
            palette='viridis', # Color map
            ax=ax,
            # legend='auto' # Let seaborn handle legend - sometimes needs help
            sizes=(20, 200), # Control point sizes
            alpha=0.7 # Add transparency
        )

        ax.set_title(title)
        ax.set_xlabel("Support")
        ax.set_ylabel("Confidence")

        # Improve legend
        norm = plt.Normalize(rules_df[lift_col].min(), rules_df[lift_col].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])

        # Remove the old legend if seaborn created one
        if ax.get_legend():
             ax.get_legend().remove()

        # Add the colorbar
        cbar = fig.colorbar(sm, ax=ax, label='Lift')

        plt.tight_layout()
        logger.info("Association rules plot generated successfully.")
        return fig

    except Exception as e:
        logger.error(f"Error generating scatter plot: {e}", exc_info=True)
        return None


# Example Usage (optional, for testing)
if __name__ == '__main__':
    # Create dummy data
    dummy_rules = pd.DataFrame({
        'antecedents': [frozenset({'A'}), frozenset({'B'}), frozenset({'A', 'B'}), frozenset({'C'})],
        'consequents': [frozenset({'C'}), frozenset({'A'}), frozenset({'C'}), frozenset({'A'})],
        'support': [0.1, 0.2, 0.05, 0.3],
        'confidence': [0.5, 0.6, 0.8, 0.7],
        'lift': [1.5, 1.2, 2.0, 1.1],
        'leverage': [0.01, 0.02, 0.005, 0.03],
        'conviction': [1.2, 1.3, 1.5, 1.1]
    })
    fig = plot_association_rules(dummy_rules)
    if fig:
        plt.show()
