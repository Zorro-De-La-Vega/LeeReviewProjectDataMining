"""
Helper functions for feature placeholders and special displays in the Streamlit app.
"""
import streamlit as st

def show_feature_coming_soon(data_loader, agent=None):
    """Display a placeholder for features that are coming soon due to missing dependencies.
    
    Args:
        data_loader: DataLoader instance
        agent: Optional HousingImpactAgent instance
    """
    st.subheader("Feature Coming Soon")
    st.info("This feature is currently unavailable due to missing dependencies. We're working to resolve this issue.")
    st.markdown("""
    ### Why is this feature unavailable?
    
    The Geospatial Hotspot Analysis requires specialized Python packages (`pysal`, `contextily`, etc.) 
    that couldn't be installed due to disk space limitations. 
    
    To use this feature, you'll need to:
    1. Free up disk space on your system
    2. Install the required packages manually using: `pip install pysal contextily`
    3. Restart the application
    """)
    
    # Show what this feature would provide
    st.subheader("What You're Missing")
    st.markdown("""
    **Geospatial Hotspot Analysis** would allow you to:
    - Identify statistically significant clusters of high and low values
    - Compute spatial autocorrelation statistics (Moran's I)
    - Apply DBSCAN or Getis-Ord Gi* for hotspot detection
    - Visualize results with interactive heat maps
    """)
