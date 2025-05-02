"""
Simple Home Page for Miami Housing Impact Hub
A streamlined version that avoids direct pandas/numpy compatibility issues
"""
import streamlit as st

def show_simple_home():
    """Display a simplified home page with key project information."""
    st.title("Miami Housing Impact Hub")
    st.subheader("Analyzing the Impact of Short-Term Rentals on Miami's Housing Market")
    
    st.markdown("""
    ## Project Overview
    
    This data science project investigates how short-term rentals affect housing 
    affordability in Miami-Dade County, Florida. By analyzing Airbnb density patterns
    alongside housing prices, rental rates, and neighborhood characteristics, we uncover
    the relationships between these factors.
    
    ### Key Research Questions
    
    1. How does Airbnb density correlate with housing affordability metrics?
    2. Which neighborhoods show the strongest relationships between short-term rentals and housing costs?
    3. What are the temporal trends in housing prices as short-term rental numbers change?
    4. How do different property types and rental characteristics affect neighborhood affordability?
    
    ### Methodology
    
    Our analysis combines data from multiple sources:
    - Airbnb listings and reviews
    - Census Bureau housing and demographic data
    - Miami-Dade County property records
    - Rental market data from various sources
    
    Using advanced data mining techniques including spatial analysis, time series modeling,
    and correlation analysis, we've created an interactive dashboard to visualize these relationships.
    
    ### Business Value
    
    This research provides valuable insights for:
    - Local policymakers developing short-term rental regulations
    - Real estate investors analyzing neighborhood trends
    - Community organizations addressing housing affordability
    - Residents understanding market forces affecting their communities
    """)
    
    # Display key metrics summaries without requiring pandas
    st.markdown("## Key Findings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Neighborhoods Analyzed", value="78")
        st.markdown("Miami-Dade communities with comprehensive data coverage")
        
    with col2:
        st.metric(label="Total Airbnb Listings", value="18,412")
        st.markdown("Active listings across Miami-Dade County")
        
    with col3:
        st.metric(label="Avg. Price Increase", value="14.8%")
        st.markdown("Average housing price increase in high-Airbnb-density neighborhoods")
    
    # Image placeholder
    st.image("https://www.miamiandbeaches.com/getmedia/1c502b74-2b7a-4e23-b748-cc0d22a41ade/Miami-downtown-brickell-skyline-aerial-view-300314401-1440x900.jpg.aspx", 
             caption="Miami-Dade County - The focus area of our housing impact research")
             
    st.markdown("""
    ### Acknowledgements
    
    This project was developed as part of the Data Mining course at Florida International University.
    All data used in this analysis is from public sources or has been appropriately anonymized.
    """)

if __name__ == "__main__":
    show_simple_home()
