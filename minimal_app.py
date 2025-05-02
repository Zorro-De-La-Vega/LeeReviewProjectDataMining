"""
Miami Housing Impact Hub - Minimal Entry Point for Streamlit Cloud
This file serves as a simplified entry point to avoid pandas import issues.
"""
import streamlit as st
import os
import sys

# Add the current directory to the path so we can import from modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Streamlit app header
st.set_page_config(
    page_title="Miami Housing Impact Hub",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Miami Housing Impact Hub")
st.subheader("Analyzing the Impact of Short-Term Rentals on Miami's Housing Market")

# Attempt to load the main app with error handling
try:
    # Only import pandas where it's actually needed
    st.write("Loading application components...")
    
    # Try a safer import approach that avoids direct pandas imports at the module level
    from src.app.pages.home import show_home
    
    # Display the main content
    show_home()
    
except Exception as e:
    st.error(f"Error loading the application: {str(e)}")
    st.write("Technical details for troubleshooting:")
    st.code(f"Error type: {type(e).__name__}\nError details: {str(e)}")
    
    # Show system information that might help with debugging
    import platform
    st.write("### System Information")
    st.write(f"Python version: {platform.python_version()}")
    st.write(f"Platform: {platform.platform()}")
    
    # Provide fallback content
    st.write("### Project Overview")
    st.write("""
    The Miami Housing Impact Hub analyzes the relationship between short-term rentals 
    and housing affordability in Miami-Dade County. This research investigates how 
    Airbnb density correlates with housing prices, rent affordability, and neighborhood 
    characteristics across Miami's diverse communities.
    """)
