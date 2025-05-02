"""
Miami Housing Impact Hub - Streamlit App Entry Point
This file serves as the main entry point for Streamlit Cloud deployment.
"""
import streamlit as st
import sys
import os

# Add the current directory to the path so we can import from modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app
try:
    from src.app.app import main
    
    # Run the main app function
    if __name__ == "__main__":
        main()
        
except Exception as e:
    st.error(f"Error loading the application: {e}")
    st.write("### Troubleshooting Information")
    st.write("If you're seeing this error, the application encountered an issue during startup.")
    st.write("#### Details:")
    st.code(f"Error type: {type(e).__name__}\nError message: {str(e)}")
    
    # Display the directory structure for debugging
    st.write("#### Project Structure:")
    
    def list_files(startpath):
        output = []
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * level
            output.append(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                output.append(f"{subindent}{f}")
        return '\n'.join(output)
    
    st.code(list_files('.'))
