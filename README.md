# Miami Housing Impact Hub

## Project Overview
A comprehensive analysis of the relationship between short-term rentals (Airbnb) and housing affordability in Miami-Dade County. This interactive application provides data-driven insights into neighborhood-level impacts, economic implications, and policy recommendations.

## Key Features
- Interactive dashboard with real-time filtering of Miami-Dade neighborhoods
- Detailed Airbnb density visualization with geographic mapping
- Housing affordability metrics with rent-to-income ratio analysis
- Economic impact assessment of short-term rentals
- Data-driven policy recommendations based on empirical analysis

## Live Demo
Access the application at: [Miami Housing Impact Hub](https://miami-housing-impact-hub.streamlit.app/)
*(Link will be active once deployed to Streamlit Cloud)*

## Technical Details

### Data Sources
- Miami-Dade property and housing data
- Airbnb listing data from Inside Airbnb
- Demographic and income data from Census Reporter
- Geographic and spatial data for neighborhood mapping

### Tech Stack
- **Python 3.9+**: Core programming language
- **Streamlit**: Interactive web application framework
- **Pandas/NumPy**: Data processing and analysis
- **Plotly**: Interactive data visualizations
- **Firebase**: Optional data storage and retrieval

## Local Deployment

### Prerequisites
- Python 3.9 or higher
- Git

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/miami-housing-impact-hub.git
   cd miami-housing-impact-hub
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run streamlit_app.py
   ```

## Streamlit Cloud Deployment

### Preparation Steps
1. Ensure your repository is hosted on GitHub
2. Data files must be included in the repository or accessible via external URLs
3. All dependencies are listed in requirements.txt
4. The `streamlit_app.py` file is in the repository root

### Deployment Process
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect to your GitHub repository
3. Select the repository, branch, and the `streamlit_app.py` file
4. Deploy and share the generated URL

## Acknowledgements
- Data Mining course at FIU
- Miami-Dade County Open Data Portal
- Inside Airbnb dataset providers
- Census Reporter for demographic data

## License
MIT License

## Contact
For questions or feedback, please contact [Your Name](mailto:your.email@example.com).
