"""
Script: download_census_granular.py
Purpose: Download granular (tract-level) ACS data for Miami-Dade County using the censusdata Python package.
Outputs: data/raw/census/acs5_miamidade_tracts.csv

Instructions:
- Requires: pip install censusdata pandas
- Edit the VARIABLES list for your needs (see ACS variable list online)
"""
import censusdata
import pandas as pd

# Miami-Dade County FIPS codes
STATE_FIPS = '12'  # Florida
COUNTY_FIPS = '086'  # Miami-Dade
YEAR = 2022  # Change to latest available if needed

# List of ACS 5-year variables to download (edit as needed)
VARIABLES = [
    'B19013_001E',  # Median household income
    'B01003_001E',  # Total population
    'B25064_001E',  # Median gross rent
    'B25077_001E',  # Median home value
    'B25002_003E',  # Vacant housing units
    # Add more variables as needed
]

print("Downloading ACS 5-year data for Miami-Dade tracts...")
df = censusdata.download(
    'acs5',
    YEAR,
    censusdata.censusgeo([('state', STATE_FIPS), ('county', COUNTY_FIPS), ('tract', '*')]),
    VARIABLES
)

# Reset index to get GEOID as a column
print("Formatting output DataFrame...")
df = df.reset_index()
df.rename(columns={'index': 'censusgeo'}, inplace=True)
df['tract'] = df['censusgeo'].apply(lambda x: x.geo[2][1])
df['county'] = df['censusgeo'].apply(lambda x: x.geo[1][1])
df['state'] = df['censusgeo'].apply(lambda x: x.geo[0][1])
df['GEOID'] = df['state'] + df['county'] + df['tract']

# Save to CSV
output_path = r'data/raw/census/acs5_miamidade_tracts.csv'
df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
