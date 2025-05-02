"""
Standardizes Miami neighborhood names across datasets for more reliable merging.
- Lowercases, strips whitespace, and applies a mapping for common variants.
- Add more mappings as needed for project growth.
"""
import re

# Add more mappings as needed
NEIGHBORHOOD_MAP = {
    'brickell district': 'brickell',
    'brickell area': 'brickell',
    'brickell': 'brickell',
    'downtown miami': 'downtown',
    'miami river': 'downtown',
    'wynwood art district': 'wynwood',
    'wynwood': 'wynwood',
    'little havana': 'little havana',
    'west little havana': 'little havana',
    'east little havana': 'little havana',
    'flagami': 'flagami',
    'allapattah': 'allapattah',
    'liberty city': 'liberty city',
    'overtown': 'overtown',
    'coconut grove': 'coconut grove',
    'south coconut grove': 'coconut grove',
    'design district': 'design district',
    'media and entertainment district': 'media and entertainment district',
    # Add more as needed
}

def standardize_neighborhood(name):
    if not isinstance(name, str):
        return ''
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9 ]', '', name)
    return NEIGHBORHOOD_MAP.get(name, name)

# Vectorized version for pandas

def standardize_neighborhood_series(series):
    return series.apply(standardize_neighborhood)
