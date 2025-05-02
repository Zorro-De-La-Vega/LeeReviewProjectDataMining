"""
Data optimization script for Streamlit Cloud deployment.
This script creates smaller, optimized versions of key datasets for cloud deployment.
"""
import pandas as pd
import os
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to path for imports
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

# Define paths
data_dir = project_root / "data"
processed_dir = data_dir / "processed"
cloud_dir = data_dir / "cloud_optimized"

# Create cloud optimized directory if it doesn't exist
os.makedirs(cloud_dir, exist_ok=True)

def optimize_csv_for_cloud(input_path, output_path, sample_size=None, compression='gzip'):
    """
    Optimize a CSV file for cloud deployment by:
    1. Optionally taking a sample if the dataset is large
    2. Converting to efficient dtypes
    3. Applying compression
    
    Args:
        input_path (Path): Path to input CSV
        output_path (Path): Path to save optimized CSV
        sample_size (int, optional): Number of rows to sample (None = all rows)
        compression (str): Compression method ('gzip' recommended for cloud)
    """
    try:
        logger.info(f"Optimizing {input_path.name}...")
        
        # Read the dataset
        df = pd.read_csv(input_path)
        original_size = df.memory_usage(deep=True).sum()
        
        # Apply sampling if requested
        if sample_size is not None and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} rows from {len(df)} total rows")
            df = df.sample(sample_size, random_state=42)
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Save with compression
        df.to_csv(output_path, index=False, compression=compression)
        
        # Log size reduction
        optimized_size = os.path.getsize(output_path)
        reduction = (1 - optimized_size / original_size) * 100
        logger.info(f"Optimized {input_path.name}: {original_size/1e6:.2f}MB → {optimized_size/1e6:.2f}MB ({reduction:.1f}% reduction)")
        
        return True
    except Exception as e:
        logger.error(f"Error optimizing {input_path}: {str(e)}")
        return False

def main():
    """Optimize key datasets for Streamlit Cloud deployment."""
    # List of key files to optimize with their sampling parameters
    files_to_optimize = [
        {"path": processed_dir / "miami_dade_merged_data.csv", "sample": None},  # Keep all rows
        {"path": processed_dir / "airbnb" / "airbnb_cleaned.csv", "sample": 5000},  # Sample if large
    ]
    
    # Process each file
    for file_info in files_to_optimize:
        input_path = file_info["path"]
        if input_path.exists():
            output_path = cloud_dir / f"{input_path.stem}_cloud{input_path.suffix}.gz"
            optimize_csv_for_cloud(input_path, output_path, file_info["sample"])
        else:
            logger.warning(f"File not found: {input_path}")
    
    # Create a manifest of optimized files
    with open(cloud_dir / "cloud_data_manifest.txt", "w") as f:
        f.write("# Cloud-optimized data files for Streamlit deployment\n")
        f.write("# Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        for file in cloud_dir.glob("*.gz"):
            size_mb = os.path.getsize(file) / 1e6
            f.write(f"{file.name}: {size_mb:.2f}MB\n")
    
    logger.info(f"Data optimization complete. Optimized files saved to {cloud_dir}")

if __name__ == "__main__":
    main()
