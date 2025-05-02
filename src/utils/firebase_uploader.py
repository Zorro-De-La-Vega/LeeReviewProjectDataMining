#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firebase Storage Uploader

This utility script uploads data files to Firebase Storage.
It helps manage large files that shouldn't be committed to GitHub.
"""

import firebase_admin
from firebase_admin import credentials, storage
import os
import argparse
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_file_to_firebase(local_path, firebase_path, bucket=None):
    """
    Upload a single file to Firebase Storage.
    
    Args:
        local_path (str): Path to the local file to upload
        firebase_path (str): Path where to store the file in Firebase
        bucket: Firebase storage bucket
        
    Returns:
        bool: True if successful, False otherwise
    """
    if bucket is None:
        return False
        
    try:
        # Create a blob and upload the file
        blob = bucket.blob(firebase_path)
        blob.upload_from_filename(local_path)
        
        # Set appropriate Cache-Control headers for data files
        blob.cache_control = 'public, max-age=3600'  # 1 hour
        blob.patch()
        
        # Make the blob publicly accessible
        blob.make_public()
        
        # Get the public URL
        public_url = blob.public_url
        
        logger.info(f"✓ Uploaded {local_path} to {firebase_path}")
        logger.info(f"  Public URL: {public_url}")
        return True
    except Exception as e:
        logger.error(f"Error uploading {local_path}: {e}")
        return False

def upload_data_to_firebase(cred_path, data_dir, bucket_name=None, include_patterns=None):
    """
    Upload data files to Firebase Storage.
    
    Args:
        cred_path: Path to Firebase credentials JSON file
        data_dir: Directory containing data files to upload
        bucket_name: Firebase Storage bucket name (optional if in credentials)
        include_patterns: List of file patterns to include (e.g., ['.csv', '.json'])
    """
    # Initialize Firebase
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            if bucket_name:
                firebase_admin.initialize_app(cred, {
                    'storageBucket': bucket_name
                })
            else:
                firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return
    
    # Default patterns to include
    if include_patterns is None:
        include_patterns = ['.csv']
    
    # Get bucket
    try:
        bucket = storage.bucket()
    except Exception as e:
        logger.error(f"Failed to access storage bucket: {e}")
        return
    
    # Track stats
    stats = {
        'uploaded': 0,
        'failed': 0,
        'skipped': 0,
        'total_size': 0
    }
    
    # Store public URLs for direct access config
    public_urls = {}
    
    # Walk through the data directory
    for root, _, files in os.walk(data_dir):
        for file in files:
            # Check if file matches include patterns
            if not any(file.endswith(pattern) for pattern in include_patterns):
                stats['skipped'] += 1
                continue
                
            local_path = os.path.join(root, file)
            file_size = os.path.getsize(local_path) / (1024 * 1024)  # Size in MB
            
            # Determine the Firebase path (remove the data_dir prefix)
            rel_path = os.path.relpath(local_path, data_dir)
            firebase_path = rel_path.replace('\\', '/')
            
            logger.info(f"Uploading {local_path} ({file_size:.2f} MB) to Firebase as {firebase_path}")
            
            # Upload file
            success = upload_file_to_firebase(local_path, firebase_path, bucket)
            
            if success:
                stats['uploaded'] += 1
                stats['total_size'] += file_size
                
                # Store the public URL
                blob = bucket.blob(firebase_path)
                public_urls[firebase_path] = blob.public_url
            else:
                stats['failed'] += 1
    
    # Create a JSON file with direct download URLs
    try:
        # Save to project root
        if public_urls:
            output_path = Path(data_dir).parent / "firebase-urls.json"
            with open(output_path, 'w') as f:
                import json
                json.dump(public_urls, f, indent=2)
            logger.info(f"Saved {len(public_urls)} direct download URLs to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save URL config: {e}")
    
    # Print summary
    logger.info("\n=== Upload Summary ===")
    logger.info(f"Files uploaded: {stats['uploaded']}")
    logger.info(f"Files failed: {stats['failed']}")
    logger.info(f"Files skipped: {stats['skipped']}")
    logger.info(f"Total data uploaded: {stats['total_size']:.2f} MB")
    
    # Check if there were any failures
    if stats['failed'] > 0:
        logger.warning("Some files failed to upload. Check the logs for details.")

def create_gitignore_for_large_files(project_dir, min_size_mb=90):
    """
    Create a .gitignore entry for data files over a certain size.
    
    Args:
        project_dir: Project root directory
        min_size_mb: Minimum size in MB to ignore (default: 90MB)
    """
    large_files = []
    data_dir = os.path.join(project_dir, "data")
    
    # Find large files
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb >= min_size_mb:
                    # Get relative path to project dir
                    rel_path = os.path.relpath(file_path, project_dir)
                    large_files.append((rel_path, size_mb))
            except:
                continue
    
    # Sort by size (largest first)
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    if not large_files:
        logger.info("No files larger than 90MB found")
        return
    
    # Create gitignore entries
    logger.info(f"Found {len(large_files)} files larger than {min_size_mb}MB:")
    
    gitignore_path = os.path.join(project_dir, ".gitignore")
    gitignore_entries = []
    
    # Read existing gitignore
    existing_entries = set()
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            existing_entries = set(line.strip() for line in f.readlines())
    
    # Add new entries
    with open(gitignore_path, 'a') as f:
        f.write("\n# Large data files (>90MB)\n")
        for file_path, size in large_files:
            normalized_path = file_path.replace('\\', '/')
            if normalized_path not in existing_entries:
                f.write(f"{normalized_path}  # {size:.2f}MB\n")
                gitignore_entries.append(normalized_path)
                logger.info(f"  - {normalized_path} ({size:.2f}MB)")
    
    if gitignore_entries:
        logger.info(f"Added {len(gitignore_entries)} entries to .gitignore")
    else:
        logger.info("All large files are already in .gitignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload data files to Firebase Storage")
    
    parser.add_argument("--cred", type=str, help="Path to Firebase credentials JSON file",
                       default="firebase-credentials.json")
    parser.add_argument("--data", type=str, help="Directory containing data files to upload",
                       default="data")
    parser.add_argument("--bucket", type=str, help="Firebase Storage bucket name")
    parser.add_argument("--include", type=str, nargs="+", help="File patterns to include (default: .csv)",
                       default=[".csv"])
    parser.add_argument("--gitignore", action="store_true", help="Create .gitignore entries for large files")
    parser.add_argument("--min-size", type=float, help="Minimum size in MB to ignore (default: 90MB)", 
                       default=90)
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Path to Firebase credentials JSON
    cred_path = project_root / args.cred if not os.path.isabs(args.cred) else args.cred
    
    # Data directory to upload
    data_dir = project_root / args.data if not os.path.isabs(args.data) else args.data
    
    # Create gitignore entries if requested
    if args.gitignore:
        create_gitignore_for_large_files(project_root, args.min_size)
    
    # Check if paths exist
    if not os.path.exists(cred_path):
        logger.error(f"Firebase credentials file does not exist: {cred_path}")
        sys.exit(1)
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Upload the data
    upload_data_to_firebase(cred_path, data_dir, args.bucket, args.include)
