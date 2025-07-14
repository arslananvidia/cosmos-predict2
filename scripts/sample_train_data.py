#!/usr/bin/env python3
"""
Sample training data from S3 buckets based on JSON configuration.

This script reads a JSON configuration file specifying input buckets and sample counts,
randomly samples video files from each bucket, and copies the corresponding files
(videos/*.mp4, metas/*.txt, t5_xxl/*.pickle) to an output bucket while maintaining
the directory structure.
"""

import json
import os
import random
import argparse
import boto3
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class S3DataSampler:
    def __init__(self, aws_profile=None, credentials_path=None):
        """Initialize S3 client with specified credentials."""
        self.session = boto3.Session()
        
        if aws_profile:
            self.session = boto3.Session(profile_name=aws_profile)
        elif credentials_path:
            # Read credentials from custom path
            import configparser
            config = configparser.ConfigParser()
            config.read(credentials_path)
            
            if 'default' in config:
                self.session = boto3.Session(
                    aws_access_key_id=config['default']['aws_access_key_id'],
                    aws_secret_access_key=config['default']['aws_secret_access_key'],
                    region_name=config['default'].get('region', 'us-east-1')
                )
        
        self.s3_client = self.session.client('s3')
        
    def parse_s3_path(self, s3_path: str) -> Tuple[str, str]:
        """Parse S3 path into bucket and prefix."""
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ''
        
        return bucket, prefix
    
    def list_video_files(self, bucket: str, prefix: str) -> List[str]:
        """List all .mp4 files in the videos/ subdirectory."""
        videos_prefix = f"{prefix}videos/" if prefix else "videos/"
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=videos_prefix)
            
            video_files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.mp4'):
                            # Get the relative path from videos/ directory
                            relative_path = key[len(videos_prefix):]
                            video_files.append(relative_path)
            
            logger.info(f"Found {len(video_files)} video files in {bucket}/{videos_prefix}")
            return video_files
            
        except Exception as e:
            logger.error(f"Error listing files in {bucket}/{videos_prefix}: {e}")
            return []
    
    def get_corresponding_files(self, video_file: str, bucket: str, prefix: str) -> Dict[str, str]:
        """Get the corresponding meta and t5_xxl files for a video file."""
        # Remove .mp4 extension and get base name
        base_name = video_file.replace('.mp4', '')
        
        # Construct paths for corresponding files
        video_key = f"{prefix}videos/{video_file}" if prefix else f"videos/{video_file}"
        meta_key = f"{prefix}metas/{base_name}.txt" if prefix else f"metas/{base_name}.txt"
        t5_key = f"{prefix}t5_xxl/{base_name}.pickle" if prefix else f"t5_xxl/{base_name}.pickle"
        
        return {
            'video': video_key,
            'meta': meta_key,
            't5_xxl': t5_key
        }
    
    def check_file_exists(self, bucket: str, key: str) -> bool:
        """Check if a file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
    
    def copy_file(self, source_bucket: str, source_key: str, dest_bucket: str, dest_key: str):
        """Copy a file from source to destination in S3."""
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            self.s3_client.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
            logger.debug(f"Copied {source_bucket}/{source_key} to {dest_bucket}/{dest_key}")
        except Exception as e:
            logger.error(f"Error copying {source_bucket}/{source_key} to {dest_bucket}/{dest_key}: {e}")
            raise
    
    def sample_and_copy_files(self, input_bucket_config: Dict, output_bucket: str, output_prefix: str = ""):
        """Sample files from input bucket and copy to output bucket."""
        input_bucket_url = input_bucket_config['input_bucket']
        num_samples = input_bucket_config['number_of_samples']
        
        logger.info(f"Processing {input_bucket_url} - sampling {num_samples} files")
        
        # Parse input bucket path
        bucket, prefix = self.parse_s3_path(input_bucket_url)
        
        # List all video files
        video_files = self.list_video_files(bucket, prefix)
        
        if len(video_files) < num_samples:
            logger.warning(f"Only {len(video_files)} files available, but {num_samples} requested from {input_bucket_url}")
            num_samples = len(video_files)
        
        # Randomly sample files
        sampled_files = random.sample(video_files, num_samples)
        logger.info(f"Sampled {len(sampled_files)} files from {input_bucket_url}")
        
        # Copy sampled files
        successful_copies = 0
        failed_copies = 0
        
        for video_file in sampled_files:
            try:
                # Get corresponding file paths
                file_paths = self.get_corresponding_files(video_file, bucket, prefix)
                
                # Check if all required files exist
                missing_files = []
                for file_type, file_key in file_paths.items():
                    if not self.check_file_exists(bucket, file_key):
                        missing_files.append(f"{file_type}: {file_key}")
                
                if missing_files:
                    logger.warning(f"Skipping {video_file} - missing files: {missing_files}")
                    failed_copies += 1
                    continue
                
                # Copy all files while maintaining directory structure
                for file_type, source_key in file_paths.items():
                    # Construct destination key maintaining directory structure
                    if file_type == 'video':
                        dest_key = f"{output_prefix}videos/{video_file}" if output_prefix else f"videos/{video_file}"
                    elif file_type == 'meta':
                        dest_key = f"{output_prefix}metas/{video_file.replace('.mp4', '.txt')}" if output_prefix else f"metas/{video_file.replace('.mp4', '.txt')}"
                    elif file_type == 't5_xxl':
                        dest_key = f"{output_prefix}t5_xxl/{video_file.replace('.mp4', '.pickle')}" if output_prefix else f"t5_xxl/{video_file.replace('.mp4', '.pickle')}"
                    
                    self.copy_file(bucket, source_key, output_bucket, dest_key)
                
                successful_copies += 1
                
            except Exception as e:
                logger.error(f"Error processing {video_file}: {e}")
                failed_copies += 1
        
        logger.info(f"Completed {input_bucket_url}: {successful_copies} successful, {failed_copies} failed")
        return successful_copies, failed_copies

def main():
    parser = argparse.ArgumentParser(description='Sample training data from S3 buckets')
    parser.add_argument('--config', required=True, help='JSON configuration file with input buckets')
    parser.add_argument('--output-bucket', required=True, help='Output S3 bucket URL (e.g., s3://bucket-name/path/)')
    parser.add_argument('--aws-profile', help='AWS profile name to use')
    parser.add_argument('--credentials-path', help='Path to AWS credentials file (default: ~/.aws/credentials)')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of concurrent workers')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {args.config}: {e}")
        return 1
    
    # Parse output bucket
    try:
        output_bucket, output_prefix = S3DataSampler(args.aws_profile, args.credentials_path).parse_s3_path(args.output_bucket)
    except Exception as e:
        logger.error(f"Error parsing output bucket: {e}")
        return 1
    
    # Initialize sampler
    sampler = S3DataSampler(args.aws_profile, args.credentials_path)
    
    # Process each input bucket
    total_successful = 0
    total_failed = 0
    
    for bucket_config in config:
        try:
            successful, failed = sampler.sample_and_copy_files(bucket_config, output_bucket, output_prefix)
            total_successful += successful
            total_failed += failed
        except Exception as e:
            logger.error(f"Error processing bucket {bucket_config.get('input_bucket', 'unknown')}: {e}")
            total_failed += bucket_config.get('number_of_samples', 0)
    
    logger.info(f"Total processing complete: {total_successful} successful, {total_failed} failed")
    
    return 0 if total_failed == 0 else 1

# Example usage:
# python sample_train_data.py --config input_buckets.json --output-bucket s3://lha-datasets/metropolis/train/v3/combined_1000_1000/ --aws-profile default --credentials-path ~/.aws/credentials
if __name__ == '__main__':
    exit(main())
