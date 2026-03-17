#!/usr/bin/env python3
"""
Direct Sentinel-1 Data Download Script for TACC
Downloads Sentinel-1 GRD data directly from Google Earth Engine to TACC scratch
NO Google Cloud Storage or Drive needed, no chunk subfolders
"""

import ee
import requests
import os
import sys
import time
from datetime import datetime, timedelta

def authenticate_gee(service_account_email, key_file_path):
    """Authenticate with Google Earth Engine"""
    try:
        credentials = ee.ServiceAccountCredentials(service_account_email, key_file_path)
        ee.Initialize(credentials)
        print("✓ Successfully authenticated with Google Earth Engine")
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False

def create_aoi(bounds):
    """Create Area of Interest geometry"""
    min_lon, min_lat, max_lon, max_lat = bounds
    
    aoi = ee.Geometry.Polygon([[
        [min_lon, min_lat],
        [max_lon, min_lat], 
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat]
    ]])
    return aoi

def download_file_from_url(url, output_path):
    """Download file from URL with progress"""
    try:
        print(f"Downloading to: {output_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\n✓ Downloaded: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def download_sentinel1_images_direct(aoi, start_date, end_date, output_dir, scale=1000, max_images=50):
    """Download Sentinel-1 images directly via URL"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Downloading Sentinel-1 data from {start_date} to {end_date}")
        print(f"Scale: {scale}m")
        print(f"Output directory: {output_dir}")
        
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi) \
            .select(['VV', 'VH']) \
            .limit(max_images)
        
        collection_size = collection.size().getInfo()
        print(f"Found {collection_size} images in collection (limited to {max_images})")
        
        if collection_size == 0:
            print("No images found for the specified criteria")
            return False
        
        image_list = collection.toList(collection_size)
        download_count = 0
        
        for i in range(collection_size):
            try:
                print(f"\n--- Processing image {i+1}/{collection_size} ---")
                
                image = ee.Image(image_list.get(i))
                props = image.getInfo()['properties']
                image_id = props.get('system:index', f'image_{i}')
                date_str = props.get('system:time_start', 'unknown')
                
                safe_image_id = image_id.replace('/', '_').replace(':', '_')
                
                print(f"Image ID: {image_id}")
                print(f"Date: {date_str}")
                
                url = image.getDownloadURL({
                    'name': f's1_{safe_image_id}',
                    'region': aoi,
                    'scale': scale,
                    'crs': 'EPSG:4326',
                    'format': 'GEO_TIFF'
                })
                
                output_filename = f's1_{safe_image_id}_{start_date}_{end_date}.tif'
                output_path = os.path.join(output_dir, output_filename)
                
                if download_file_from_url(url, output_path):
                    download_count += 1
                
                time.sleep(2)
                
            except Exception as e:
                print(f"✗ Error processing image {i+1}: {e}")
                continue
        
        print(f"\n✓ Successfully processed {download_count}/{collection_size} images")
        
        tiff_files = [f for f in os.listdir(output_dir) if f.endswith(('.tif', '.tiff'))]
        print(f"✓ Total GeoTIFF files: {len(tiff_files)}")
        
        return download_count > 0
        
    except Exception as e:
        print(f"✗ Error in download process: {e}")
        return False

def download_composite_image(aoi, start_date, end_date, output_dir, scale=1000):
    """Download a single composite/mosaic image as GeoTIFF"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating composite image from {start_date} to {end_date}")
        print(f"Scale: {scale}m")
        
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(start_date, end_date) \
            .filterBounds(aoi) \
            .select(['VV', 'VH'])
        
        collection_size = collection.size().getInfo()
        print(f"Found {collection_size} images for composite")
        
        if collection_size == 0:
            print("No images found")
            return False
        
        composite = collection.median().clip(aoi)
        
        url = composite.getDownloadURL({
            'name': f'sentinel1_composite_{start_date}_{end_date}',
            'region': aoi,
            'scale': scale,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF'
        })
        
        output_filename = f'sentinel1_composite_{start_date}_{end_date}.tif'
        output_path = os.path.join(output_dir, output_filename)
        
        return download_file_from_url(url, output_path)
        
    except Exception as e:
        print(f"✗ Error creating composite: {e}")
        return False

def split_date_range(start_date, end_date, chunk_days=30):
    """Split large date ranges into smaller chunks"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    chunks = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime('%Y-%m-%d'),
            chunk_end.strftime('%Y-%m-%d')
        ))
        current = chunk_end + timedelta(days=1)
    
    return chunks

def main():
    # Configuration
    SERVICE_ACCOUNT = 'gee-829@elliptical-flow-451715-i9.iam.gserviceaccount.com'
    KEY_FILE = '/scratch/10778/prabhjotschugh/sentinel1_data/elliptical-flow-451715-i9-5a626e8f9544.json'
    
    # Study area bounds [min_lon, min_lat, max_lon, max_lat]
    BOUNDS = [-105.24894905306857, 31.429758967461108, -102.13960634966884, 33.34119795385639]
    
    # Date range
    START_DATE = '2024-05-27'
    END_DATE = '2024-11-11'
    
    # Download parameters
    SCALE = 1000
    OUTPUT_DIR = '/scratch/10778/prabhjotschugh/sentinel1_data/data'
    DOWNLOAD_MODE = 'composite'  # 'individual' or 'composite'
    MAX_IMAGES_PER_CHUNK = 10
    
    print("=" * 60)
    print("DIRECT SENTINEL-1 DATA DOWNLOAD SCRIPT")
    print("=" * 60)
    print(f"Download mode: {DOWNLOAD_MODE}")
    
    if not os.path.exists(KEY_FILE):
        print(f"✗ Key file not found: {KEY_FILE}")
        sys.exit(1)
    
    if not authenticate_gee(SERVICE_ACCOUNT, KEY_FILE):
        sys.exit(1)
    
    aoi = create_aoi(BOUNDS)
    print(f"✓ Created AOI: {BOUNDS}")
    
    date_chunks = split_date_range(START_DATE, END_DATE, chunk_days=30)
    print(f"✓ Split into {len(date_chunks)} chunks")
    
    success_count = 0
    for i, (chunk_start, chunk_end) in enumerate(date_chunks):
        print(f"\n{'='*40}")
        print(f"PROCESSING CHUNK {i+1}/{len(date_chunks)}")
        print(f"Date range: {chunk_start} to {chunk_end}")
        print(f"{'='*40}")
        
        if DOWNLOAD_MODE == 'composite':
            success = download_composite_image(aoi, chunk_start, chunk_end, OUTPUT_DIR, SCALE)
        else:
            success = download_sentinel1_images_direct(aoi, chunk_start, chunk_end, OUTPUT_DIR, SCALE, MAX_IMAGES_PER_CHUNK)
        
        if success:
            success_count += 1
        
        time.sleep(5)
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"Successfully processed {success_count}/{len(date_chunks)} chunks")
    print(f"Data saved to: {OUTPUT_DIR}")
    
    if os.path.exists(OUTPUT_DIR):
        total_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.tif', '.tiff'))])
        print(f"Total GeoTIFF files downloaded: {total_files}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()