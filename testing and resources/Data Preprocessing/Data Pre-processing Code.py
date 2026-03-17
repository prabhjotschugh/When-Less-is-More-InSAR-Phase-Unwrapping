#!/usr/bin/env python3
"""
Preprocess all OPERA and Sentinel-1 data for model training, handling missing georeferencing
"""

import xarray as xr
import rasterio
import numpy as np
from pathlib import Path
import re
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.RasterioDeprecationWarning)

def extract_dates(filename, is_opera=True):
    """Extract dates from OPERA or Sentinel-1 filenames"""
    if is_opera:
        patterns = [
            r'(\d{8}T\d{6}Z)_(\d{8}T\d{6}Z)',  # Standard OPERA pattern
            r'(\d{8}).*?(\d{8})',               # Just dates
        ]
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.groups()
    else:
        patterns = [
            r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD format
            r'(\d{8})',                           # YYYYMMDD
        ]
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            if matches:
                return matches[0] if len(matches) == 1 else matches
    return None

def get_georeferencing(ds, disp):
    """Extract georeferencing from NetCDF dataset"""
    # Check for transform in attributes
    transform = ds.attrs.get('transform') or disp.attrs.get('transform')
    if transform:
        return Affine(*transform[:6]), ds.attrs.get('crs', 'EPSG:32613')

    # Check for x/y coordinate grids
    if 'x' in ds.coords and 'y' in ds.coords:
        x = ds['x'].values
        y = ds['y'].values
        # Assume regular grid, estimate transform
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        transform = Affine(dx, 0, x[0], 0, dy, y[0])
        return transform, ds.attrs.get('crs', 'EPSG:32613')

    # Default bounds for study area (adjust based on your data's approximate region)
    print(f"Warning: No georeferencing found for {ds.encoding.get('source', 'unknown')}. Using default bounds.")
    default_bounds = (-112, 39, -108, 43)  # Example: Western US, adjust as needed
    transform = Affine(0.001, 0, default_bounds[0], 0, -0.001, default_bounds[3])
    return transform, 'EPSG:32613'

def process_opera_file(opera_file, output_dir):
    """Process OPERA NetCDF file"""
    try:
        # Load dataset
        ds = xr.open_dataset(opera_file)
        
        # Find displacement variable
        disp_var = None
        possible_names = ['displacement', 'phase', 'unwrapped_phase', 'los_displacement', 'range_change']
        for name in possible_names:
            if name in ds:
                disp_var = name
                break
        
        if not disp_var:
            print(f"No displacement variable in {opera_file.name}")
            return None
        
        disp = ds[disp_var]
        
        # Get georeferencing
        src_transform, src_crs = get_georeferencing(ds, disp)
        
        # Reproject to WGS84 (EPSG:4326)
        dest_crs = 'EPSG:4326'
        try:
            transform, width, height = calculate_default_transform(
                rasterio.crs.CRS.from_string(src_crs),
                rasterio.crs.CRS.from_string(dest_crs),
                disp.shape[-1], disp.shape[-2],
                *disp.rio.bounds() if hasattr(disp, 'rio') else (src_transform.c, src_transform.f + src_transform.e * disp.shape[-2], src_transform.c + src_transform.a * disp.shape[-1], src_transform.f)
            )
        except Exception as e:
            print(f"Georeferencing error for {opera_file.name}: {e}. Using default bounds.")
            default_bounds = (-112, 39, -108, 43)  # Adjust as needed
            transform = Affine(0.001, 0, default_bounds[0], 0, -0.001, default_bounds[3])
            width, height = disp.shape[-1], disp.shape[-2]
        
        dest_data = np.zeros((height, width), dtype=disp.dtype)
        reproject(
            source=disp.values,
            destination=dest_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dest_crs,
            resampling=Resampling.bilinear
        )
        
        # Create 1D coordinates
        x = np.linspace(transform.c, transform.c + transform.a * width, width)
        y = np.linspace(transform.f, transform.f + transform.e * height, height)
        
        # Create DataArray with 1D coords
        disp_1d = xr.DataArray(
            data=dest_data,
            dims=('lat', 'lon'),
            coords={'lat': y, 'lon': x},
            name='displacement'
        )
        
        # Create Dataset
        ds_1d = xr.Dataset({'displacement': disp_1d})
        ds_1d.attrs['crs'] = dest_crs
        
        # Extract dates
        dates = extract_dates(opera_file.name, is_opera=True)
        if dates:
            ds_1d.attrs['start_date'] = dates[0]
            ds_1d.attrs['end_date'] = dates[1]
        
        # Save to NetCDF
        output_file = output_dir / f"processed_opera_{opera_file.name}"
        ds_1d.to_netcdf(output_file)
        print(f"Saved processed OPERA file: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error processing OPERA file {opera_file.name}: {e}")
        return None

def process_sentinel_file(sentinel_file, output_dir):
    """Process Sentinel-1 GeoTIFF file"""
    try:
        # Load GeoTIFF
        with rasterio.open(sentinel_file) as src:
            data = src.read(masked=True)
            transform = src.transform
            width, height = src.width, src.height
            crs = src.crs if src.crs else 'EPSG:4326'  # Assume WGS84 if no CRS
            
            # Create coordinates
            x = np.array([transform.c * i + transform.e for i in range(width)])
            y = np.array([transform.f + transform.b * j for j in range(height)])
            
            # Create DataArray
            if data.shape[0] == 1:
                dims = ('lat', 'lon')
                coords = {'lat': y, 'lon': x}
            else:
                dims = ('band', 'lat', 'lon')
                coords = {'band': np.arange(1, data.shape[0] + 1), 'lat': y, 'lon': x}
            
            data_s1 = xr.DataArray(
                data, coords=coords, dims=dims, name='backscatter'
            )
            
            # Reproject to WGS84 if needed
            if crs != 'EPSG:4326':
                dest_transform, dest_width, dest_height = calculate_default_transform(
                    rasterio.crs.CRS.from_string(crs), rasterio.crs.CRS.from_string('EPSG:4326'),
                    width, height, *src.bounds
                )
                dest_data = np.zeros((data.shape[0], dest_height, dest_width), dtype=data.dtype)
                for band in range(data.shape[0]):
                    reproject(
                        source=data[band],
                        destination=dest_data[band],
                        src_transform=transform,
                        src_crs=crs,
                        dst_transform=dest_transform,
                        dst_crs='EPSG:4326',
                        resampling=Resampling.bilinear
                    )
                x = np.linspace(dest_transform.c, dest_transform.c + dest_transform.a * dest_width, dest_width)
                y = np.linspace(dest_transform.f, dest_transform.f + dest_transform.e * dest_height, dest_height)
                data_s1 = xr.DataArray(
                    dest_data, coords={'band': np.arange(1, data.shape[0] + 1), 'lat': y, 'lon': x},
                    dims=('band', 'lat', 'lon'), name='backscatter'
                )
            
            # Create Dataset
            ds_s1 = xr.Dataset({'backscatter': data_s1})
            ds_s1.attrs['crs'] = 'EPSG:4326'
            
            # Extract dates
            dates = extract_dates(sentinel_file.name, is_opera=False)
            if dates:
                ds_s1.attrs['start_date'] = dates[0]
                ds_s1.attrs['end_date'] = dates[1] if len(dates) > 1 else dates[0]
            
            # Save to NetCDF
            output_file = output_dir / f"processed_sentinel_{sentinel_file.name.replace('.tif', '.nc')}"
            ds_s1.to_netcdf(output_file)
            print(f"Saved processed Sentinel file: {output_file}")
            return output_file
    except Exception as e:
        print(f"Error processing Sentinel file {sentinel_file.name}: {e}")
        return None

def main():
    """Preprocess all OPERA and Sentinel-1 files"""
    print("Starting Data Preprocessing")
    print("=" * 40)
    
    # Define directories
    opera_dir = Path("/scratch/10778/prabhjotschugh/insar_data")
    sentinel_dir = Path("/scratch/10778/prabhjotschugh/sentinel1_data/data")
    output_dir = Path("/scratch/10778/prabhjotschugh/processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Get file lists
    opera_files = list(opera_dir.glob("*.nc"))
    sentinel_files = list(sentinel_dir.glob("*.tif"))
    
    print(f"Found {len(opera_files)} OPERA files")
    print(f"Found {len(sentinel_files)} Sentinel-1 files")
    
    # Process OPERA files
    for opera_file in opera_files:
        process_opera_file(opera_file, output_dir)
    
    # Process Sentinel-1 files
    for sentinel_file in sentinel_files:
        process_sentinel_file(sentinel_file, output_dir)
    
    print("\n" + "=" * 40)
    print("Preprocessing completed. Output saved in:", output_dir)

if __name__ == "__main__":
    main()