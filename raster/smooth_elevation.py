import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter

def buffer_elevation(input_path, output_path, buffer_size_meters):
    # Open the DEM file
    with rasterio.open(input_path) as src:
        dem_data = src.read(1)  # Read the first band (DEM)
        profile = src.profile  # Copy metadata
        resolution = src.res[0]  # Assume square pixels; take resolution of first axis
        nodata = src.nodata  # Get nodata value, if defined

    # Create a mask for valid data
    valid_mask = np.ones_like(dem_data, dtype=bool)
    if nodata is not None:
        valid_mask = dem_data != nodata  # True for valid data, False for nodata

    # Convert buffer size in meters to pixels
    buffer_size_pixels = int(buffer_size_meters / resolution)

    # Create an inward buffer mask to ignore edges
    # This ensures we exclude regions within `buffer_size_pixels` of the DEM boundary
    inward_buffer_mask = np.zeros_like(valid_mask, dtype=bool)
    inward_buffer_mask[
        buffer_size_pixels:-buffer_size_pixels, buffer_size_pixels:-buffer_size_pixels
    ] = True

    # Combine inward buffer mask with valid data mask
    processing_mask = inward_buffer_mask & valid_mask

    # Apply Gaussian blur only within the valid inward buffer area
    smoothed_dem = np.zeros_like(dem_data, dtype=np.float32)
    gaussian_filter(
        dem_data * processing_mask,
        sigma=buffer_size_pixels,
        output=smoothed_dem,
        mode="constant",
        cval=0.0,
    )

    # Normalize to account for mask influence
    count_filter = np.zeros_like(dem_data, dtype=np.float32)
    gaussian_filter(
        processing_mask.astype(np.float32),
        sigma=buffer_size_pixels,
        output=count_filter,
        mode="constant",
        cval=0.0,
    )
    with np.errstate(invalid="ignore"):  # Handle divisions by zero gracefully
        smoothed_dem = np.where(
            count_filter > 0, smoothed_dem / count_filter, nodata if nodata is not None else 0
        )

    # Apply inward buffer mask to ensure only the core region is retained
    smoothed_dem[~processing_mask] = nodata if nodata is not None else 0

    # Update metadata for output raster
    profile.update(dtype=rasterio.float32)

    # Write the smoothed DEM to the output file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(smoothed_dem.astype(rasterio.float32), 1)


buffer_elevation(
    input_path="/Users/kanoalindiwe/Downloads/temp/buffering2.tif",
    output_path="/Users/kanoalindiwe/Downloads/temp/smooth8.tif",
    buffer_size_meters=1)