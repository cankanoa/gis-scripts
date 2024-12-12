#Gausian
# import rasterio
# import numpy as np
# from scipy.ndimage import gaussian_filter
#
# def smooth_elevation(input_path, output_path, buffer_size_meters):
#     # Open the DEM file
#     with rasterio.open(input_path) as src:
#         dem_data = src.read(1)  # Read the first band (DEM)
#         profile = src.profile  # Copy metadata
#         resolution = src.res[0]  # Assume square pixels; take resolution of first axis
#         nodata = src.nodata  # Get nodata value, if defined
#
#     # Create a mask for valid data
#     valid_mask = np.ones_like(dem_data, dtype=bool)
#     if nodata is not None:
#         valid_mask = dem_data != nodata  # True for valid data, False for nodata
#
#     # Convert buffer size in meters to pixels
#     buffer_size_pixels = int(buffer_size_meters / resolution)
#
#     # Create an inward buffer mask to ignore edges
#     # This ensures we exclude regions within `buffer_size_pixels` of the DEM boundary
#     inward_buffer_mask = np.zeros_like(valid_mask, dtype=bool)
#     inward_buffer_mask[
#         buffer_size_pixels:-buffer_size_pixels, buffer_size_pixels:-buffer_size_pixels
#     ] = True
#
#     # Combine inward buffer mask with valid data mask
#     processing_mask = inward_buffer_mask & valid_mask
#
#     # Apply Gaussian blur only within the valid inward buffer area
#     smoothed_dem = np.zeros_like(dem_data, dtype=np.float32)
#     gaussian_filter(
#         dem_data * processing_mask,
#         sigma=buffer_size_pixels,
#         output=smoothed_dem,
#         mode="constant",
#         cval=0.0,
#     )
#
#     # Normalize to account for mask influence
#     count_filter = np.zeros_like(dem_data, dtype=np.float32)
#     gaussian_filter(
#         processing_mask.astype(np.float32),
#         sigma=buffer_size_pixels,
#         output=count_filter,
#         mode="constant",
#         cval=0.0,
#     )
#     with np.errstate(invalid="ignore"):  # Handle divisions by zero gracefully
#         smoothed_dem = np.where(
#             count_filter > 0, smoothed_dem / count_filter, nodata if nodata is not None else 0
#         )
#
#     # Apply inward buffer mask to ensure only the core region is retained
#     smoothed_dem[~processing_mask] = nodata if nodata is not None else 0
#
#     # Update metadata for output raster
#     profile.update(dtype=rasterio.float32)
#
#     # Write the smoothed DEM to the output file
#     with rasterio.open(output_path, 'w', **profile) as dst:
#         dst.write(smoothed_dem.astype(rasterio.float32), 1)
#
# if __name__ == "__main__":
#     smooth_elevation(
#         input_path="/Users/kanoalindiwe/Downloads/temp/buffering22.tif",
#         output_path="/Users/kanoalindiwe/Downloads/temp/smooth22b.tif",
#         buffer_size_meters=2)







# Gausian only high
import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_elevation(input_path, output_path, smooth_size_meters):
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
    buffer_size_pixels = int(smooth_size_meters / resolution)

    # Create an inward buffer mask to ignore edges
    inward_buffer_mask = np.zeros_like(valid_mask, dtype=bool)
    inward_buffer_mask[
        buffer_size_pixels:-buffer_size_pixels, buffer_size_pixels:-buffer_size_pixels
    ] = True

    # Combine inward buffer mask with valid data mask
    processing_mask = inward_buffer_mask & valid_mask

    # Apply Gaussian blur only within the valid inward buffer area
    smoothed_dem = np.zeros_like(dem_data, dtype=np.float32)

    # Masked data for Gaussian filter
    masked_data = np.where(processing_mask, dem_data, 0)

    # Apply Gaussian filter
    gaussian_result = gaussian_filter(masked_data, sigma=buffer_size_pixels, mode="constant", cval=0.0)

    # Take only the maximum values between original DEM and Gaussian result
    smoothed_dem = np.where(gaussian_result > dem_data, gaussian_result, dem_data)

    # Apply inward buffer mask to ensure only the core region is retained
    smoothed_dem[~processing_mask] = nodata if nodata is not None else 0

    # Update metadata for output raster
    profile.update(dtype=rasterio.float32)

    # Write the smoothed DEM to the output file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(smoothed_dem.astype(rasterio.float32), 1)

    print(f"Smoothed to: {output_path}")

if __name__ == "__main__":
    smooth_elevation(
        input_path="/Users/kanoalindiwe/Downloads/temp/buffer2.tif",
        output_path="/Users/kanoalindiwe/Downloads/temp/smooth2.tif",
        smooth_size_meters=2
    )


#
# # Local Maximum
# import os
# import numpy as np
# from osgeo import gdal, gdalconst
#
# def smooth_elevation(input_path, output_path, buffer_size_meters):
#     # Open the input raster
#     src_ds = gdal.Open(input_path, gdalconst.GA_ReadOnly)
#     if not src_ds:
#         raise FileNotFoundError(f"Unable to open file: {input_path}")
#
#     # Get geo-transform and projection
#     geo_transform = src_ds.GetGeoTransform()
#     projection = src_ds.GetProjection()
#
#     # Calculate buffer size in pixels
#     pixel_size_x = abs(geo_transform[1])  # Pixel size in X direction
#     pixel_size_y = abs(geo_transform[5])  # Pixel size in Y direction
#     buffer_size_pixels = int(buffer_size_meters / max(pixel_size_x, pixel_size_y))
#
#     # Read the raster data as a NumPy array
#     band = src_ds.GetRasterBand(1)
#     data = band.ReadAsArray()
#
#     # Handle nodata values
#     nodata_value = band.GetNoDataValue()
#     if nodata_value is not None:
#         mask = (data == nodata_value)
#     else:
#         mask = None
#
#     # Perform maximum filter using a buffer (dilation approach)
#     from scipy.ndimage import maximum_filter
#     smoothed_data = maximum_filter(data, size=buffer_size_pixels)
#
#     # Restore nodata mask
#     if mask is not None:
#         smoothed_data[mask] = nodata_value
#
#     # Create the output raster
#     driver = gdal.GetDriverByName("GTiff")
#     dst_ds = driver.Create(output_path, src_ds.RasterXSize, src_ds.RasterYSize, 1, gdalconst.GDT_Float32)
#     if not dst_ds:
#         raise RuntimeError(f"Unable to create output file: {output_path}")
#
#     # Set geo-transform and projection for output raster
#     dst_ds.SetGeoTransform(geo_transform)
#     dst_ds.SetProjection(projection)
#
#     # Write smoothed data to output raster
#     out_band = dst_ds.GetRasterBand(1)
#     out_band.WriteArray(smoothed_data)
#
#     # Set nodata value for output raster
#     if nodata_value is not None:
#         out_band.SetNoDataValue(nodata_value)
#
#     # Flush and close datasets
#     out_band.FlushCache()
#     dst_ds = None
#     src_ds = None
#
#     print(f"Buffered elevation raster saved to: {output_path}")
#
# if __name__ == "__main__":
#     smooth_elevation(
#         input_path="/Users/kanoalindiwe/Downloads/temp/buffer2.tif",
#         output_path="/Users/kanoalindiwe/Downloads/temp/smoothMax.tif",
#         buffer_size_meters=2
#     )