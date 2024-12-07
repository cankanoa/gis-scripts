import os
import numpy as np
from osgeo import gdal

def buffer_elevation(input_path, output_path, buffer_size_meters):
    # Open the input DEM
    dataset = gdal.Open(input_path)
    if not dataset:
        raise FileNotFoundError(f"Could not open {input_path}")

    # Get the spatial resolution (pixel size) from GeoTransform
    geotransform = dataset.GetGeoTransform()
    pixel_size_x = abs(geotransform[1])  # Pixel size in X direction
    pixel_size_y = abs(geotransform[5])  # Pixel size in Y direction
    pixel_size = max(pixel_size_x, pixel_size_y)  # Use the larger of the two dimensions for safety

    # Calculate buffer size in pixels
    buffer_pixel_size = int(buffer_size_meters / pixel_size)
    if buffer_pixel_size <= 0:
        raise ValueError(f"Buffer size in meters ({buffer_size_meters}) is too small for the raster resolution ({pixel_size} meters per pixel).")

    # Read DEM data as a NumPy array
    dem_band = dataset.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray()
    no_data_value = dem_band.GetNoDataValue()

    # Replace no-data values with NaN for processing
    if no_data_value is not None:
        dem_data = np.where(dem_data == no_data_value, np.nan, dem_data)

    # Define kernel size for the moving window
    kernel_size = 2 * buffer_pixel_size + 1

    # Pad the DEM to handle edges during convolution
    padded_dem = np.pad(dem_data, buffer_pixel_size, mode='constant', constant_values=np.nan)

    # Create an output array
    buffered_dem = np.full_like(dem_data, np.nan)

    # Apply the moving window
    for i in range(buffer_pixel_size, padded_dem.shape[0] - buffer_pixel_size):
        for j in range(buffer_pixel_size, padded_dem.shape[1] - buffer_pixel_size):
            # Extract the window
            window = padded_dem[i - buffer_pixel_size:i + buffer_pixel_size + 1,
                                 j - buffer_pixel_size:j + buffer_pixel_size + 1]
            # Compute the maximum value in the window
            max_value = np.nanmax(window)

            # Calculate the buffered value
            buffered_value = max_value - padded_dem[i, j] + padded_dem[i, j]
            buffered_dem[i - buffer_pixel_size, j - buffer_pixel_size] = buffered_value

    # Replace NaN back to no-data value if necessary
    if no_data_value is not None:
        buffered_dem = np.where(np.isnan(buffered_dem), no_data_value, buffered_dem)

    # Write the buffered DEM to the output path
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(
        output_path,
        dataset.RasterXSize,
        dataset.RasterYSize,
        1,
        gdal.GDT_Float32
    )
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(buffered_dem)
    if no_data_value is not None:
        out_band.SetNoDataValue(no_data_value)
    out_band.FlushCache()
    out_dataset = None
    dataset = None
    print(f"Buffered DEM saved to {output_path}")


buffer_elevation(
    input_path="/Users/kanoalindiwe/Downloads/temp/smallrast.tif",
    output_path="/Users/kanoalindiwe/Downloads/temp/buffering2.tif",
    buffer_size_meters=1  # Example: 50 meters
)