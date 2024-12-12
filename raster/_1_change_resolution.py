from osgeo import gdal
import os

def change_raster_resolution(input_path, output_path, resolution):
    """
    Changes the resolution of a raster while preserving the NoData value and
    resampling using the highest value (max pooling).

    Parameters:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the output raster file.
        resolution (float): Desired resolution (pixel size in the raster's units).
    """
    # Open the input raster
    input_raster = gdal.Open(input_path)
    if not input_raster:
        raise FileNotFoundError(f"Input raster not found: {input_path}")

    # Get the input raster's geotransform and spatial reference
    geotransform = list(input_raster.GetGeoTransform())
    spatial_reference = input_raster.GetProjection()
    band = input_raster.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()

    # Calculate the new raster dimensions
    original_x_res = geotransform[1]
    original_y_res = geotransform[5]
    new_x_size = int((input_raster.RasterXSize * original_x_res) / resolution)
    new_y_size = int((input_raster.RasterYSize * abs(original_y_res)) / resolution)

    # Update the geotransform with the new resolution
    geotransform[1] = resolution  # Pixel width
    geotransform[5] = -resolution  # Pixel height (negative for North-up)

    # Create the output raster
    driver = gdal.GetDriverByName("GTiff")
    output_raster = driver.Create(output_path, new_x_size, new_y_size, input_raster.RasterCount, gdal.GDT_Float32)
    if not output_raster:
        raise RuntimeError(f"Failed to create output raster: {output_path}")

    # Set geotransform and spatial reference for the output raster
    output_raster.SetGeoTransform(geotransform)
    output_raster.SetProjection(spatial_reference)

    # Preserve NoData value for all bands
    for i in range(1, input_raster.RasterCount + 1):
        input_band = input_raster.GetRasterBand(i)
        output_band = output_raster.GetRasterBand(i)
        output_band.SetNoDataValue(input_band.GetNoDataValue())

    # Resample the input raster using the highest value (max pooling)
    gdal.ReprojectImage(
        input_raster, output_raster,
        spatial_reference, spatial_reference,
        gdal.GRA_Max  # Use maximum value resampling
    )

    # Close the datasets
    input_raster = None
    output_raster = None

    print(f"Resolution changed to: {output_path}")

if __name__ == "__main__":
    # Example usage
    input_raster_path = "/Users/kanoalindiwe/Downloads/temp/smallrast.tif"
    output_raster_path = "/Users/kanoalindiwe/Downloads/temp/resolution1.tif"
    resolution = 1

    change_raster_resolution(input_raster_path, output_raster_path, resolution)