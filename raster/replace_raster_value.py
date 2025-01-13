from osgeo import gdal, gdalnumeric
import numpy as np
import os
from tqdm import tqdm

def replace_raster_value(input_image_path, find_value, replace_value, output_raster_path):
    # Open the input raster
    src_ds = gdal.Open(input_image_path, gdal.GA_ReadOnly)
    if src_ds is None:
        raise IOError(f"Cannot open input raster: {input_image_path}")

    # Get raster metadata
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    bands = src_ds.RasterCount
    geotransform = src_ds.GetGeoTransform()
    projection = src_ds.GetProjection()
    metadata = src_ds.GetMetadata()

    # Determine the input raster data type
    input_dtype = src_ds.GetRasterBand(1).DataType

    # Create the output raster with the same properties
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_raster_path, cols, rows, bands, input_dtype)
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_ds.SetMetadata(metadata)  # Copy metadata

    for i in range(1, bands + 1):
        # Read the band data into a NumPy array
        band = src_ds.GetRasterBand(i)
        data = band.ReadAsArray()

        # Replace the specified value
        data = np.where(data == find_value, replace_value, data)

        # Write the modified data to the output raster
        out_band = out_ds.GetRasterBand(i)
        out_band.WriteArray(data)
        out_band.SetNoDataValue(band.GetNoDataValue())  # Preserve NoData value
        out_band.SetDescription(band.GetDescription())  # Preserve band description
        out_band.SetMetadata(band.GetMetadata())  # Preserve band metadata
        out_band.FlushCache()

    # Close datasets
    src_ds = None
    out_ds = None

    print(f"Raster saved to {output_raster_path}")


    # Close datasets
    src_ds = None
    out_ds = None

    print(f"Raster saved to {output_raster_path}")

# Example usage
# input_image_path = '/Users/kanoalindiwe/Downloads/temp/Merged_GlobalHistMatch_goodWholeMea3n.tif'
output_raster_path = '/Users/kanoalindiwe/Downloads/temp/Merged_GlobalHistMatch_goodWholeMea3nreplaced.tif'
find_value = -9999  # Value to replace
replace_value = 0  # New value

# replace_raster_value(input_image_path, find_value, replace_value, output_raster_path)
gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
input_image_path_array =[
    '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif3.tif',
    '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif4.tif',
    '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif5.tif',
    '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif15.tif',
    '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif16.tif',
]

for path in tqdm(input_image_path_array, desc="Processing images"):
    replace_raster_value(path, find_value, replace_value, os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + "_replaced" + os.path.splitext(path)[1]))
