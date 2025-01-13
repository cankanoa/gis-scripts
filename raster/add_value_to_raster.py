import os

import rasterio
import numpy as np


def get_lowest_pixel_value(raster_path):
    """
Get the lowest pixel value in a single raster file.

Parameters:
raster_path (str): Path to the raster file.

Returns:
float: The lowest pixel value in the raster.
    """
    with rasterio.open(raster_path) as src:
        # Read the first band as a NumPy array
        data = src.read(1)
        # Replace nodata values with NaN
        nodata_value = src.nodata
        if nodata_value is not None:
            data = np.where(data == nodata_value, np.nan, data)
        # Get the minimum value, ignoring NaNs
        return np.nanmin(data)

def add_value_to_raster(input_image_path, output_image_path, value):
    """
    Opens a raster, adds 'value' only to valid pixels (excluding nodata pixels),
    and writes the result to a new raster, preserving the original nodata value.
    """
    # Open the source raster
    with rasterio.open(input_image_path) as src:
        # Read data as a masked array: nodata pixels are masked
        data = src.read(masked=True)

        # Retrieve the nodata value and data type
        nodata_value = src.nodata
        raster_dtype = src.dtypes[0]  # Data type of the first band

        # Ensure the value is cast to the same type as the raster data
        value = np.array(value, dtype=raster_dtype)

        # Apply the value addition only to valid (non-masked) pixels
        data += value  # Offset only the valid pixels

        # Prepare metadata for output
        out_meta = src.meta.copy()

        # Use the original data type in the output metadata
        out_meta.update({'nodata': nodata_value, 'dtype': raster_dtype})

        # Fill the masked areas with the original nodata value before writing
        out_data = data.filled(nodata_value)

        # Write out the modified raster
        with rasterio.open(output_image_path, 'w', **out_meta) as dst:
            dst.write(out_data)

if __name__ == "__main__":
    raster_paths = [
        "/Users/kanoalindiwe/Downloads/temp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif",
        "/Users/kanoalindiwe/Downloads/temp/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif",
        "/Users/kanoalindiwe/Downloads/temp/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar_GlobalMatch.tif",
        '/Users/kanoalindiwe/Downloads/temp/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar_GlobalMatch2.tif',
        '/Users/kanoalindiwe/Downloads/temp/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar_GlobalMatch2.tif',
    ]
    lowest_value: float = None
    for raster_path in raster_paths:
        value = get_lowest_pixel_value(raster_path)
        if lowest_value is None or value < lowest_value:
            lowest_value = value

    if lowest_value < 0:
        lowest_value = abs(lowest_value)
        for raster_path in raster_paths:
            output_image_path = os.path.join(os.path.dirname(raster_path), f"{os.path.splitext(os.path.basename(raster_path))[0]}_{int(lowest_value)}Added{os.path.splitext(raster_path)[1]}")
            add_value_to_raster(raster_path, output_image_path, lowest_value)
            print(f"Added {lowest_value} to raster and saved as: {output_image_path}")
    else:
        print('No lowest value found')