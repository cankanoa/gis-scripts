# Convert paths to WSL format
input_directory_path = "/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/merge"
output_name = "merge.tif"

import os
from osgeo import gdal
def merge_rasters(input_directory_path, output_name):
    """
    Merges all rasters in the input directory and applies nodata values and datatype.

    Parameters:
    - input_directory_path (str): Path to the directory containing input rasters.
    - output_name (str): Name of the output merged raster.

    Returns:
    - str: Path to the output raster.
    """
    # Get all raster file paths in the input directory
    raster_files = [
        os.path.join(input_directory_path, f)
        for f in os.listdir(input_directory_path)
        if f.endswith(('.tif', '.tiff'))
    ]

    if not raster_files:
        raise FileNotFoundError("No raster files found in the specified directory.")

    # Full path for the output raster
    output_path = os.path.join(input_directory_path, output_name)

    # Open input rasters to gather metadata
    input_rasters = [gdal.Open(raster) for raster in raster_files]
    nodata_values = [raster.GetRasterBand(1).GetNoDataValue() for raster in input_rasters]
    datatypes = [raster.GetRasterBand(1).DataType for raster in input_rasters]

    # Determine the common nodata value (assuming the first as default)
    common_nodata = nodata_values[0]

    # Determine the common datatype (assuming the first as default)
    common_datatype = datatypes[0]

    # Close input rasters
    for raster in input_rasters:
        raster = None


    # Build the gdal_merge command
    merge_command = [
        "gdal_merge.py",
        "-o", output_path,
        "-n", str(common_nodata),
        "-a_nodata", str(common_nodata),
        "-ot", gdal.GetDataTypeName(common_datatype)
    ] + raster_files

    # Execute the merge command
    gdal_merge_command = " ".join(merge_command)
    os.system(gdal_merge_command)

    return output_path

merge_rasters(input_directory_path, output_name)