import os
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling

def merge_rasters_with_rasterio(input_directory_path, output_name):
    """
Merges all rasters in the input directory using rasterio and applies nodata values.

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

    # Open all rasters
    datasets = [rasterio.open(raster) for raster in raster_files]

    # Perform the merge
    mosaic, transform = merge(datasets)

    # Use metadata from the first file for consistency
    metadata = datasets[0].meta.copy()
    metadata.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": transform,
        "nodata": datasets[0].nodata,
        "dtype": str(mosaic.dtype),
    })

    # Output raster path
    output_path = os.path.join(input_directory_path, output_name)

    # Write the merged raster to disk
    with rasterio.open(output_path, "w", **metadata) as dest:
        dest.write(mosaic)

    # Close all datasets
    for dataset in datasets:
        dataset.close()

    return output_path

# Usage
input_directory_path = "/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/merge"
output_name = "merge.tif"
output_path = merge_rasters_with_rasterio(input_directory_path, output_name)
print(f"Merged raster saved at: {output_path}")