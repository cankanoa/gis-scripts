import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np

def create_overlaping_area_mask(input_images_path_array, output_image_path, nodata=None):
    """
Create a binary mask (1/0) indicating where all rasters have valid data.
Resamples all rasters to match the first raster's resolution, alignment, and bounds.

:param input_images_path_array: List of file paths to input rasters.
:param output_image_path: File path to the output mask raster.
:param nodata: (Optional) Nodata value to assume for all rasters if none is stored.
    """
    if not input_images_path_array:
        raise ValueError("No input rasters provided.")

    # 1. Open the first raster - this is our 'reference' grid
    with rasterio.open(input_images_path_array[0]) as ref_src:
        ref_meta = ref_src.meta.copy()
        ref_crs = ref_src.crs
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height
        # Get nodata from the first raster if not provided
        ref_nodata = ref_src.nodata if (nodata is None) else nodata

        # Read the first band of the reference raster
        ref_data = ref_src.read(1)

    # Convert reference raster into a mask (True = valid, False = NoData)
    ref_valid_mask = (ref_data != ref_nodata)

    # 2. Initialize a cumulative "overlap" mask with the reference
    overlap_mask = ref_valid_mask.copy()

    # 3. For each other raster, reproject to the reference grid, build the mask
    for raster_path in input_images_path_array[1:]:
        with rasterio.open(raster_path) as src:
            src_nodata = src.nodata if (nodata is None) else nodata

            # Prepare an array for the reprojected data
            dst_data = np.zeros((ref_height, ref_width), dtype=src.meta['dtype'])

            # Reproject the current raster onto the reference grid
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.nearest
            )

            # Build a validity mask for this raster
            current_valid_mask = (dst_data != src_nodata)

            # Combine with the cumulative overlap
            overlap_mask &= current_valid_mask

    # 4. Convert boolean mask to 1/0
    overlap_mask_uint8 = overlap_mask.astype(np.uint8)

    # 5. Update metadata for the output
    out_meta = ref_meta.copy()
    out_meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0  # or you can remove nodata if you prefer
    })

    # 6. Write out the overlap mask
    with rasterio.open(output_image_path, 'w', **out_meta) as dst:
        dst.write(overlap_mask_uint8, 1)

    print(f"Overlap mask created at: {output_image_path}")

if __name__ == "__main__":
    input_images_path_array = [
        '/Users/kanoalindiwe/Downloads/temp/testDroneImages/DJI1_Geo_Nodata_GlobalMatch.tif',
        '/Users/kanoalindiwe/Downloads/temp/testDroneImages/DJI2_Geo_Nodata_GlobalMatch.tif',
    ]

    create_overlaping_area_mask(input_images_path_array, '/Users/kanoalindiwe/Downloads/temp/testDroneImages/overlap_mask.tif')