import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np


def clip_image_by_binary_mask(input_image_path, output_image_path, mask_path, nodata=None):
    """
Clip an input (multi-band) raster using a binary mask.
Where the mask is 1, retain the input raster's pixel value.
Where the mask is 0, set the pixel to NoData.

Steps:
1. Read the mask metadata (extent, resolution, transform, etc.).
2. Reproject the input raster to match the mask's grid (if needed).
3. Apply the mask to each band.
4. Write out all bands with updated metadata.

:param input_image_path: Path to the input raster to be clipped.
:param output_image_path: Path to save the output clipped raster.
:param mask_path: Path to the binary mask raster (1=keep, 0=NoData).
:param nodata: (Optional) NoData value to assign in the output.
If None, uses the input raster's NoData value (if available).
    """
    # 1. Read the mask
    with rasterio.open(mask_path) as mask_src:
        mask_data = mask_src.read(1)  # Single-band mask
        mask_meta = mask_src.meta.copy()
        mask_transform = mask_src.transform
        mask_crs = mask_src.crs
        # Dimensions of the mask
        mask_height = mask_src.height
        mask_width = mask_src.width

    # 2. Read the input raster
    with rasterio.open(input_image_path) as src:
        input_meta = src.meta.copy()
        input_crs = src.crs
        input_transform = src.transform
        input_count = src.count  # Number of bands
        input_dtype = src.dtypes[0]  # Assume all bands have same dtype
        input_nodata = src.nodata if nodata is None else nodata

        # Prepare an array to hold the reprojected data
        # Shape: (bands, height, width)
        reprojected_data = np.zeros((input_count, mask_height, mask_width), dtype=input_dtype)

        # Reproject each band to match the mask’s grid
        for band_idx in range(1, input_count + 1):
            band_data = src.read(band_idx)

            reproject(
                source=band_data,
                destination=reprojected_data[band_idx - 1],
                src_transform=input_transform,
                src_crs=input_crs,
                dst_transform=mask_transform,
                dst_crs=mask_crs,
                resampling=Resampling.nearest
            )

    # 3. Apply the binary mask to each band
    # Create an empty array for the clipped data
    clipped_data = np.zeros_like(reprojected_data, dtype=input_dtype)

    for band_idx in range(input_count):
        # Where mask==1, keep the reprojected value; where mask==0, set NoData
        clipped_data[band_idx] = np.where(mask_data == 1,
                                          reprojected_data[band_idx],
                                          input_nodata)

    # 4. Update metadata for the output
    #    - Make sure the height, width, transform, etc. match the mask
    out_meta = input_meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mask_height,
        "width": mask_width,
        "count": input_count,
        "dtype": str(input_dtype),
        "crs": mask_crs,
        "transform": mask_transform,
        "nodata": input_nodata,
    })

    # 5. Write the output raster with all bands
    with rasterio.open(output_image_path, "w", **out_meta) as dst:
        dst.write(clipped_data)

    print(f"Clipped (masked) raster saved at: {output_image_path}")


if __name__ == "__main__":
    # File paths
    mask_path = "/Users/kanoalindiwe/Downloads/temp/testDroneImages/overlap_mask.tif"
    input_raster_path = "/Users/kanoalindiwe/Downloads/temp/testDroneImages/DJI2_Geo_Nodata_GlobalMatch.tif"
    output_raster_path = "/Users/kanoalindiwe/Downloads/temp/testDroneImages/DJI2_masked.tif"

    # Call the function
    clip_image_by_binary_mask(
        input_image_path=input_raster_path,
        output_image_path=output_raster_path,
        mask_path=mask_path
    )