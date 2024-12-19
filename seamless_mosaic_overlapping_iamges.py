import numpy as np
from osgeo import gdal, ogr
from scipy.optimize import least_squares
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Helper to read raster data
def read_raster(image_path):
    dataset = gdal.Open(image_path)
    if not dataset:
        raise FileNotFoundError(f"Cannot open {image_path}")

    num_bands = dataset.RasterCount
    data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(num_bands)]
    nodata_values = [dataset.GetRasterBand(i + 1).GetNoDataValue() for i in range(num_bands)]
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    masks = [(band != nodata) if nodata is not None else np.ones_like(band, dtype=bool)
             for band, nodata in zip(data, nodata_values)]

    return data, geo_transform, projection, masks, nodata_values

# Calculate overlapping area geometry between two rasters
def calculate_overlap_geometry(geo1, mask1, geo2, mask2):
    x_min1 = geo1[0]
    x_max1 = geo1[0] + geo1[1] * mask1.shape[1]
    y_min1 = geo1[3] + geo1[5] * mask1.shape[0]
    y_max1 = geo1[3]

    x_min2 = geo2[0]
    x_max2 = geo2[0] + geo2[1] * mask2.shape[1]
    y_min2 = geo2[3] + geo2[5] * mask2.shape[0]
    y_max2 = geo2[3]

    x_min_overlap = max(x_min1, x_min2)
    x_max_overlap = min(x_max1, x_max2)
    y_min_overlap = max(y_min1, y_min2)
    y_max_overlap = min(y_max1, y_max2)

    if x_min_overlap >= x_max_overlap or y_min_overlap >= y_max_overlap:
        return None  # No overlap

    col_min1 = int((x_min_overlap - geo1[0]) / geo1[1])
    col_max1 = int((x_max_overlap - geo1[0]) / geo1[1])
    row_min1 = int((y_max_overlap - geo1[3]) / geo1[5])  # Flip y_min/y_max for rows
    row_max1 = int((y_min_overlap - geo1[3]) / geo1[5])

    col_min2 = int((x_min_overlap - geo2[0]) / geo2[1])
    col_max2 = int((x_max_overlap - geo2[0]) / geo2[1])
    row_min2 = int((y_max_overlap - geo2[3]) / geo2[5])  # Flip y_min/y_max for rows
    row_max2 = int((y_min_overlap - geo2[3]) / geo2[5])

    if row_min1 >= row_max1 or col_min1 >= col_max1 or row_min2 >= row_max2 or col_min2 >= col_max2:
        print(f"Invalid overlap region: skipping overlap calculation")

    # Calculate the minimum overlap shape
    overlap_rows = min(row_max1 - row_min1, row_max2 - row_min2)
    overlap_cols = min(col_max1 - col_min1, col_max2 - col_min2)

    # Adjust bounds to match the minimum shape
    row_max1 = row_min1 + overlap_rows
    row_max2 = row_min2 + overlap_rows
    col_max1 = col_min1 + overlap_cols
    col_max2 = col_min2 + overlap_cols

    return (
        (row_min1, row_max1, col_min1, col_max1),
        (row_min2, row_max2, col_min2, col_max2),
    )

# Calculate mean and std for overlap
def calculate_overlap_stats(data1, mask1, data2, mask2, overlap_coords):
    (r1_min, r1_max, c1_min, c1_max), (r2_min, r2_max, c2_min, c2_max) = overlap_coords

    valid_mask1 = mask1[r1_min:r1_max, c1_min:c1_max]
    valid_mask2 = mask2[r2_min:r2_max, c2_min:c2_max]
    overlap_mask = valid_mask1 & valid_mask2

    overlap_data1 = data1[r1_min:r1_max, c1_min:c1_max][overlap_mask]
    overlap_data2 = data2[r2_min:r2_max, c2_min:c2_max][overlap_mask]

    mean1 = np.mean(overlap_data1)
    std1 = np.std(overlap_data1)
    mean2 = np.mean(overlap_data2)
    std2 = np.std(overlap_data2)

    print('\t\t\t',f'size: {len(valid_mask1)}px', 'vs', f'{len(valid_mask2)}px,', 'mean:',f'{mean1:.2f}', 'vs',f'{mean2:.2f},',f'std: {std1:.2f}', 'vs',f'{std2:.2f}')
    return (mean1, std1, mean2, std2)

# Perform least squares adjustment for an image
from scipy.optimize import least_squares
import numpy as np

def least_squares_adjustment(mean_1, std_1, mean_2, std_2):
    """
    Perform least squares adjustment for a single overlap with given stats.
    mean_1, std_1: Mean and standard deviation for image 1 in overlap.
    mean_2, std_2: Mean and standard deviation for image 2 in overlap.
    """
    def residuals(params):
        a, b = params
        return [
            a * mean_1 + b - mean_2,  # Mean constraint
            a * std_1 - std_2         # Variance constraint
        ]

    # Initial guess for parameters [a, b]
    initial_params = [1.0, 0.0]

    # Solve using least squares optimization
    result = least_squares(residuals, initial_params)

    return result.x  # Return [a, b] for the overlap


# Adjust image based on calculated parameters
def adjust_image(data, mask, a, b):
    adjusted_data = np.copy(data)
    adjusted_data[mask] = a * data[mask] + b # The adjustment math

    return adjusted_data

def calculate_mean_and_std(data):
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

# Save as GeoTIFF
def save_multiband_as_geotiff(array, geo_transform, projection, path, nodata_values):
    driver = gdal.GetDriverByName("GTiff")
    num_bands, rows, cols = array.shape
    out_ds = driver.Create(path, cols, rows, num_bands, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)

    for i in range(num_bands):
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(array[i])
        if nodata_values and nodata_values[i] is not None:
            out_band.SetNoDataValue(nodata_values[i])  # Set the NoData value for this band

    out_ds.FlushCache()

def save_adjusted_raster(adjusted_data, geo_transform, projection, input_image_path, output_image_folder, output_global_basename, nodata_value=None):
    import os

    # Extract the input file name without extension
    input_filename = os.path.basename(input_image_path)
    output_filename = os.path.splitext(input_filename)[0] + output_global_basename + ".tif"
    output_path = os.path.join(output_image_folder, output_filename)

    # Prepare single-band array for saving
    single_band_array = adjusted_data[np.newaxis, :, :]  # Add a new axis for single-band

    # Save the adjusted raster
    save_multiband_as_geotiff(single_band_array, geo_transform, projection, output_path, [nodata_value])

    print(f"Saved adjusted raster to {output_path}")

def process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename):
    datasets = [read_raster(image_path) for image_path in input_image_paths_array]
    num_bands = len(datasets[0][0])
    num_images = len(datasets)

    for band_idx in range(num_bands):
        # ---------- Per band
        print(f"Processing band {band_idx + 1}/{num_bands}:")
        band_data = [data[band_idx] for data, _, _, _, _ in datasets]
        band_masks = [masks[band_idx] for _, _, _, masks, _ in datasets]
        geo_transforms = [geo for _, geo, _, _, _ in datasets]
        projections = [proj for _, _, proj, _, _ in datasets]

        adjustment_params = [] # Will hold final a,b for each image
        constraint_matrix = [] # Will hold all constraints for this band
        observed_values_vector = [] # Will hold all observed values (L) for this band

        # Gather constraints and observed values for all overlaps in this band
        for i, (data1, mask1, geo1) in enumerate(zip(band_data, band_masks, geo_transforms)):
            print('\tOverlaps detected (self vs other):') if not globals().get('overlap_printed') and globals().update({'overlap_printed': True}) is None else None
            print('\t\t', f'Image {i}:')

            for j, (data2, mask2, geo2) in enumerate(zip(band_data, band_masks, geo_transforms)):
                if i != j:
                    overlap_coords = calculate_overlap_geometry(geo1, mask1, geo2, mask2)
                    if overlap_coords:
                        mean_1, std_1, mean_2, std_2 = calculate_overlap_stats(data1, mask1, data2, mask2, overlap_coords)

                        # Append observed values for this overlap
                        observed_values_vector.append(0)  # mean constraint target difference
                        observed_values_vector.append(0)  # std constraint target difference

                        # Create constraints for mean difference
                        num_params = 2 * num_images
                        mean_row = [0] * num_params
                        mean_row[2 * i] = mean_1
                        mean_row[2 * i + 1] = 1
                        mean_row[2 * j] = -mean_2
                        mean_row[2 * j + 1] = -1
                        constraint_matrix.append(mean_row)

                        # Create constraints for std difference
                        std_row = [0] * num_params
                        std_row[2 * i] = std_1
                        std_row[2 * j] = -std_2
                        constraint_matrix.append(std_row)

        # Now that all overlaps for this band are collected, convert lists to arrays
        if len(constraint_matrix) > 0:
            constraint_matrix = np.array(constraint_matrix)
            observed_values_vector = np.array(observed_values_vector)

            def residuals(params):
                return constraint_matrix @ params - observed_values_vector

            initial_params = [1.0, 0.0] * num_images
            result = least_squares(residuals, initial_params)

            adjustment_params = result.x.reshape(num_images, 2)

            # Apply the adjustments to each image for this band
            for k, (data, mask) in enumerate(zip(band_data, band_masks)):
                a, b = adjustment_params[k]
                adjusted_data = np.where(mask, a * data + b, data)
                save_adjusted_raster(
                    adjusted_data=adjusted_data,
                    geo_transform=geo_transforms[k],
                    projection=projections[k],
                    input_image_path=input_image_paths_array[k],
                    output_image_folder=output_image_folder,
                    output_global_basename=output_global_basename + str(band_idx),
                    nodata_value=None
                )
        else:
            # No constraints or overlaps found for this band
            print(f"No overlaps found for band {band_idx + 1}. Skipping optimization.")

        print(f"Shape: constraint_matrix: {constraint_matrix.shape if isinstance(constraint_matrix, np.ndarray) else 0}, "
              f"adjustment_params: {adjustment_params.shape if isinstance(adjustment_params, np.ndarray) else 0}, "
              f"observed_values_vector: {observed_values_vector.shape if isinstance(observed_values_vector, np.ndarray) else 0}")
        print("constraint_matrix:\n", constraint_matrix)
        print('adjustment_params:\n', adjustment_params)
        print('observed_values_vector:\n', observed_values_vector)



    #
    #     # Adjust images for this band
    #     adjusted_bands = []
    #     for i, (data, mask, geo, proj, stats_for_print) in enumerate(zip(band_data, band_masks, geo_transforms, projections, all_stats)):
    #         a, b = adjustment_params[i]
    #         print('\tWhole adjustment stats (original vs adjusted):') if not globals().get('overlap_adjust_printed') and globals().update({'overlap_adjust_printed': True}) is None else None
    #         adjusted_band = adjust_image(data, mask, a, b)
    #         adjusted_bands.append(adjusted_band)
    #         datasets[i][0][band_idx] = adjusted_band  # Update band data
    #
    #         # Prints
    #         stats_for_print
    #         print('\t\t',f'Image {i}: ')
    #         original_mean, original_std = calculate_mean_and_std(data[mask])
    #         adjusted_mean, adjusted_std = calculate_mean_and_std(adjusted_band[mask])
    #         print(f'\t\t\tsize: {len(data[mask])}px', 'vs', f'{len(adjusted_band[mask])}px,', 'mean:',f'{original_mean:.2f}', 'vs',f'{adjusted_mean:.2f},',f'std: {original_std:.2f}', 'vs',f'{adjusted_std:.2f}')
    #
    #     # Recalculate overlap stats for adjusted images
    #     print("\tOverlap adjustment stats (original vs adjusted):")
    #     for i, (data1, mask1, geo1) in enumerate(zip(adjusted_bands, band_masks, geo_transforms)):
    #         print(f'\t\tImage {i}:')
    #         for j, (data2, mask2, geo2) in enumerate(zip(adjusted_bands, band_masks, geo_transforms)):
    #             if i != j:
    #                 overlap_coords = calculate_overlap_geometry(geo1, mask1, geo2, mask2)
    #                 if overlap_coords:
    #                     calculate_overlap_stats(data1, mask1, data2, mask2, overlap_coords)
    #
    #     globals()['overlap_adjust_printed'] = None
    #     globals()['overlap_printed'] = None
    #
    # # Save the adjusted multi-band images
    # for i, (data, geo, proj, _, nodata_values) in enumerate(datasets):
    #     # Stack adjusted bands
    #     adjusted_data = np.stack(data, axis=0)
    #     input_basename = os.path.basename(input_image_paths_array[i])
    #     output_filename = f"{os.path.splitext(input_basename)[0]}{output_global_basename}.tif"
    #     output_path = os.path.join(output_image_folder, output_filename)
    #
    #     save_multiband_as_geotiff(adjusted_data, geo, proj, output_path, nodata_values)

# Call the main function
input_image_paths_array = [
    "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar.tif",
    "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar.tif",
    "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar.tif",
    '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar.tif',
    # '/Users/kanoalindiwe/Downloads/temp/3subset.tif',
    # '/Users/kanoalindiwe/Downloads/temp/4subset.tif',
]
output_image_folder = "/Users/kanoalindiwe/Downloads/temp/"
output_global_basename = "_GlobalHistMatch"
process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename)








# ----------------------- LOCAL HIST MATCH -----------------------
#
# def calculate_block_statistics(band_data, mask, block_size):
#     num_blocks_x = band_data.shape[1] // block_size
#     num_blocks_y = band_data.shape[0] // block_size
#
#     block_means = np.zeros((num_blocks_y, num_blocks_x))
#     block_stds = np.zeros((num_blocks_y, num_blocks_x))
#
#     for by in range(num_blocks_y):
#         for bx in range(num_blocks_x):
#             block = band_data[
#                 by * block_size:(by + 1) * block_size,
#                 bx * block_size:(bx + 1) * block_size
#             ]
#             block_mask = mask[
#                 by * block_size:(by + 1) * block_size,
#                 bx * block_size:(bx + 1) * block_size
#             ]
#             valid_data = block[block_mask]
#             if valid_data.size > 0:
#                 block_means[by, bx] = np.mean(valid_data)
#                 block_stds[by, bx] = np.std(valid_data)
#             else:
#                 block_means[by, bx] = 0
#                 block_stds[by, bx] = 1
#
#     return block_means, block_stds
#
# def interpolate_block_values(block_data, image_shape):
#     from scipy.ndimage import zoom
#     scale_y = image_shape[0] / block_data.shape[0]
#     scale_x = image_shape[1] / block_data.shape[1]
#     return zoom(block_data, (scale_y, scale_x), order=1)
#
# def apply_gamma_correction(band_data, mask, reference_means, block_means):
#     adjusted_band = np.copy(band_data)
#     gamma = np.log(reference_means + 1e-5) / np.log(block_means + 1e-5)
#     adjusted_band[mask] = band_data[mask] ** gamma[mask]
#     return adjusted_band
#
# def process_single_image_local(input_image_path, output_path, block_size=128):
#     # Read raster data
#     data, geo_transform, projection, masks, nodata_values = read_raster(input_image_path)
#
#     adjusted_bands = []
#     for band_idx, band_data in enumerate(data):
#         mask = masks[band_idx]
#
#         # Step 1: Calculate block statistics
#         block_means, block_stds = calculate_block_statistics(band_data, mask, block_size)
#
#         # Step 2: Interpolate block statistics
#         reference_means = interpolate_block_values(block_means, band_data.shape)
#         reference_stds = interpolate_block_values(block_stds, band_data.shape)
#
#         # Step 3: Apply local gamma correction
#         adjusted_band = apply_gamma_correction(band_data, mask, reference_means, block_means)
#         adjusted_bands.append(adjusted_band)
#
#     save_multiband_as_geotiff(
#         np.stack(adjusted_bands, axis=0),
#         geo_transform,
#         projection,
#         output_path,
#         nodata_values
#     )
#     print(f"Saved local histogram matched image to {output_path}")
#
# def process_local_histogram_matching(input_image_paths_array, output_image_folder, output_local_basename):
#     for input_image_path in input_image_paths_array:
#         # Create output file path here
#         input_basename = os.path.basename(input_image_path)
#         output_filename = f"{os.path.splitext(input_basename)[0]}{output_local_basename}.tif"
#         output_path = os.path.join(output_image_folder, output_filename)
#
#         process_single_image_local(input_image_path, output_path)
#
# output_local_basename = "_LocalHistMatch"
# process_local_histogram_matching(input_image_paths_array, output_image_folder, output_local_basename)