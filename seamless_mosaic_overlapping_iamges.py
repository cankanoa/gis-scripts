import sys

import numpy as np
from osgeo import gdal
from scipy.optimize import least_squares
import os
np.set_printoptions(
    suppress=True,
    precision=3,
    linewidth=300,
    formatter={'float_kind':lambda x: f"{x: .3f}"}
)

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
        return None

    col_min1 = int((x_min_overlap - geo1[0]) / geo1[1])
    col_max1 = int((x_max_overlap - geo1[0]) / geo1[1])
    row_min1 = int((y_max_overlap - geo1[3]) / geo1[5])
    row_max1 = int((y_min_overlap - geo1[3]) / geo1[5])

    col_min2 = int((x_min_overlap - geo2[0]) / geo2[1])
    col_max2 = int((x_max_overlap - geo2[0]) / geo2[1])
    row_min2 = int((y_max_overlap - geo2[3]) / geo2[5])
    row_max2 = int((y_min_overlap - geo2[3]) / geo2[5])

    if row_min1 >= row_max1 or col_min1 >= col_max1 or row_min2 >= row_max2 or col_min2 >= col_max2:
        return None

    overlap_rows = min(row_max1 - row_min1, row_max2 - row_min2)
    overlap_cols = min(col_max1 - col_min1, col_max2 - col_min2)

    row_max1 = row_min1 + overlap_rows
    row_max2 = row_min2 + overlap_rows
    col_max1 = col_min1 + overlap_cols
    col_max2 = col_min2 + overlap_cols

    return ((row_min1, row_max1, col_min1, col_max1),
        (row_min2, row_max2, col_min2, col_max2))

def calculate_overlap_stats(data1, mask1, data2, mask2, overlap_coords):
    (r1_min, r1_max, c1_min, c1_max), (r2_min, r2_max, c2_min, c2_max) = overlap_coords

    valid_mask1 = mask1[r1_min:r1_max, c1_min:c1_max]
    valid_mask2 = mask2[r2_min:r2_max, c2_min:c2_max]
    overlap_mask = valid_mask1 & valid_mask2

    overlap_data1 = data1[r1_min:r1_max, c1_min:c1_max][overlap_mask]
    overlap_data2 = data2[r2_min:r2_max, c2_min:c2_max][overlap_mask]

    mean1 = np.mean(overlap_data1) if overlap_data1.size > 0 else 0
    std1 = np.std(overlap_data1) if overlap_data1.size > 0 else 0
    mean2 = np.mean(overlap_data2) if overlap_data2.size > 0 else 0
    std2 = np.std(overlap_data2) if overlap_data2.size > 0 else 0

    print('\t', f'size: {overlap_mask.size}px, mean:{mean1:.2f} vs {mean2:.2f}, std:{std1:.2f} vs {std2:.2f}')
    return mean1, std1, mean2, std2, overlap_mask.size

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
            out_band.SetNoDataValue(nodata_values[i])

    out_ds.FlushCache()

def merge_rasters(input_array, output_image_folder, output_file_name="merge.tif"):

    output_path = os.path.join(output_image_folder, output_file_name)
    input_datasets = [gdal.Open(path) for path in input_array if gdal.Open(path)]
    gdal.Warp(
        output_path,
        input_datasets,
        format='GTiff',
    )

    print(f"Merged raster saved to: {output_path}")

def process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename, custom_mean_factor, custom_std_factor, num_bands):
# ---------------------------------------- Get Images
    print('-------------------- Opening images')
    datasets = [read_raster(image_path) for image_path in input_image_paths_array]
    # num_bands = len(datasets[0][0])
    num_bands = num_bands
    num_images = len(input_image_paths_array)
    overlap_pairs = []

    band_data_list = [data for data, _, _, _, _ in datasets]
    band_masks_list = [masks for _, _, _, masks, _ in datasets]
    geo_transforms = [geo for _, geo, _, _, _ in datasets]
    projections = [proj for _, _, proj, _, _ in datasets]
    nodata_values_list = [nodata for _, _, _, _, nodata in datasets]

    # Compute original means and stds per band per image
    original_means = np.zeros((num_bands, num_images), dtype=float)
    original_stds = np.zeros((num_bands, num_images), dtype=float)
# ---------------------------------------- Get statistics
    print('-------------------- Calculating statistics')
    for img_idx, (data, _, _, masks, _) in enumerate(datasets):
        for band_idx in range(num_bands):
            band_data = data[band_idx]
            band_mask = masks[band_idx]
            valid_pixels = band_data[band_mask]
            Mj = np.mean(valid_pixels)
            Vj = np.std(valid_pixels)
            original_means[band_idx, img_idx] = Mj
            original_stds[band_idx, img_idx] = Vj

    all_adjustment_params = np.zeros((num_bands, 2 * num_images, 1), dtype=float)

    for band_idx in range(num_bands):
        print(f"Processing band {band_idx+1}/{num_bands}:")
        overlap_pairs = []
        band_data = [band_data_list[i][band_idx] for i in range(num_images)]
        band_masks = [band_masks_list[i][band_idx] for i in range(num_images)]

        constraint_matrix = []
        observed_values_vector = []
        total_overlap_pixels = 0

        printed_overlap_header = False

        # Gather overlap constraints
        for i, (data1, mask1, geo1) in enumerate(zip(band_data, band_masks, geo_transforms)):
            if not printed_overlap_header:
                print('Overlaps detected:')
                printed_overlap_header = True

            for j, (data2, mask2, geo2) in enumerate(zip(band_data, band_masks, geo_transforms)):
                if i < j:  # process each pair once (i<j)
                    overlap_coords = calculate_overlap_geometry(geo1, mask1, geo2, mask2)
                    if overlap_coords:
                        overlap_pairs.append((i, j))
                        print( f"\tOverlap({i}-{j}):", end="")
                        mean_1, std_1, mean_2, std_2, overlap_size = calculate_overlap_stats(data1, mask1, data2, mask2, overlap_coords)
                        total_overlap_pixels += overlap_size

                        # mean difference: a_i * M_i + b_i - (a_j * M_j + b_j) = 0
                        # std difference: a_i * V_i - a_j * V_j = 0
                        num_params = 2 * num_images

                        # mean difference row
                        mean_row = [0]*num_params
                        mean_row[2*i] = mean_1
                        mean_row[2*i+1] = 1
                        mean_row[2*j] = -mean_2
                        mean_row[2*j+1] = -1

                        # std difference row
                        std_row = [0]*num_params
                        std_row[2*i] = std_1
                        std_row[2*j] = -std_2

                        # Apply overlap weight (p_ij = s_ij)
                        mean_row = [val * overlap_size * custom_mean_factor for val in mean_row]
                        std_row = [val * overlap_size * custom_std_factor for val in std_row]

                        # Observed values (targets) are 0 for these constraints
                        observed_values_vector.append(0)  # mean diff
                        observed_values_vector.append(0)  # std diff

                        constraint_matrix.append(mean_row)
                        constraint_matrix.append(std_row)

        # Compute p_jj
        if total_overlap_pixels == 0:
            pjj = 1.0
        else:
            pjj = total_overlap_pixels / (2.0 * num_images)

        # Add mean and std constraints for each image to keep it close to original stats
        # mean constraint: a_j*M_j + b_j = M_j  => a_j*M_j + b_j - M_j = 0
        # std constraint: a_j*V_j = V_j        => a_j*V_j - V_j = 0

        for img_idx in range(num_images):
            Mj = original_means[band_idx, img_idx]
            Vj = original_stds[band_idx, img_idx]

            # mean constraint row
            mean_row = [0]*(2*num_images)
            mean_row[2*img_idx] = Mj
            mean_row[2*img_idx+1] = 1.0
            # we want: a_j*M_j + b_j - M_j = 0 => observed = M_j
            mean_obs = Mj

            # std constraint row
            std_row = [0]*(2*num_images)
            std_row[2*img_idx] = Vj
            # we want: a_j*V_j - V_j = 0 => observed = V_j
            std_obs = Vj

            # Weight these rows by p_jj
            mean_row = [val * pjj for val in mean_row]
            std_row = [val * pjj for val in std_row]

            mean_obs *= pjj
            std_obs *= pjj

            constraint_matrix.append(mean_row)
            observed_values_vector = np.append(observed_values_vector, mean_obs)

            constraint_matrix.append(std_row)
            observed_values_vector = np.append(observed_values_vector, std_obs)

# ---------------------------------------- Model building
        if len(constraint_matrix) > 0:
            constraint_matrix = np.array(constraint_matrix)
            observed_values_vector = np.array(observed_values_vector)

            def residuals(params):
                return constraint_matrix @ params - observed_values_vector

            initial_params = [1.0, 0.0] * num_images
            result = least_squares(residuals, initial_params)
            adjustment_params = result.x.reshape((2 * num_images, 1))
        else:
            print(f"No overlaps found for band {band_idx+1}")
            adjustment_params = np.tile([1.0, 0.0], (num_images, 1))

        all_adjustment_params[band_idx] = adjustment_params

# ---------------------------------------- Print info
        print(f"Shape: constraint_matrix: {constraint_matrix.shape}, adjustment_params: {adjustment_params.shape}, observed_values_vector: {observed_values_vector.shape}")
        print("constraint_matrix with labels:")
        # np.savetxt(sys.stdout, constraint_matrix, fmt="%16.3f")

        row_labels = []
        overlap_count = len(overlap_pairs)  # You must have recorded overlaps somewhere

        # Add two labels per overlap pair
        for (i, j) in overlap_pairs:
            row_labels.append(f"Overlap({i}-{j}) Mean Diff")
            row_labels.append(f"Overlap({i}-{j}) Std Diff")

        # Then add two labels per image for mean/std constraints
        for img_idx in range(num_images):
            row_labels.append(f"Image {img_idx} Mean Cnstr")
            row_labels.append(f"Image {img_idx} Std Cnstr")

        # Now row_labels should have exactly constraint_matrix.shape[0] elements

        # Print column labels as before
        num_params = 2 * num_images
        col_labels = []
        for i in range(num_images):
            col_labels.append(f"a{i}")
            col_labels.append(f"b{i}")

        header = " " * 24  # extra space for row label
        for lbl in col_labels:
            header += f"{lbl:>18}"
        print(header)

        # Print each row with its label
        for row_label, row in zip(row_labels, constraint_matrix):
            line = f"{row_label:>24}"  # adjust the width as needed
            for val in row:
                line += f"{val:18.3f}"
            print(line)

        print('adjustment_params:')
        np.savetxt(sys.stdout, adjustment_params, fmt="%18.3f")
        print('observed_values_vector:')
        np.savetxt(sys.stdout, observed_values_vector, fmt="%18.3f")
        # ---------------------------------------- End print info

# ---------------------------------------- Apply adjustments
    print('-------------------- Apply adjustments and saving results')
    output_path_array = []
    for img_idx in range(num_images):
        adjusted_bands = []
        for band_idx in range(num_bands):
            a = all_adjustment_params[band_idx, 2 * img_idx, 0]
            b = all_adjustment_params[band_idx, 2 * img_idx + 1, 0]
            data = band_data_list[img_idx][band_idx]
            mask = band_masks_list[img_idx][band_idx]
            adjusted_band = np.where(mask, a * data + b, data)
            adjusted_bands.append(adjusted_band)

        adjusted_bands_array = np.stack(adjusted_bands, axis=0)
        input_filename = os.path.basename(input_image_paths_array[img_idx])
        output_filename = os.path.splitext(input_filename)[0] + output_global_basename + ".tif"
        output_path = os.path.join(output_image_folder, output_filename)
        output_path_array.append(output_path)

        save_multiband_as_geotiff(
            adjusted_bands_array,
            geo_transforms[img_idx],
            projections[img_idx],
            output_path,
            nodata_values_list[img_idx]
        )
        print(f"Saved file {img_idx} to: {output_path}")
# ---------------------------------------- Merge rasters
    print('-------------------- Merging rasters and saving result')
    merge_rasters(output_path_array, output_image_folder, output_file_name=f"Merged{output_global_basename}.tif")


# ---------------------------------------- Call function
input_image_paths_array = [
    # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar.tif",
    # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar.tif",
    # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar.tif",
    # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar.tif',
    # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar.tif',
    # '/Users/kanoalindiwe/Downloads/temp/3subset.tif',
    # '/Users/kanoalindiwe/Downloads/temp/4subset.tif',
    '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu3.tif',
    '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu4.tif',
    '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu5.tif',
    '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu15.tif',
    '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu16.tif',

]
output_image_folder = "/Users/kanoalindiwe/Downloads/temp/"
output_global_basename = "_GlobalHistMatch_ClippedToPuu"
custom_mean_factor = 1 # Defualt 1
custom_std_factor = 1 # Defualt 1
process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename, custom_mean_factor, custom_std_factor, num_bands=8)





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