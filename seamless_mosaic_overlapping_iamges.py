import numpy as np
from osgeo import gdal, ogr
from scipy.optimize import least_squares
import os

# Helper to read raster data
def read_raster(image_path):
    dataset = gdal.Open(image_path)
    if not dataset:
        raise FileNotFoundError(f"Cannot open {image_path}")
    band = dataset.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    data = band.ReadAsArray()
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    mask = data != nodata_value  # Create mask for valid pixels
    return data, geo_transform, projection, mask

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
    row_min1 = int((y_min_overlap - geo1[3]) / geo1[5])
    row_max1 = int((y_max_overlap - geo1[3]) / geo1[5])

    col_min2 = int((x_min_overlap - geo2[0]) / geo2[1])
    col_max2 = int((x_max_overlap - geo2[0]) / geo2[1])
    row_min2 = int((y_min_overlap - geo2[3]) / geo2[5])
    row_max2 = int((y_max_overlap - geo2[3]) / geo2[5])

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

    return (mean1, std1, mean2, std2)

# Perform least squares adjustment for an image
def least_squares_adjustment(image_idx, image_paths, all_stats):
    def residuals(params):
        a, b = params
        res = []
        for stats in all_stats[image_idx]:
            mean1, std1, mean2, std2 = stats
            res.extend([a * mean1 + b - mean2, a * std1 - std2])
        return res

    initial_params = [1.0, 0.0]
    result = least_squares(residuals, initial_params)
    return result.x

# Adjust image based on calculated parameters
def adjust_image(data, mask, a, b):
    adjusted_data = np.copy(data)
    adjusted_data[mask] = a * data[mask] + b
    return adjusted_data

# Process images
def process_images(input_image_paths_array, output_image_path):
    datasets = [read_raster(image_path) for image_path in input_image_paths_array]
    all_stats = [[] for _ in range(len(datasets))]

    # Calculate overlaps and stats
    for i, (data1, geo1, proj1, mask1) in enumerate(datasets):
        print('---', data1, '---', geo1, '---', proj1, '---', mask1)
        for j, (data2, geo2, proj2, mask2) in enumerate(datasets):
            if i != j:
                overlap_coords = calculate_overlap_geometry(geo1, mask1, geo2, mask2)
                if overlap_coords:
                    stats = calculate_overlap_stats(data1, mask1, data2, mask2, overlap_coords)
                    all_stats[i].append(stats)

    # Perform least squares adjustment for each image
    adjustments = [least_squares_adjustment(i, input_image_paths_array, all_stats) for i in range(len(datasets))]

    # Adjust images and save
    adjusted_images = []
    for i, (data, geo, proj, mask) in enumerate(datasets):
        a, b = adjustments[i]
        adjusted_images.append(adjust_image(data, mask, a, b))

    # Combine adjusted images (mean value for overlaps)
    combined_image = np.mean(adjusted_images, axis=0)

    # Save the result
    save_as_geotiff(combined_image, geo1, proj1, output_image_path)

# Save as GeoTIFF
def save_as_geotiff(array, geo_transform, projection, path):
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = array.shape
    out_ds = driver.Create(path, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_ds.FlushCache()

# Call the main function
input_image_paths_array = ["/Users/kanoalindiwe/Downloads/resources/worldview/016445319010_01_P003_MUL/17DEC08211758-M1BS-016445319010_01_P003.TIF", "/Users/kanoalindiwe/Downloads/resources/worldview/016445319010_01_P004_MUL/17DEC08211800-M1BS-016445319010_01_P004.TIF"]
output_image_path = "/Users/kanoalindiwe/Downloads/temp/merge.tif"
process_images(input_image_paths_array, output_image_path)