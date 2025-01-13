import os
from osgeo import gdal
import numpy as np

def parse_condition(condition):
    """
Parse a condition string (e.g., '>4', '<10', '3') into a lambda function.
    """
    if condition.startswith(">"):
        value = float(condition[1:])
        return lambda x: x > value
    elif condition.startswith("<"):
        value = float(condition[1:])
        return lambda x: x < value
    else:
        value = float(condition)
        return lambda x: x == value

def evaluate_conditions(data, conditions):
    """
Evaluate AND logic for conditions on the data.
    """
    if isinstance(conditions, list):
        result = np.ones_like(data, dtype=bool)
        for cond in conditions:
            result &= parse_condition(cond)(data)
        return result
    else:
        return parse_condition(conditions)(data)

def count_raster_values(input_image_paths_array, count_array):
    """
Count pixel values in multiple raster images based on conditions in count_array.

Args:
input_image_paths_array (list): List of paths to raster images.
count_array (list): List of conditions (e.g., [['>4', '<10'], '3']).

Outputs:
Creates individual stats files for each raster and an overall_stats.txt file
with combined averages, sums, standard deviations, and total pixel counts.
    """
    overall_stats = {}

    for input_image_path in input_image_paths_array:
        # Create output folder
        input_folder = os.path.dirname(input_image_path)
        input_filename = os.path.basename(input_image_path)
        output_folder = os.path.join(input_folder, "stats")
        os.makedirs(output_folder, exist_ok=True)

        # Prepare output file path
        base_name = os.path.splitext(input_filename)[0]
        output_file_path = os.path.join(output_folder, f"{base_name}_stats.txt")

        # Open raster file
        raster = gdal.Open(input_image_path)
        if raster is None:
            raise FileNotFoundError(f"Unable to open raster file: {input_image_path}")

        # Initialize stats storage
        image_stats = []

        # Loop over bands
        for band_idx in range(1, raster.RasterCount + 1):
            band = raster.GetRasterBand(band_idx)
            band_data = band.ReadAsArray()
            band_results = []

            for cond in count_array:
                matched_pixels = evaluate_conditions(band_data, cond)
                count = np.sum(matched_pixels)
                std = np.std(band_data[matched_pixels])
                total_pixels = band_data.size
                band_results.append((cond, count, std, total_pixels))

            image_stats.append(band_results)

        # Write individual stats to file
        with open(output_file_path, "w") as file:
            file.write(f"Statistics for raster: {input_image_path}\n\n")
            for band_idx, band_results in enumerate(image_stats, start=1):
                file.write(f"Band {band_idx}:\n")
                for cond, count, std, total_pixels in band_results:
                    file.write(f"  Condition {cond}: {count} pixels\n")
                    file.write(f"    STD: {std:.2f}\n")
                    file.write(f"    Total Pixels: {total_pixels}\n")
                file.write("\n")

        # Update overall stats
        for band_results in image_stats:
            for cond, count, std, total_pixels in band_results:
                cond_key = str(cond)
                if cond_key not in overall_stats:
                    overall_stats[cond_key] = {"counts": [], "stds": [], "total_pixels": []}
                overall_stats[cond_key]["counts"].append(count)
                overall_stats[cond_key]["stds"].append(std)
                overall_stats[cond_key]["total_pixels"].append(total_pixels)

    # Calculate overall averages, sums, stds, and total pixel counts
    overall_file_path = os.path.join(output_folder, "overall_stats.txt")
    with open(overall_file_path, "w") as file:
        file.write("Overall Statistics:\n\n")
        for cond, stats in overall_stats.items():
            total_sum = sum(stats["counts"])
            avg_count = total_sum / len(stats["counts"])
            avg_std = np.mean(stats["stds"])
            total_pixel_count = sum(stats["total_pixels"])
            file.write(f"Condition {cond}:\n")
            file.write(f"  Total Sum: {total_sum}\n")
            file.write(f"  Average Count: {avg_count:.2f}\n")
            file.write(f"  Average STD: {avg_std:.2f}\n")
            file.write(f"  Total Pixels: {total_pixel_count}\n\n")

    print(f"Overall statistics written to: {overall_file_path}")

# Example usage
if __name__ == "__main__":
    input_image_paths = [
        "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar.tif",
        '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar.tif',
        '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar.tif',
        '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar.tif',
        '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar.tif',
    ]
    count_array = ['<0']
    count_raster_values(input_image_paths, count_array)