import pdal
import os

def add_projection_to_laz(input_file, output_file):
    pipeline = f"""
    [
        "{input_file}",
        {{
            "type": "filters.reprojection",
            "in_srs": "EPSG:6635",
            "out_srs": "EPSG:6635"
        }},
        "{output_file}"
    ]
    """
    pdal_pipeline = pdal.Pipeline(pipeline)
    pdal_pipeline.execute()

def process_all_las_files(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all .las files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".las"):
            input_file = os.path.join(input_folder, filename)

            # Construct the output file path using the same filename
            output_file = os.path.join(output_folder, filename)

            # Process the file
            add_projection_to_laz(input_file, output_file)

# Example usage with WSL-compatible paths
input_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/BigIslandLidar20182019"
output_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/BigIslandLidar20182019_2"
process_all_las_files(input_folder, output_folder)
