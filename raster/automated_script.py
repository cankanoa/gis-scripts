import os
from pathlib import Path
from _1_change_resolution import change_raster_resolution
from _2_buffer_elevation import buffer_elevation
from _3_smooth_elevation import smooth_elevation

# Define input and output paths
input_raster_path = "/Users/kanoalindiwe/Downloads/temp/smallrast.tif"
output_folder_path = "/Users/kanoalindiwe/Downloads/temp"

# Define output filenames
output_resolution_name = "Res.tif"
resolution = 1

output_buffer_name = "Buf.tif"
buffer_size_meters = 2

output_smooth_name = "Smth.tif"
smooth_size_meters = 2

#------------------------------------------------------------- Script  -------------------------------------------------------------

# Create output directory if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Step 1: Change resolution
resolution_output = os.path.join(output_folder_path, f"{Path(input_raster_path).stem}_{output_resolution_name}")
change_raster_resolution(input_raster_path, resolution_output, resolution)

# Step 2: Buffer elevation
buffer_output = os.path.join(output_folder_path, f"{Path(resolution_output).stem}_{output_buffer_name}")
buffer_elevation(resolution_output, buffer_output, buffer_size_meters)

# Step 3: Smooth elevation
smooth_output = os.path.join(output_folder_path, f"{Path(buffer_output).stem}_{output_smooth_name}")
smooth_elevation(buffer_output, smooth_output, smooth_size_meters)
