import os
from osgeo import gdal

# Convert paths to WSL format
input_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/DEM"
output_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar"
output_file = os.path.join(output_folder, "merged_output2.tif")

# Find all .tif files in the input directory and verify their existence
tif_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]
tif_files = [f for f in tif_files if os.path.exists(f)]

if not tif_files:
    raise FileNotFoundError("No .tif files found in the input directory.")

# Create a virtual raster (VRT) to handle merging
vrt_file = os.path.join(output_folder, "temp_merged.vrt")
gdal.BuildVRT(vrt_file, tif_files)

# Translate the VRT to a single output .tif file in ENVI format
gdal.Translate(output_file, vrt_file, format='ENVI')

# Clean up the temporary VRT file
os.remove(vrt_file)

print(f"Merging completed. Output saved at: {output_file}")
