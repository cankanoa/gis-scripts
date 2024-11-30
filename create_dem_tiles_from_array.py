import os
import json
import pdal
import subprocess
import geopandas as gpd
from tqdm import tqdm  # Import tqdm for progress bar

# Define input and output folders
input_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/BigIslandLidar20182019_2/"
output_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/DEM/"
geopackage_path = "/mnt/x/PROJECTS_2/Big_Island/LandCover/Tiles/BigIsland_Tiles.gpkg"
layer_name = "BigIsland_Tiles"  # Replace with the actual layer name in your GeoPackage

# Load the GeoPackage
gdf = gpd.read_file(geopackage_path, layer=layer_name)

# Array of file names
file_names = [
    "BigIslandLidar20182019_067_014.las",
    "BigIslandLidar20182019_067_015.las",
    "BigIslandLidar20182019_066_016.las",
    "BigIslandLidar20182019_067_016.las",
    "BigIslandLidar20182019_066_017.las",
    "BigIslandLidar20182019_067_017.las",
    "BigIslandLidar20182019_040_018.las",
    "BigIslandLidar20182019_041_018.las",
    "BigIslandLidar20182019_038_019.las",
    "BigIslandLidar20182019_039_019.las",
    "BigIslandLidar20182019_040_019.las",
    "BigIslandLidar20182019_041_019.las",
    "BigIslandLidar20182019_037_020.las",
    "BigIslandLidar20182019_038_020.las",
    "BigIslandLidar20182019_039_020.las",
    "BigIslandLidar20182019_040_020.las",
    "BigIslandLidar20182019_041_020.las",
    "BigIslandLidar20182019_037_021.las",
    "BigIslandLidar20182019_038_021.las",
    "BigIslandLidar20182019_039_021.las",
    "BigIslandLidar20182019_040_021.las",
    "BigIslandLidar20182019_041_021.las",
    "BigIslandLidar20182019_042_021.las",
    "BigIslandLidar20182019_043_021.las",
    "BigIslandLidar20182019_044_021.las",
    "BigIslandLidar20182019_045_021.las",
    "BigIslandLidar20182019_066_021.las",
    "BigIslandLidar20182019_067_021.las",
    "BigIslandLidar20182019_039_022.las",
    "BigIslandLidar20182019_040_022.las",
    "BigIslandLidar20182019_041_022.las",
    "BigIslandLidar20182019_042_022.las",
    "BigIslandLidar20182019_043_022.las",
    "BigIslandLidar20182019_044_022.las",
    "BigIslandLidar20182019_045_022.las",
    "BigIslandLidar20182019_063_022.las",
    "BigIslandLidar20182019_066_022.las",
    "BigIslandLidar20182019_067_022.las",
    "BigIslandLidar20182019_040_023.las",
    "BigIslandLidar20182019_041_023.las",
    "BigIslandLidar20182019_042_023.las",
    "BigIslandLidar20182019_043_023.las",
    "BigIslandLidar20182019_044_023.las",
    "BigIslandLidar20182019_045_023.las",
    "BigIslandLidar20182019_041_024.las",
    "BigIslandLidar20182019_042_024.las",
    "BigIslandLidar20182019_043_024.las",
    "BigIslandLidar20182019_044_024.las",
    "BigIslandLidar20182019_045_024.las",
    "BigIslandLidar20182019_042_025.las",
    "BigIslandLidar20182019_043_025.las",
    "BigIslandLidar20182019_044_025.las",
    "BigIslandLidar20182019_045_025.las",
    "BigIslandLidar20182019_043_026.las",
    "BigIslandLidar20182019_044_026.las",
    "BigIslandLidar20182019_045_026.las",
    "BigIslandLidar20182019_055_026.las",
    "BigIslandLidar20182019_045_027.las",
    "BigIslandLidar20182019_046_027.las",
    "BigIslandLidar20182019_047_027.las",
    "BigIslandLidar20182019_048_027.las",
    "BigIslandLidar20182019_049_027.las",
    "BigIslandLidar20182019_050_027.las",
    "BigIslandLidar20182019_051_027.las",
    "BigIslandLidar20182019_052_027.las",
    "BigIslandLidar20182019_053_027.las",
    "BigIslandLidar20182019_054_027.las",
    "BigIslandLidar20182019_055_027.las",
    "BigIslandLidar20182019_056_027.las",
    "BigIslandLidar20182019_057_027.las",
    "BigIslandLidar20182019_058_027.las",
    "BigIslandLidar20182019_059_027.las",
    "BigIslandLidar20182019_060_027.las",
    "BigIslandLidar20182019_061_027.las",
    "BigIslandLidar20182019_062_027.las",
    "BigIslandLidar20182019_063_027.las",
    "BigIslandLidar20182019_064_027.las",
    "BigIslandLidar20182019_065_027.las",
    "BigIslandLidar20182019_066_027.las",
    "BigIslandLidar20182019_067_027.las",
    "BigIslandLidar20182019_046_028.las",
    "BigIslandLidar20182019_047_028.las",
    "BigIslandLidar20182019_048_028.las",
    "BigIslandLidar20182019_049_028.las",
    "BigIslandLidar20182019_050_028.las",
    "BigIslandLidar20182019_051_028.las",
    "BigIslandLidar20182019_052_028.las",
    "BigIslandLidar20182019_053_028.las",
    "BigIslandLidar20182019_054_028.las",
    "BigIslandLidar20182019_055_028.las",
    "BigIslandLidar20182019_056_028.las",
    "BigIslandLidar20182019_057_028.las",
    "BigIslandLidar20182019_058_028.las",
    "BigIslandLidar20182019_059_028.las",
    "BigIslandLidar20182019_060_028.las",
    "BigIslandLidar20182019_061_028.las",
    "BigIslandLidar20182019_062_028.las",
    "BigIslandLidar20182019_063_028.las",
    "BigIslandLidar20182019_064_028.las",
    "BigIslandLidar20182019_065_028.las",
    "BigIslandLidar20182019_066_028.las",
    "BigIslandLidar20182019_067_028.las",
    "BigIslandLidar20182019_047_029.las",
    "BigIslandLidar20182019_048_029.las",
    "BigIslandLidar20182019_049_029.las",
    "BigIslandLidar20182019_050_029.las",
    "BigIslandLidar20182019_051_029.las",
    "BigIslandLidar20182019_052_029.las",
    "BigIslandLidar20182019_053_029.las",
    "BigIslandLidar20182019_054_029.las",
    "BigIslandLidar20182019_055_029.las",
    "BigIslandLidar20182019_056_029.las",
    "BigIslandLidar20182019_057_029.las",
    "BigIslandLidar20182019_058_029.las",
    "BigIslandLidar20182019_059_029.las",
    "BigIslandLidar20182019_060_029.las",
    "BigIslandLidar20182019_061_029.las",
    "BigIslandLidar20182019_062_029.las",
    "BigIslandLidar20182019_063_029.las",
    "BigIslandLidar20182019_064_029.las",
    "BigIslandLidar20182019_065_029.las",
    "BigIslandLidar20182019_066_029.las",
    "BigIslandLidar20182019_067_029.las",
    "BigIslandLidar20182019_048_030.las",
    "BigIslandLidar20182019_049_030.las",
    "BigIslandLidar20182019_050_030.las",
    "BigIslandLidar20182019_051_030.las",
    "BigIslandLidar20182019_052_030.las",
    "BigIslandLidar20182019_053_030.las",
    "BigIslandLidar20182019_054_030.las",
    "BigIslandLidar20182019_055_030.las",
    "BigIslandLidar20182019_056_030.las",
    "BigIslandLidar20182019_057_030.las",
    "BigIslandLidar20182019_058_030.las",
    "BigIslandLidar20182019_059_030.las",
    "BigIslandLidar20182019_060_030.las",
    "BigIslandLidar20182019_061_030.las",
    "BigIslandLidar20182019_062_030.las",
    "BigIslandLidar20182019_063_030.las",
    "BigIslandLidar20182019_064_030.las",
    "BigIslandLidar20182019_065_030.las",
    "BigIslandLidar20182019_066_030.las",
    "BigIslandLidar20182019_067_030.las",
    "BigIslandLidar20182019_049_031.las",
    "BigIslandLidar20182019_050_031.las",
    "BigIslandLidar20182019_051_031.las",
    "BigIslandLidar20182019_052_031.las",
    "BigIslandLidar20182019_053_031.las",
    "BigIslandLidar20182019_054_031.las",
    "BigIslandLidar20182019_055_031.las",
    "BigIslandLidar20182019_056_031.las",
    "BigIslandLidar20182019_057_031.las",
    "BigIslandLidar20182019_058_031.las",
    "BigIslandLidar20182019_059_031.las",
    "BigIslandLidar20182019_060_031.las",
    "BigIslandLidar20182019_061_031.las",
    "BigIslandLidar20182019_062_031.las",
    "BigIslandLidar20182019_063_031.las",
    "BigIslandLidar20182019_064_031.las",
    "BigIslandLidar20182019_065_031.las",
    "BigIslandLidar20182019_066_031.las",
    "BigIslandLidar20182019_067_031.las",
    "BigIslandLidar20182019_051_032.las",
    "BigIslandLidar20182019_052_032.las",
    "BigIslandLidar20182019_053_032.las",
    "BigIslandLidar20182019_054_032.las",
    "BigIslandLidar20182019_055_032.las",
    "BigIslandLidar20182019_056_032.las",
    "BigIslandLidar20182019_057_032.las",
    "BigIslandLidar20182019_058_032.las",
    "BigIslandLidar20182019_059_032.las",
    "BigIslandLidar20182019_060_032.las",
    "BigIslandLidar20182019_061_032.las",
    "BigIslandLidar20182019_062_032.las",
    "BigIslandLidar20182019_063_032.las",
    "BigIslandLidar20182019_064_032.las",
    "BigIslandLidar20182019_065_032.las",
    "BigIslandLidar20182019_066_032.las",
    "BigIslandLidar20182019_052_033.las",
    "BigIslandLidar20182019_053_033.las",
    "BigIslandLidar20182019_054_033.las",
    "BigIslandLidar20182019_055_033.las",
    "BigIslandLidar20182019_056_033.las",
    "BigIslandLidar20182019_057_033.las",
    "BigIslandLidar20182019_058_033.las",
    "BigIslandLidar20182019_059_033.las",
    "BigIslandLidar20182019_060_033.las",
    "BigIslandLidar20182019_061_033.las",
    "BigIslandLidar20182019_062_033.las",
    "BigIslandLidar20182019_063_033.las"
]

# Loop through each .las file in the provided array with progress bar
for filename in tqdm(file_names, desc="Processing LAS files"):
    las_file = os.path.join(input_folder, filename)
    output_dem = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dem.tif")
    output_dem_tap = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dem_tap.tif")
    output_dem_cropped = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dem_cropped.tif")

    # Check if the LAS file exists before processing
    if not os.path.isfile(las_file):
        print(f"File not found: {las_file}. Skipping...")
        continue

    # Define the PDAL pipeline configuration
    pipeline_config = [
        {
            "type": "readers.las",
            "filename": las_file,
            "override_srs": "EPSG:6635"
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:2]"
        },
        {
            "type": "writers.gdal",
            "filename": output_dem,
            "output_type": "idw",
            "window_size": 30,
            "resolution": 0.2
        }
    ]

    # Execute the PDAL pipeline
    pipeline = pdal.Pipeline(json.dumps(pipeline_config))
    try:
        pipeline.execute()
        print(f"DEM created: {output_dem}")

        # Apply TAP using gdalwarp
        gdal_tap_command = [
            "gdalwarp",
            "-t_srs", "EPSG:6635",
            "-tr", "0.2", "0.2",
            "-tap",
            output_dem,
            output_dem_tap
        ]
        subprocess.run(gdal_tap_command, check=True)
        print(f"TAP applied to: {output_dem_tap}")

        # Find the matching polygon in the GeoPackage for cropping
        matching_polygon = gdf[gdf['filename'] == filename]
        if not matching_polygon.empty:
            # Save the matching polygon as a temporary shapefile for gdalwarp
            temp_cutline_path = "/tmp/temp_cutline.shp"
            matching_polygon.to_file(temp_cutline_path)

            # Crop the TAP-aligned DEM to the polygon
            gdal_crop_command = [
                "gdalwarp",
                "-cutline", temp_cutline_path,
                "-crop_to_cutline",
                output_dem_tap,
                output_dem_cropped
            ]
            subprocess.run(gdal_crop_command, check=True)
            print(f"Cropped DEM created: {output_dem_cropped}")

            # Clean up temporary files
            os.remove(temp_cutline_path)
        else:
            print(f"No matching polygon found in GeoPackage for {filename}")

        # Optionally, delete intermediate files
        os.remove(output_dem)
        os.remove(output_dem_tap)

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Failed to process {las_file}: {e}")
