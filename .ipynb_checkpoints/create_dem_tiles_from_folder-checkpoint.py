import os
import json
import pdal
import subprocess
import geopandas as gpd
from tqdm import tqdm  # Import tqdm for progress bar

# Define input and output folders
input_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/PuuLidar"
output_folder = "/mnt/x/Imagery/Lidar/Big_Island/lidar/PuuLidar/3m"
geopackage_path = "/mnt/c/Users/admin/Downloads/Cliff/outline.gpkg"
layer_name = "outline"  # Replace with the actual layer name in your GeoPackage

# Load the GeoPackage
gdf = gpd.read_file(geopackage_path, layer=layer_name)

# Get list of .las files
las_files = [f for f in os.listdir(input_folder) if f.endswith(".las")]

# Loop through each .las file in the input folder with progress bar
for filename in tqdm(las_files, desc="Processing LAS files"):
    las_file = os.path.join(input_folder, filename)
    output_dem = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dem.tif")
    output_dem_tap = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dsm_tap.tif")
    output_dem_cropped = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_dem_cropped.tif")

    # Define the PDAL pipeline configuration
    pipeline_config = [
        {
            "type": "readers.las",
            "filename": las_file,
            "override_srs": "EPSG:6635"
        },
        {
            "type": "filters.range",
            "limits": "Classification[2:5]"
        },
        {
            "type": "writers.gdal",
            "filename": output_dem,
            "output_type": "idw",
            "window_size": 30,
            "resolution": 3
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
        # matching_polygon = gdf[gdf['filename'] == filename]
        # if not matching_polygon.empty:
        #     # Save the matching polygon as a temporary shapefile for gdalwarp
        #     temp_cutline_path = "/tmp/temp_cutline.shp"
        #     matching_polygon.to_file(temp_cutline_path)
        #
        #     # Crop the TAP-aligned DEM to the polygon
        #     gdal_crop_command = [
        #         "gdalwarp",
        #         "-cutline", temp_cutline_path,
        #         "-crop_to_cutline",
        #         output_dem_tap,
        #         output_dem_cropped
        #     ]
        #     subprocess.run(gdal_crop_command, check=True)
        #     print(f"Cropped DEM created: {output_dem_cropped}")
        #
        #     # Clean up temporary files
        #     os.remove(temp_cutline_path)
        # else:
        #     print(f"No matching polygon found in GeoPackage for {filename}")

        # Optionally, delete intermediate files
        os.remove(output_dem)
        # os.remove(output_dem_tap)

    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Failed to process {las_file}: {e}")
