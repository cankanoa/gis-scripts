import os
from osgeo import gdal, ogr

# Convert paths to WSL format
geopackage_path = "/mnt/x/PROJECTS_2/Big_Island/LandCover/Extents/Ahupuaa_PuuWaawaa/Ahupuaa_PuuWaawaa_1kmBuffer.gpkg"
raster_paths = [
    "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171208_36cm_WV03_BAB_016445318010/20171208_36cm_WV03_BAB_FLAASH/17DEC08211840-M1BS-016445318010_01_P015_FLAASH.dat",
    "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171208_36cm_WV03_BAB_016445318010/20171208_36cm_WV03_BAB_FLAASH/17DEC08211841-M1BS-016445318010_01_P016_FLAASH.dat",
    "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171208_36cm_WV03_BAB_016445319010/20171208_36cm_WV03_BAB_2_FLAASH/17DEC08211758-M1BS-016445319010_01_P003_FLAASH.dat",
    "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171208_36cm_WV03_BAB_016445319010/20171208_36cm_WV03_BAB_2_FLAASH/17DEC08211800-M1BS-016445319010_01_P004_FLAASH.dat",
    "/mnt/s/Satellite_Imagery/Big_Island/Unprocessed/PuuWaawaaImages/20171208_36cm_WV03_BAB_016445319010/20171208_36cm_WV03_BAB_2_FLAASH/17DEC08211801-M1BS-016445319010_01_P005_FLAASH.dat"
]
output_folder = "/mnt/x/PROJECTS_2/Big_Island/LandCover/Input/satellite/1kmClip"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Open the GeoPackage and get the extent
driver = ogr.GetDriverByName('GPKG')
geopackage = driver.Open(geopackage_path, 0)  # 0 means read-only

if not geopackage:
    raise Exception("Could not open GeoPackage")

layer = geopackage.GetLayer(0)  # Assuming you want the first layer
extent = layer.GetExtent()  # (minX, maxX, minY, maxY)

# Clip each raster
for raster in raster_paths:
    # Open the raster
    src_ds = gdal.Open(raster)
    if not src_ds:
        print(f"Failed to open raster: {raster}")
        continue

    # Create a new raster dataset with the desired dimensions and data type
    output_file = os.path.join(output_folder, os.path.basename(raster).replace('.dat', '_clipped.dat'))
    gdal.Translate(output_file, src_ds,
                   projWin=extent,  # Specify the extent for clipping
                   format='ENVI')  # Output format as ENVI .dat

    print(f"Clipped {raster} to {output_file}")

print("Clipping completed. Output files are in:", output_folder)
