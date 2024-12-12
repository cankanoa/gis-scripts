import os
import numpy as np
from osgeo import gdal, ogr
from shapely.geometry import shape, Polygon, mapping
from shapely.affinity import translate
import fiona
import math

def crop_dem_to_polygon(dem_path, plot_polygon_path, output_dem_path):
    """Crop the DEM to the given polygon."""
    gdal.Warp(output_dem_path, dem_path, cutlineDSName=plot_polygon_path, cropToCutline=True, dstNodata=-9999)

def calculate_slope_azimuth(dem_path):
    """Calculate the slope and azimuth of the DEM."""
    ds = gdal.Open(dem_path)
    band = ds.GetRasterBand(1)
    dem_array = band.ReadAsArray()
    gt = ds.GetGeoTransform()

    # Calculate gradients
    x, y = np.gradient(dem_array, gt[1], gt[5])

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(x**2 + y**2)))

    # Azimuth (aspect)
    azimuth = np.degrees(np.arctan2(-y, x))
    azimuth[azimuth < 0] += 360

    average_slope = np.mean(slope[np.isfinite(slope)])
    average_azimuth = np.mean(azimuth[np.isfinite(azimuth)])

    return average_slope, average_azimuth

def create_offset_polygon(plot_polygon_path, camera_angle, drone_height, output_polygon_path):
    """Create an offset polygon where the drone needs to fly."""
    with fiona.open(plot_polygon_path, 'r') as source:
        # Check if the GeoPackage has features
        if len(source) == 0:
            raise ValueError("The provided GeoPackage file has no features.")

        # Extract the first feature's geometry
        first_feature = next(iter(source))
        plot_geom = shape(first_feature['geometry'])

        # Get the CRS from the source
        source_crs = source.crs

    # Calculate the offset distance
    offset_distance = drone_height * math.tan(math.radians(camera_angle))

    # Translate the polygon along the azimuth
    azimuth_radians = math.radians(90)  # Assume perpendicular offset
    offset_x = offset_distance * math.cos(azimuth_radians)
    offset_y = offset_distance * math.sin(azimuth_radians)

    offset_polygon = translate(plot_geom, xoff=offset_x, yoff=offset_y)

    # Save the new polygon as a GeoPackage
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'}
    }
    with fiona.open(output_polygon_path, 'w', driver='GPKG', crs=source_crs, schema=schema, layer='flight_boundary') as sink:
        sink.write({
            'geometry': mapping(offset_polygon),
            'properties': {'id': 1}
        })

def main(dem_path, plot_polygon_path, camera_angle, drone_height, output_flight_boundary_path):
    cropped_dem_path = "cropped_dem.tif"

    print("Cropping DEM to polygon...")
    crop_dem_to_polygon(dem_path, plot_polygon_path, cropped_dem_path)

    print("Calculating slope and azimuth...")
    slope, azimuth = calculate_slope_azimuth(cropped_dem_path)
    print(f"Average Slope: {slope:.2f} degrees")
    print(f"Average Azimuth: {azimuth:.2f} degrees")

    print("Creating offset polygon...")
    create_offset_polygon(plot_polygon_path, camera_angle, drone_height, output_flight_boundary_path)
    print(f"Offset polygon saved to {output_flight_boundary_path}")


dem_path = "/Users/kanoalindiwe/Downloads/temp/smooth2.tif"
plot_polygon_path = "/Users/kanoalindiwe/Downloads/temp/bounds.gpkg"
camera_angle = 45  # degrees
drone_height = 30  # meters
output_flight_boundary_path = "/Users/kanoalindiwe/Downloads/temp/flight_boundary.gpkg"
main(dem_path, plot_polygon_path, camera_angle, drone_height, output_flight_boundary_path)