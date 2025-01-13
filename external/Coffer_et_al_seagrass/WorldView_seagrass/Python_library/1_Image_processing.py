# import libraries
import glob
import os
import sys
import subprocess

# define all necessary input variables
 # name of file to be processed 
sat_file = "Back_Sound_20130327"
# set the aoi extent with ul_lon, ul_lat, lr_lon, lr_lat (decimal degrees)
extent_coord = [-76.7, 34.7, -76.5, 34.5] 
# define the EPSG code as a character string: https://spatialreference.org/ref/epsg/
epsg_code = "EPSG:32618" 
# set working directory
wdir = "C:\\Users\\PWHITMAN\\Documents\\For_Megan\\WorldView_seagrass" 

# install required libraries
subprocess.call([sys.executable, "-m", "pip", "install", "-r", os.path.join(wdir, "Python_library", "requirements.txt"), "--user"])

# import seagrass library
library_dir = os.path.join(wdir, "Python_library")
sys.path.insert(0, library_dir)
from seagrass_lib import *

# increase cache size to avoid memory constraints
from osgeo import gdal
gdal.SetCacheMax(1000000000)
 
# list multispectral tiles in zipped folder 
input_zip = os.path.join(wdir, "Input_data", "1_Processing_data", "1_Zipped_data", sat_file + ".zip")
folder_df = list_files(zip_fp = input_zip, aoi_extent = extent_coord)

# unzip data
dir_list = folder_df.loc[folder_df['AOI_COVERAGE'] > 0, 'DIRECTORY'].tolist()
output_folder = os.path.join(wdir, "Input_data", "1_Processing_data", "2_Unzipped_data", sat_file)
unzip_tiles(zip_fp = input_zip, tile_dir = dir_list, output_dir = output_folder)

# perform radiometric calibration and embed DOS value in the metadata of each image
unzipped_images = recursive_search(os.path.join(wdir, "Input_data", "1_Processing_data", "2_Unzipped_data", sat_file), pattern = "*.TIF")
for i in range(len(unzipped_images)):
    input_image = unzipped_images[i]
    output_image = os.path.join(wdir, "Input_data", "1_Processing_data", "3_Radiometric_calibration", sat_file, os.path.basename(input_image))
    rad_cal(image_fp = input_image, output_fp = output_image, aoi_extent = extent_coord)
    embed_dos_val(image_fp = output_image)

# atmospheric correction
rad_cal_images = glob.glob(os.path.join(wdir, "Input_data", "1_Processing_data", "3_Radiometric_calibration", sat_file, "*.TIF"))
min_dos = min_dos_value(rad_cal_images)
for i in range(len(rad_cal_images)):
    input_image = rad_cal_images[i]
    output_image = os.path.join(wdir, "Input_data", "1_Processing_data", "4_Atmospheric_correction", sat_file, os.path.basename(input_image))
    atm_cor(image_fp = input_image, output_fp = output_image, rayleighExp = 4.75, dos_value = min_dos)

# project images
atm_corr_images = glob.glob(os.path.join(wdir, "Input_data", "1_Processing_data", "4_Atmospheric_correction", sat_file, "*.TIF"))
for i in range(len(atm_corr_images)):
    input_image = atm_corr_images[i]
    output_image = os.path.join(wdir, "Input_data", "1_Processing_data", "5_Projected_image", sat_file, os.path.basename(input_image))
    project_image(image_fp = input_image, output_fp = output_image, target_coord = epsg_code, res = 2, rpc = True, resampling_method = "bilinear")

# clip images
proj_images = glob.glob(os.path.join(wdir, "Input_data", "1_Processing_data", "5_Projected_image", sat_file, "*.TIF"))
for i in range(len(proj_images)):
    input_image = proj_images[i]
    output_image = os.path.join(wdir, "Input_data", "1_Processing_data", "6_Clipped_image", sat_file, os.path.basename(input_image))
    clip_image(image_fp = input_image, output_fp = output_image, aoi_extent = extent_coord)