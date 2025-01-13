# import libraries
import os
import sys
import subprocess

# define all necessary input variables
# name of processed satellite file
sat_file = "13MAR27161235-M1BS-012697279010_01_P002" 
# name of the scene that the satellite file belongs to 
scene = "Back_Sound_20130327" 
# set working directory (same directory as 1_Image_processing.py)
wdir = "C:\\Users\\PWHITMAN\\Documents\\For_Megan\\WorldView_seagrass" 

# install required libraries
subprocess.call([sys.executable, "-m", "pip", "install", "-r", os.path.join(wdir, "Python_library", "requirements.txt"), "--user"])

# import seagrass library
library_dir = os.path.join(wdir, "Python_library")
sys.path.insert(0, library_dir)
from seagrass_lib import *

# increase cache size to avoid memory constraints
from osgeo import gdal
gdal.SetCacheMax(2000000000)

# split multipart polygon into singlepart polygon
input_shp = os.path.join(wdir, "Input_data", "2_Classification_data", "1_ROIs", scene, sat_file, sat_file + ".shp")
output_shp = os.path.join(wdir, "Input_data", "2_Classification_data", "1_ROIs", scene, sat_file, sat_file + "_singlepart.shp") 
multipart_to_singlepart(shp_fp = input_shp, out_fp = output_shp)

# extract image ROIs using singlepart polygons
input_image = os.path.join(wdir, "Input_data", "1_Processing_data", "6_Clipped_image", scene, sat_file + ".TIF")
input_shp = os.path.join(wdir, "Input_data", "2_Classification_data", "1_ROIs", scene, sat_file, sat_file + "_singlepart.shp") 
output_ROIs = os.path.join(wdir, "Input_data", "2_Classification_data", "1_ROIs", scene, sat_file)
shp_to_roi(image_fp = input_image, output_dir = output_ROIs, shp_fp = input_shp, field_name = 'CLASS_NAME')

# train DCNN with image ROIs
dcnn_fp = os.path.join(wdir, "Input_data", "2_Classification_data", "2_DCNN_model", scene, sat_file, sat_file + ".h5")
input_data = os.path.join(wdir, "Input_data", "2_Classification_data", "1_ROIs", scene, sat_file)
image_classes = roi_classes(shp_fp = input_shp, field_name = 'CLASS_NAME')
train_dcnn(cnnFileName = dcnn_fp, training_data_directory = input_data, class_names = image_classes, numChannels = 8, dimension = 3)

# classify input image with trained dcnn
input_image = os.path.join(wdir, "Input_data", "1_Processing_data", "6_Clipped_image", scene, sat_file + ".TIF")
output_classification = os.path.join(wdir, "Input_data", "2_Classification_data", "3_Classified_image", scene, sat_file, "ouput_classification.TIF")
dcnn_fp = os.path.join(wdir, "Input_data", "2_Classification_data", "2_DCNN_model",  scene, sat_file, sat_file + ".h5")
dcnn_classification(image_fp = input_image, dcnn_fp = dcnn_fp, output_fp = output_classification)