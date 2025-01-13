from stretch_spectral_values import stretch_spectral_values
from local_match import process_local_histogram_matching
from global_match import process_global_histogram_matching

# # -------------------- Stretch Spectral Values
# input_image_paths_array = [
#     "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar.tif",
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar.tif",
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar.tif",
#     # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar.tif',
#     # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/3spot.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/4spot.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/tempspot.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu3.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu4.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu5.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu15.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu16.tif',
# ]
# # Dictionary {x,y}, where x is the input value and y is the output stretched value.
# # int or float for absolute value, % to calculate percent value on all numbers, @ to calculate percent value on positive numbers.
# # stretch_dictionary = {
# #     "0%": 0,
# #     "10@": '10@',
# #     "100%": "100%",
# # }
#
# stretch_dictionary = {
#     "0%": 0,
#     "100%": '1',
# }
#
# output_image_folder = "/Users/kanoalindiwe/Downloads/temp/"
# output_basename = "_stretch01"
# stretch_spectral_values(
#         input_image_paths_array,
#         output_image_folder,
#         output_basename,
#         stretch_dictionary,
#         smoothing=0,
#         dtype_override=None,
#         # offset=None
# )




# ---------------------------------------- Global Histogram Matching
# input_image_paths_array = [
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar.tif",
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar.tif",
#     # "/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar.tif",
#     # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar.tif',
#     # '/Users/kanoalindiwe/Downloads/resources/worldview/Flaash_OrthoFrom20182019Lidar/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/3subset.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/4subset.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu3.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu4.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu5.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu15.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/ClippedToPuu16.tif',
#     '/Users/kanoalindiwe/Downloads/temp/testDroneImages/GeoNodata/DJI1_Geo_No.tif',
#     '/Users/kanoalindiwe/Downloads/temp/testDroneImages/GeoNodata/DJI2_Geo_No.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif3_replaced_No.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif4_replaced_No.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif5_replaced_No.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif15_replaced_No.tif',
#     # '/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/FLAASH_Ortho_PanCliptif16_replaced_No.tif',
#
# ]
# output_image_folder = "/Users/kanoalindiwe/Downloads/temp/testDroneImages/global"
# # output_image_folder = "/Users/kanoalindiwe/Downloads/temp/tempBounds/raster/"
# output_global_basename = "_GlobalMatch"
# custom_mean_factor = 3 # Defualt 1; 3 works well sometimes
# custom_std_factor = 1 # Defualt 1
# process_global_histogram_matching(input_image_paths_array, output_image_folder, output_global_basename, custom_mean_factor, custom_std_factor)



# -------------------- Local Histogram Matching

input_image_paths_array = [
    # "/Users/kanoalindiwe/Downloads/temp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif",
    # "/Users/kanoalindiwe/Downloads/temp/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif",
    # "/Users/kanoalindiwe/Downloads/temp/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar_GlobalMatch.tif",
    # '/Users/kanoalindiwe/Downloads/temp/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar_GlobalMatch2.tif',
    # '/Users/kanoalindiwe/Downloads/temp/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar_GlobalMatch2.tif',
    '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif',
    '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif',
    '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar_GlobalMatch.tif',
    '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif',
    '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar_GlobalMatch.tif',
    # '/Users/kanoalindiwe/Downloads/temp/3spot.tif',
    # '/Users/kanoalindiwe/Downloads/temp/4spot.tif',
    # '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/DJI1_Geo_No_GlobalMatch.tif',
    # '/Users/kanoalindiwe/Downloads/temp/testDroneImages/global/DJI2_Geo_No_GlobalMatch.tif',
    # "/Users/kanoalindiwe/Downloads/temp/17DEC08211758-M1BS-016445319010_01_P003_FLAASH_OrthoFrom20182019Lidar_GlobalMatch_362Added.tif",
    # "/Users/kanoalindiwe/Downloads/temp/17DEC08211800-M1BS-016445319010_01_P004_FLAASH_OrthoFrom20182019Lidar_GlobalMatch_362Added.tif",
    # "/Users/kanoalindiwe/Downloads/temp/17DEC08211801-M1BS-016445319010_01_P005_FLAASH_PuuSubset_OrthoFrom20182019Lidar_GlobalMatch_362Added.tif",
    # '/Users/kanoalindiwe/Downloads/temp/17DEC08211840-M1BS-016445318010_01_P015_FLAASH_OrthoFrom20182019Lidar_GlobalMatch2_362Added.tif',
    # '/Users/kanoalindiwe/Downloads/temp/17DEC08211841-M1BS-016445318010_01_P016_FLAASH_OrthoFrom20182019Lidar_GlobalMatch2_362Added.tif',
]
output_image_folder = "/Users/kanoalindiwe/Downloads/temp/testDroneImages/local"
output_local_basename = "_LocalMatch"

process_local_histogram_matching(
    input_image_paths_array,
    output_image_folder,
    output_local_basename,
    target_blocks_per_image = 100,
    global_nodata_value=-9999,
)
