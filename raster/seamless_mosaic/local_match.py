import os
import numpy as np
from osgeo import gdal
from math import log, floor
from typing import Tuple, List, Optional

##############################################################################
# HELPER FUNCTIONS (unchanged except for the vectorization in compute_local_distribution_map)
##############################################################################
def merge_rasters(input_array, output_image_folder, output_file_name="merge.tif"):
    output_path = os.path.join(output_image_folder, output_file_name)
    input_datasets = [gdal.Open(path) for path in input_array if gdal.Open(path)]
    gdal.Warp(
        output_path,
        input_datasets,
        format='GTiff',
    )
    print(f"Merged raster saved to: {output_path}")

def get_image_metadata(input_image_path):
    """
Get metadata of a TIFF image, including transform, projection, nodata, and bounds.

Args:
input_image_path (str): Path to the input image file.

Returns:
tuple: A tuple containing (transform, projection, nodata, bounds).
    """
    try:
        dataset = gdal.Open(input_image_path, gdal.GA_ReadOnly)
        if dataset is not None:
            # Get GeoTransform
            transform = dataset.GetGeoTransform()

            # Get Projection
            projection = dataset.GetProjection()

            # Get NoData value (assuming from the first band)
            nodata = dataset.GetRasterBand(1).GetNoDataValue() if dataset.RasterCount > 0 else None

            # Calculate bounds
            if transform:
                x_min = transform[0]
                y_max = transform[3]
                x_max = x_min + (dataset.RasterXSize * transform[1])
                y_min = y_max + (dataset.RasterYSize * transform[5])
                bounds = {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                }
            else:
                bounds = None

            dataset = None  # Close the dataset
            return transform, projection, nodata, bounds
        else:
            print(f"Could not open the file: {input_image_path}")
    except Exception as e:
        print(f"Error processing {input_image_path}: {e}")
    return None, None, None, None

def get_bounding_rectangle(image_paths: List[str]):
    """
Computes the minimum bounding rectangle (x_min, y_min, x_max, y_max)
that covers all input images.
    """
    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []

    for path in image_paths:
        transform, proj, nodata, bounds = get_image_metadata(path)
        if bounds is not None:
            x_mins.append(bounds["x_min"])
            y_mins.append(bounds["y_min"])
            x_maxs.append(bounds["x_max"])
            y_maxs.append(bounds["y_max"])

    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))

def compute_mosaic_coefficient_of_variation(
        image_paths: List[str],
        band_index: int = 1,
        nodata_value: float = None
) -> float:
    """
Computes the mosaic-level coefficient of variation for the specified band
across all images. This is a simplified example that loads entire bands.
For truly large imagery, adapt to read partial blocks/tiles at a time.
    """
    all_pixels = []

    for path in image_paths:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue
        band = ds.GetRasterBand(band_index)
        arr = band.ReadAsArray().astype(np.float64)
        if nodata_value is not None:
            mask = (arr != nodata_value)
            arr = arr[mask]
        all_pixels.append(arr)
        ds = None

    if len(all_pixels) == 0:
        return 0.0

    combined = np.concatenate(all_pixels)
    mean_val = np.mean(combined)
    std_val  = np.std(combined)
    if mean_val == 0:
        return 0.0

    return std_val / mean_val

def compute_block_size(
        catar: float,
        base_block_size: Tuple[int, int] = (10, 10),
        caref: float = 45.0/125.0
) -> Tuple[int, int]:
    """
Computes the grid size (M, N) using Eqs. (38) and (39) from the paper.
M = r * m
N = r * n
r = CAtar / CAref
    """
    if caref == 0:
        r = 1.0
    else:
        r = catar / caref
    m, n = base_block_size
    M = max(1, int(round(r * m)))
    N = max(1, int(round(r * n)))
    return (M, N)

def compute_reference_distribution_map(
        image_paths: List[str],
        bounding_rect: Tuple[float, float, float, float],
        M: int,
        N: int,
        num_bands: int,
        nodata_value: float = None
) -> np.ndarray:
    """
Divides the bounding rectangle into (M x N) blocks, computes
the average pixel value (across all images) in each block for each band.
Output shape = (M, N, num_bands).
    """
    print('Computing reference distribution map:')
    x_min, y_min, x_max, y_max = bounding_rect
    block_map = np.full((M, N, num_bands), np.nan, dtype=np.float64)

    sum_map = np.zeros((M, N, num_bands), dtype=np.float64)
    count_map = np.zeros((M, N, num_bands), dtype=np.float64)

    block_width = (x_max - x_min) / N
    block_height = (y_max - y_min) / M

    for idx, path in enumerate(image_paths):
        try:
            if has_run:
                print("")
        except NameError:
            pass
        has_run = True
        print(f'Image {idx}:')
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is None:
            continue
        gt = ds.GetGeoTransform()

        # Precompute grids for all rows and columns
        nX, nY = ds.RasterXSize, ds.RasterYSize
        col_index = np.arange(nX) + 0.5
        row_index = np.arange(nY) + 0.5
        Xgeo = gt[0] + col_index * gt[1]
        Ygeo = gt[3] + row_index * gt[5]
        Xgeo_2d, Ygeo_2d = np.meshgrid(Xgeo, Ygeo)
        print(f"Xgeo range: {Xgeo_2d.min()} to {Xgeo_2d.max()}")
        print(f"Ygeo range: {Ygeo_2d.min()} to {Ygeo_2d.max()}")

        for b in range(num_bands):
            print(f'Band {b}, ', end=' ')
            band = ds.GetRasterBand(b + 1)
            arr = band.ReadAsArray().astype(np.float64)

            if nodata_value is not None:
                valid_mask = (arr != nodata_value)
            else:
                valid_mask = np.ones_like(arr, dtype=bool)

            valid_inds = np.where(valid_mask)
            pix_vals = arr[valid_inds]
            pix_x = Xgeo_2d[valid_inds]
            pix_y = Ygeo_2d[valid_inds]

            # Compute block indices directly for all valid pixels
            block_cols = np.clip(((pix_x - x_min) / block_width).astype(int), 0, N - 1)
            block_rows = np.clip(((y_max - pix_y) / block_height).astype(int), 0, M - 1)

            # Use numpy's advanced indexing to accumulate
            np.add.at(sum_map[:, :, b], (block_rows, block_cols), pix_vals)
            np.add.at(count_map[:, :, b], (block_rows, block_cols), 1)

        ds = None

    # Avoid division by zero
    valid_counts = count_map > 0
    block_map[valid_counts] = sum_map[valid_counts] / count_map[valid_counts]
    return block_map

##############################################################################
# Updated function: compute_local_distribution_map
# Vectorized accumulation instead of per-pixel loop
##############################################################################
def compute_local_distribution_map(
        image_path: str,
        bounding_rect: Tuple[float, float, float, float],
        M: int,
        N: int,
        num_bands: int,
        nodata_value: float = None
) -> np.ndarray:
    """
For a single image, computes the (M x N x num_bands) block-level mean
without looping over each pixel. Uses vectorized accumulation with np.add.at.
    """
    x_min, y_min, x_max, y_max = bounding_rect
    local_map = np.full((M, N, num_bands), np.nan, dtype=np.float64)

    sum_map = np.zeros((M, N, num_bands), dtype=np.float64)
    count_map = np.zeros((M, N, num_bands), dtype=np.float64)

    ds = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds is None:
        return local_map

    gt = ds.GetGeoTransform()
    nX, nY = ds.RasterXSize, ds.RasterYSize
    block_width  = (x_max - x_min) / N
    block_height = (y_max - y_min) / M

    # Precompute geo-coordinates for all pixel centers
    col_index = np.arange(nX) + 0.5
    row_index = np.arange(nY) + 0.5
    Xgeo = gt[0] + col_index * gt[1]
    Ygeo = gt[3] + row_index * gt[5]
    Xgeo_2d, Ygeo_2d = np.meshgrid(Xgeo, Ygeo)

    for b in range(num_bands):
        band = ds.GetRasterBand(b+1)
        arr = band.ReadAsArray().astype(np.float64)

        if nodata_value is not None:
            valid_mask = (arr != nodata_value)
        else:
            valid_mask = np.ones_like(arr, dtype=bool)

        valid_inds = np.where(valid_mask)
        pix_vals = arr[valid_inds]
        pix_x = Xgeo_2d[valid_inds]
        pix_y = Ygeo_2d[valid_inds]

        # Compute block indices in a vectorized manner
        block_cols = ((pix_x - x_min) / block_width).astype(int)
        block_rows = ((y_max - pix_y) / block_height).astype(int)

        # Clip to stay in valid index range
        in_bounds = (
            (block_cols >= 0) & (block_cols < N) &
            (block_rows >= 0) & (block_rows < M)
        )
        block_cols = block_cols[in_bounds]
        block_rows = block_rows[in_bounds]
        pix_vals   = pix_vals[in_bounds]

        # Vectorized accumulation
        np.add.at(sum_map[:, :, b], (block_rows, block_cols), pix_vals)
        np.add.at(count_map[:, :, b], (block_rows, block_cols), 1)

    ds = None

    # Compute mean where count > 0
    valid_counts = (count_map > 0)
    local_map[valid_counts] = sum_map[valid_counts] / count_map[valid_counts]
    return local_map

def vectorized_bilinear_interpolation(grid, row_fs, col_fs):
    """
Perform bilinear interpolation for multiple points (row_fs, col_fs)
at once on a 2D grid.
    """
    M, N = grid.shape
    row_fs = np.clip(row_fs, 0, M - 1)
    col_fs = np.clip(col_fs, 0, N - 1)

    row0 = np.floor(row_fs).astype(int)
    col0 = np.floor(col_fs).astype(int)
    row1 = np.clip(row0 + 1, 0, M - 1)
    col1 = np.clip(col0 + 1, 0, N - 1)

    fr = row_fs - row0
    fc = col_fs - col0

    val00 = grid[row0, col0]
    val01 = grid[row0, col1]
    val10 = grid[row1, col0]
    val11 = grid[row1, col1]

    val0 = val00 * (1 - fc) + val01 * fc
    val1 = val10 * (1 - fc) + val11 * fc
    return val0 * (1 - fr) + val1 * fr

##############################################################################
# apply_local_correction: WITH GAMMA BOUNDS
##############################################################################
def apply_local_correction(
        image_path: str,
        bounding_rect: Tuple[float, float, float, float],
        ref_map: np.ndarray,         # M x N x num_bands
        local_map: np.ndarray,       # M x N x num_bands
        output_path: str,
        floor_value: Optional[float] = None,
        nodata_value: float = 0.0,
        alpha: float = 1.0,
        gamma_bounds: Optional[Tuple[float, float]] = None
):
    """
    Applies the local correction (paper’s formula):
        P_res(x,y) = ( P_in(x,y) ^ ( log(M_ref(x,y)) / log(M_in(x,y)) ) )
                     * alpha

    plus two modifications:
      1) We set output to NoData if M_ref or M_in is invalid.
      2) We do the multiply-by-alpha step explicitly at the end.

    Args:
        image_path (str): path to input image
        bounding_rect (Tuple[float,float,float,float]): (x_min, y_min, x_max, y_max)
        ref_map (np.ndarray): reference distribution map, shape (M,N,num_bands)
        local_map (np.ndarray): local distribution map, shape (M,N,num_bands)
        output_path (str): path to write the corrected image
        floor_value (float or None): if not None, clamp M_ref, M_in, and P_in above this
        nodata_value (float): NoData for the input image
        alpha (float): final scaling factor to restore original scale
        gamma_bounds (tuple or None): if given, (low, high) clamp for gamma
    """
    x_min, y_min, x_max, y_max = bounding_rect
    M, N, num_bands = ref_map.shape

    ds_in = gdal.Open(image_path, gdal.GA_ReadOnly)
    if ds_in is None:
        raise RuntimeError(f"Could not open {image_path}")

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        ds_in.RasterXSize,
        ds_in.RasterYSize,
        num_bands,
        gdal.GDT_Float32
    )
    out_ds.SetGeoTransform(ds_in.GetGeoTransform())
    out_ds.SetProjection(ds_in.GetProjection())

    block_width  = (x_max - x_min) / N
    block_height = (y_max - y_min) / M

    for b in range(num_bands):
        band_in = ds_in.GetRasterBand(b+1)
        arr_in  = band_in.ReadAsArray().astype(np.float32)

        nX = ds_in.RasterXSize
        nY = ds_in.RasterYSize
        gt = ds_in.GetGeoTransform()
        col_index = np.arange(nX) + 0.5
        row_index = np.arange(nY) + 0.5
        Xgeo = gt[0] + (col_index * gt[1])
        Ygeo = gt[3] + (row_index * gt[5])
        Xgeo_2d, Ygeo_2d = np.meshgrid(Xgeo, Ygeo)

        arr_out = np.full_like(arr_in, nodata_value, dtype=np.float32)
        valid_mask = (arr_in != nodata_value)
        valid_inds = np.where(valid_mask)

        px_vals = arr_in[valid_inds]   # P_in(x,y)
        px_x    = Xgeo_2d[valid_inds]
        px_y    = Ygeo_2d[valid_inds]

        row_fs = (y_max - px_y) / block_height
        col_fs = (px_x - x_min) / block_width

        ref_band_2d = ref_map[:, :, b]
        loc_band_2d = local_map[:, :, b]

        # Interpolate M_ref, M_in at each pixel
        Mrefs = vectorized_bilinear_interpolation(ref_band_2d, row_fs, col_fs)
        Mins  = vectorized_bilinear_interpolation(loc_band_2d, row_fs, col_fs)

        # Floor clamp if requested
        if floor_value is not None:
            px_vals = np.maximum(px_vals, floor_value)
            Mrefs   = np.maximum(Mrefs,   floor_value)
            Mins    = np.maximum(Mins,    floor_value)

        # Identify pixels with valid Mrefs, Mins, px_vals
        valid_gamma_mask = (Mrefs > 0) & (Mins > 0) & (px_vals > 0)

        # Compute gamma, but skip invalid pixels
        gamma_raw = np.zeros_like(px_vals, dtype=np.float32)
        valid_logs = valid_gamma_mask.copy()
        gamma_raw[valid_logs] = np.log(Mrefs[valid_logs]) / np.log(Mins[valid_logs])

        # Apply gamma bounds if provided
        if gamma_bounds is not None:
            low, high = gamma_bounds
            gamma_raw = np.clip(gamma_raw, low, high)

        # UPDATED HERE: Move alpha multiplication to a final step
        # 1) Exponent step
        out_vals = np.copy(px_vals)
        out_vals[valid_logs] = (px_vals[valid_logs] ** gamma_raw[valid_logs])

        # 2) If M_ref or M_in is invalid, set to NoData
        #    (catches cases where the block is partially covered or sums were 0)
        invalid_logs = ~valid_logs
        out_vals[invalid_logs] = nodata_value

        # 3) UPDATED HERE: multiply by alpha explicitly at the end
        #    This will “bring values back to their original scale”
        #    if alpha was used as your desired overall scale factor.
        out_vals[valid_logs] *= alpha

        arr_out[valid_inds] = out_vals
        out_band = out_ds.GetRasterBand(b+1)
        out_band.WriteArray(arr_out)
        out_band.SetNoDataValue(nodata_value)

    ds_in = None
    out_ds.FlushCache()
    out_ds = None

##############################################################################
# MAIN FUNCTION: Now accepts gamma_bounds
##############################################################################
def process_local_histogram_matching(
        input_image_paths_array_local: List[str],
        output_image_folder: str,
        output_local_basename: str,
        reference_mean: float = 125.0,
        reference_std: float  = 45.0,
        base_block_size: Tuple[int,int] = (10,10),
        global_nodata_value: float = -9999,
        alpha: float = 1.0,
        floor_value: Optional[float] = None,
        gamma_bounds: Optional[Tuple[float, float]] = None
):
    """
    Local histogram matching with EXACT math from the paper, plus:
      - optional floor_value to clamp negative/zero
      - optional gamma_bounds to clamp the exponent range
      - an explicit alpha multiplication step at the end (if desired)

    Steps:
      1) bounding rectangle
      2) mosaic-level coeff of variation
      3) compute (M, N) block layout
      4) compute reference distribution map
      5) for each image, compute local map
      6) apply correction
      7) merge corrected rasters
    """
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder, exist_ok=True)

    # 1) bounding rectangle
    bounding_rect = get_bounding_rectangle(input_image_paths_array_local)
    x_min, y_min, x_max, y_max = bounding_rect
    print(f"Bounding rectangle: {bounding_rect}")

    # 2) mosaic-level coefficient_of_variation
    catar = compute_mosaic_coefficient_of_variation(
        input_image_paths_array_local,
        band_index=1,
        nodata_value=global_nodata_value
    )
    print(f"Mosaic coefficient of variation (CAtar) = {catar:.4f}")

    # 3) compute (M, N)
    caref = reference_std / reference_mean
    M, N = compute_block_size(catar, base_block_size=base_block_size, caref=caref)
    print(f"Block layout: M={M}, N={N}")

    # 4) reference distribution map
    ds_tmp = gdal.Open(input_image_paths_array_local[0], gdal.GA_ReadOnly)
    num_bands = ds_tmp.RasterCount
    ds_tmp = None

    ref_map = compute_reference_distribution_map(
        input_image_paths_array_local,
        bounding_rect,
        M, N,
        num_bands,
        nodata_value=global_nodata_value
    )

    # 5) For each image, compute local map and appxly correction
    corrected_paths = []
    for img_path in input_image_paths_array_local:
        loc_map = compute_local_distribution_map(
            img_path,
            bounding_rect,
            M, N,
            num_bands,
            nodata_value=global_nodata_value
        )

        in_name = os.path.basename(img_path)
        out_name = os.path.splitext(in_name)[0] + output_local_basename + ".tif"
        out_path = os.path.join(output_image_folder, out_name)

        apply_local_correction(
            img_path,
            bounding_rect,
            ref_map,
            loc_map,
            out_path,
            floor_value=floor_value,
            nodata_value=global_nodata_value,
            alpha=alpha,
            gamma_bounds=gamma_bounds
        )
        corrected_paths.append(out_path)
        print(f"Local-histogram-corrected saved: {out_path}")

    # 6) Merge final corrected rasters
    print("Merging local-corrected rasters...")
    merge_rasters(corrected_paths, output_image_folder, output_file_name=f"Merged{output_local_basename}.tif")
    print("Local histogram matching complete!")