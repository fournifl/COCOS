"""
Created on Thu May 29 16:42:25 2023

@author: ffournier
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from glob import glob
from metpy.interpolate import interpolate_to_grid
from skimage.morphology import binary_dilation
import json
import string
from copy import copy
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdb
from scipy.interpolate import griddata
from osgeo import gdal, osr
# from dev_georef_tools import load_rectification_geo_system_2, rotate_vector, rotateAndScale
import collections


def rotate_vector(data, theta):
    # make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # rotate data vector
    return data.dot(rotation_matrix)


def read_json_to_dict(filename):
    """

    Parameters
    ----------
    filename : string
        name of the file to load

    Returns
    -------
    dict : dictionnary containing numpy arrays
    """
    try:
        with open(filename, 'r') as infile:
            dict = json.load(infile)
            for k in dict.keys():
                if type(dict[k]) is list:
                    dict[k] = np.array(dict[k])
            return dict
    except IOError:
        raise


def load_rectification_geo_system_2(pose_file_path):
    """

    :param pose_file_path:
    :return:
    Grid_Coordinate_System: horizontal reference frame used to generate camera pose file
    UTM_zone: UTM zone if the Grid was generated in UTM system, otherwise: "None"
    Grid_Coordinate_System_Offset: coordinates of (Xo,Yo) if an offset was applied, if not: "None"
    Off_set: Boolean flag indicating if an offset was applied.
    Altitude_Datum_System: vertical reference of camera position.
    """
    pose_dict = read_json_to_dict(pose_file_path)
    Grid_Coordinate_System = pose_dict['Grid_Coordinate_System']
    UTM_zone = pose_dict['UTM_Zone']
    Grid_Coordinate_System_Offset = pose_dict['Grid_Coordinate_System_Offset']
    Off_set = pose_dict['Off_Set']
    Altitude_Datum_Name = pose_dict['Altitude_Datum_Name']
    Rotated_system = pose_dict['Rotated_system']
    Rotation_angle = pose_dict['Rotation_angle']

    return Grid_Coordinate_System, UTM_zone, Grid_Coordinate_System_Offset, Off_set, Altitude_Datum_Name, \
           Rotated_system, Rotation_angle


def rotateAndScale(img, scaleFactor=0.5, degreesCCW=30):
    oldY, oldX = img.shape  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                scale=scaleFactor)  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    return rotatedImg


def hemisphere_UTM_checker(utm_zone_string):
    """

    :param utm_zone_string: example north "31T" or "19H" south
    :return: hemisphere
    """
    alphabet_string_southern_zones = string.ascii_uppercase[0:13]
    # see : https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#/media/File:Universal_Transverse_Mercator_zones.svg
    if utm_zone_string[-1] in alphabet_string_southern_zones:
        hemisphere = "southern"
    else:
        hemisphere = "northern"
    return hemisphere


def compute_gathered_grid(grids_x, grids_y):
    resolutions = {}
    for i in grids_x.keys():
        xmin_tmp = np.min(grids_x[i])
        xmax_tmp = np.max(grids_x[i])
        ymin_tmp = np.min(grids_y[i])
        ymax_tmp = np.max(grids_y[i])
        resolution_tmp = np.around(grids_x[i][0, 1] - grids_x[i][0, 0], decimals=2)
        resolutions[i] = np.around(resolution_tmp, decimals=2)
        print(f'resolution: {resolution_tmp}')

        if i == 0:
            xmin = xmin_tmp
            xmax = xmax_tmp
            ymin = ymin_tmp
            ymax = ymax_tmp
            resolution = resolution_tmp

        else:
            if xmin > xmin_tmp:
                xmin = xmin_tmp
            if xmax < xmax_tmp:
                xmax = xmax_tmp
            if ymin > ymin_tmp:
                 ymin = ymin_tmp
            if ymax < ymax_tmp:
                ymax = ymax_tmp
            # keep coarser resolution
            if resolution > resolution_tmp:
                resolution = resolution_tmp
    x = np.arange(xmin, xmax, resolution)
    y = np.arange(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y, resolution, resolutions


def transform_pixel_array_to_real(px, py, px_w, px_h, rot1, rot2, xoffset, yoffset):
    """
    Transform pixel ccordinate(s) to real world coordinate(s), based on upper left coordinate of tiff file being treated.

    FF Waves'n see

    Parameters
    ----------
    px: float or numpy array
    horizontal pixel coordinate(s)
    py: float or numpy array
    vertical pixel coordinate(s)
    px_w: pixel width in metres
    px_h: pixel height in metres
    rot1: rotation angle
    rot2: rotation angle
    xoffset: float
    upper left horizontal coordinate of tiff file being treated.
    yoffset: float
    upper left vertical coordinate of tiff file being treated.

    Returns
    -------
    posx: horizontal real world coordinate(s)
    posy: vertical real world coordinate(s)

    """
    # supposing x and y are your pixel coordinate this is how to get the coordinate in space.
    posx = px_w * px + rot1 * py + xoffset
    posy = rot2 * px + px_h * py + yoffset

    # shift to the center of the pixel
    posx += px_w / 2.0
    posy += px_h / 2.0
    return posx, posy


def interpolate_results_on_same_grid(tif_files, interp_method):
    results_Dk_common_grid = {}

    grids_x = {}
    grids_y = {}
    data = {}
    for i, tif_file in enumerate(tif_files):
        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
        ds.GetProjectionRef()

        # Read the array and the transformation
        arr = ds.ReadAsArray()

        # corner coordinates and pixel width
        xoffset, px_w, rot1, yoffset, rot2, px_h = ds.GetGeoTransform()
        print('px_w:', px_w)
        print('px_h:', px_h)
        x = np.arange(0, ds.RasterXSize, 1)
        y = np.arange(0, ds.RasterYSize, 1)
        grid_px, grid_py = np.meshgrid(x, y)
        X, Y = transform_pixel_array_to_real(grid_px, grid_py, px_w, px_h, rot1, rot2, xoffset, yoffset)

        grids_x[i] = X
        grids_y[i] = Y
        data[i] = arr

    # compute gathered grid
    X, Y, resolution, resolutions = compute_gathered_grid(grids_x, grids_y)

    # add central cam data on gathered grid last, as pixel footprint is better compared to lateral cameras
    for i in grids_x.keys():
        values = data[i][:, :].flatten()
        # points = (np.vstack([grids_x[i].flatten(), grids_y[i].flatten()])).T
        # Z = griddata(points, values, (X, Y), method='linear')
        X, Y, Z = interpolate_to_grid(grids_x[i].flatten(), grids_y[i].flatten(), values,
                                       interp_type=interp_method, gamma=10, minimum_neighbors=1, hres=resolution,
                                       search_radius=8, boundary_coords={'west': X.min(), 'south': Y.min(),
                                                                          'east': X.max(), 'north': Y.max()})
        if i == 0:
            results_Dk_common_grid_gathered = Z
        else:
           results_Dk_common_grid_gathered[~ np.isnan(Z)] = Z[~ np.isnan(Z)]
        results_Dk_common_grid[i] = np.copy(Z)
    return X, Y, results_Dk_common_grid_gathered, results_Dk_common_grid


# execution options
create_merged_tif = True
working_dir = Path('/home/florent/Projects/Palavas-les-flots/Surfreef_project/results/')
date = '20230208'
hour = '16h'
# date = '20230422'
# hour = '07h'

# vertical reference of output bathymetry elevation's tif file
vertical_ref = 'IGN69'#'WL' or 'IGN69'

# specify for each date the water level relative to IGN69
if date == '20230208':
    WL_ref_IGN69 = 0.112
if date == '20230422':
    WL_ref_IGN69 = 0.245

# vertical_shift
if vertical_ref == 'WL':
    vertical_shift_Dk = 0.0
elif vertical_ref == 'IGN69':
    vertical_shift_Dk = - WL_ref_IGN69

# configuration corresponding to given results
fieldsite = 'wavecams_palavas_stpierre'
cam_names = ['St_Pierre_3', 'St_Pierre_2']

# georefs rotated
georefs_rotated = {}
for cam_name in cam_names:
    georefs_rotated[cam_name] = f'/home/florent/Projects/Palavas-les-flots/{cam_name}/info/georef_local_rotated/' \
                                f'georef.json'

# resolution of input projected images
proj_imgs_res = 1.0

# bathy grid resolutions
bathy_grid_resolutions = [8]
calcdmd = 'standard' # standard or robust

tif_files = []

for cam_name in cam_names:
    print(cam_name)
    # georef rotated
    georef_rotated = georefs_rotated[cam_name]

    # Reading Geo System
    Grid_Coordinate_System, UTM_zone, Grid_Coordinate_System_Offset, Off_set, Altitude_Datum_Name, \
        Rotated_system, Rotation_angle = load_rectification_geo_system_2(georef_rotated)


    # for bathy_grid_resolution in bathy_grid_resolutions:
    for bathy_grid_resolution in bathy_grid_resolutions:
        # load results
        output_dir = Path(str(working_dir), f'{fieldsite}/{cam_name}/{date}/{hour}/')
        try:
            f_results = glob(str(output_dir) + f'/results_grid_res_{bathy_grid_resolution}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
        except IndexError:
            print('pouet')
            continue
        results = np.load(f_results)
        basename = Path(f_results).stem

        grid_x_rot = results['grid_X'] - Grid_Coordinate_System_Offset[0]
        grid_y_rot = results['grid_Y'] - Grid_Coordinate_System_Offset[1]

    # dst file format, driver
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)

    # metadata
    metadata = driver.GetMetadata()

    # load depth results
    z = results['Dk'][-1]

    # mask boundaries as results are most often dubious in these areas
    z[0, :] = np.nan
    z[-1, :] = np.nan
    z[:, 0] = np.nan
    z[:, -1] = np.nan
    mask = np.isnan(z)
    mask = binary_dilation(mask)
    z[np.isnan(mask)] = np.nan

    # derotating z
    z = np.flipud(z)
    derot_z = rotateAndScale(z, scaleFactor=1, degreesCCW=-np.rad2deg(-Rotation_angle))

    # apply mask of 0 values
    derot_z[derot_z == 0] = np.nan

    # transform water depth to an elevation relative to vertical reference
    derot_z = - (derot_z + vertical_shift_Dk)

    # derotating grid bounding box coordinates
    x_min = grid_x_rot.min()
    x_max = grid_x_rot.max()
    y_min = grid_y_rot.min()
    y_max = grid_y_rot.max()
    xx = np.array([x_min, x_min, x_max, x_max])
    yy = np.array([y_min, y_max, y_min, y_max])
    grid_bbox = np.array([xx, yy]).transpose()
    rotated_grid_bbox = rotate_vector(grid_bbox, -Rotation_angle)  # negative angle to come back
    xx = np.array([np.nanmin(rotated_grid_bbox[:, 0]), np.nanmax(rotated_grid_bbox[:, 0])])
    yy = np.array([np.nanmin(rotated_grid_bbox[:, 1]), np.nanmax(rotated_grid_bbox[:, 1])])

    # Applying off_set
    xx = Grid_Coordinate_System_Offset[0] + xx.transpose()
    yy = Grid_Coordinate_System_Offset[1] + yy.transpose()

    # initialize spatial reference system
    srs = osr.SpatialReference()

    # checking hemisphere
    if hemisphere_UTM_checker(UTM_zone) == "southern":
        hemisphere_boolean = 0
    else:
        hemisphere_boolean = 1

    # setting projection
    srs.SetUTM(int(UTM_zone[:-1]), hemisphere_boolean)

    # output file format, driver
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)

    # metadata
    metadata = driver.GetMetadata()
    geotiff_img_file = f_results.replace('.npz', '.tif')
    tif_files.append(geotiff_img_file)

    dst_ds = driver.Create(geotiff_img_file, xsize=derot_z.shape[1], ysize=derot_z.shape[0], bands=1,
                           eType=gdal.GDT_Float32)

    dst_ds.SetGeoTransform([np.nanmin(xx.min()), (xx.max() - xx.min()) / derot_z.shape[1], 0,
                            np.nanmax(yy.max()), 0, (yy.min() - yy.max()) / derot_z.shape[0]])
    dst_ds.SetProjection(srs.ExportToWkt())

    # remove nan values
    # derot_z[np.isnan(derot_z)] = 0
    raster = derot_z
    dst_ds.GetRasterBand(1).WriteArray(raster)

    # Once we're done, close properly the dataset
    dst_ds = None


# creation of merged data in a tif file
if create_merged_tif:
    str_cam_names = '_'.join(cam_names)
    output_dir_merged = Path(working_dir, f'{fieldsite}/{str_cam_names}/{date}/{hour}/')
    output_dir_merged.mkdir(parents=True, exist_ok=True)

    # merge results
    interp_method = 'linear' # barnes, linear, cubic
    X, Y, results_Dk_common_grid_gathered, results_Dk_common_grid = interpolate_results_on_same_grid(tif_files,
                                                                                                     interp_method)

    # save merged result in a tif file
    srs = osr.SpatialReference()
    # setting projection
    srs.SetUTM(int(UTM_zone[:-1]), hemisphere_boolean)
    fileformat = "GTiff"
    driver = gdal.GetDriverByName(fileformat)
    metadata = driver.GetMetadata()

    geotiff_img_file = str(output_dir_merged.joinpath(f'merging_{str_cam_names}_{date}_{hour}_interp_{interp_method}.tif'))
    print(geotiff_img_file)
    dst_ds = driver.Create(geotiff_img_file, xsize=X.shape[1], ysize=Y.shape[0], bands=1, eType=gdal.GDT_Float32)
    dst_ds.SetGeoTransform([np.nanmin(X.min()), (X.max() - X.min()) / X.shape[1], 0,
                            np.nanmax(Y.max()), 0, (Y.min() - Y.max()) / X.shape[0]])
    dst_ds.SetProjection(srs.ExportToWkt())
    raster = np.flipud(results_Dk_common_grid_gathered)
    dst_ds.GetRasterBand(1).WriteArray(raster)

    # Once we're done, close properly the dataset
    dst_ds = None




































