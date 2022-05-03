"""
Created on Tue Mar 29 13:46:40 2022

@author: ffournier
"""

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import Normalize
from copy import copy
from pathlib import Path
from glob import glob
import numpy as np
import pickle
import pdb
import collections


def compute_gathered_grid(results):
    resolutions = {}
    for i, cam_name in enumerate(results.keys()):
        xmin_tmp = np.min(results[cam_name]['grid_X'])
        xmax_tmp = np.max(results[cam_name]['grid_X'])
        ymin_tmp = np.min(results[cam_name]['grid_Y'])
        ymax_tmp = np.max(results[cam_name]['grid_Y'])
        resolution_tmp = results[cam_name]['grid_X'][0, 1] - results[cam_name]['grid_X'][0, 0]
        resolutions[cam_name] = resolution_tmp
        print(f'cam_name: {cam_name}')
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
            if resolution < resolution_tmp:
                resolution = resolution_tmp
    x = np.arange(xmin, xmax, resolution)
    y = np.arange(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y, resolution, resolutions


def interpolate_results_on_same_grid(results, X, Y, k, cam_names_ordered):
    results_Dk_common_grid = {}
    # add central cam data on gathered grid last, as pixel footprint is better compared to lateral cameras
    for i, cam_name in enumerate(cam_names_ordered):
        points = (np.vstack([results[cam_name]['grid_X'].flatten(), results[cam_name]['grid_Y'].flatten()])).T
        values = results[cam_name]['Dk'][k, :, :].flatten()
        Z = griddata(points, values, (X, Y))
        if i == 0:
            results_Dk_common_grid_gathered = Z
        else:
           results_Dk_common_grid_gathered[~ np.isnan(Z)] = Z[~ np.isnan(Z)]
        results_Dk_common_grid[cam_name] = np.copy(Z)
    return results_Dk_common_grid_gathered, results_Dk_common_grid


def interpolate_lidar_data_on_gathered_grid(lidar_data, X, Y, mask):
    Z = griddata((np.ravel(litto3d['Xi']), np.ravel(litto3d['Yi'])), np.ravel(litto3d['zi']), (X, Y), method='linear')
    Z[mask] = np.nan
    return Z


def plot_gathered_diff_results(cpu_speed, X, Y, results_Dk_gathered, results_Dk_common_grid, depth_lims, diff_depth_lims,
                          output_dir_plot, emprise, k, cam_name_central, vertical_ref, WL_ref_IGN69, resolutions,
                          resolution, key_dates):
    fig, ax = plt.subplots(2, 2, figsize=(15, 15), frameon=False)

    # make sure to order cam_names so as to have central camera plotted first
    cam_names = list(results_Dk_common_grid[key_dates[0]].keys())
    i_central = np.where(np.array(cam_names) == cam_name_central)[0][0]
    alphas = np.ones(len(cam_names)) * 0.5
    alphas[i_central] = 1.0

    # vertical_shift
    if vertical_ref == 'WL':
        vertical_shift_Dk = 0.0
    elif vertical_ref == 'IGN69':
        vertical_shift_Dk = - WL_ref_IGN69

    for i_date in range(len(key_dates)):
        # plot bathy from central cam first
        im0 = ax[i_date, 0].pcolor(X, Y, results_Dk_common_grid[key_dates[i_date]][cam_name_central] + vertical_shift_Dk,
                           cmap='jet_r', vmin=depth_lims[0], vmax=depth_lims[1], alpha=alphas[i_central])

        for i, cam_name in enumerate(cam_names):
            # text resolution of each camera
            ax[i_date, 0].text(np.max(X) - 250, np.max(Y) - 100 - i * 30, f'resolution {cam_name}: {resolutions[cam_name]} m')
            if cam_name != cam_name_central:
                # plot bathy from each camera
                im0 = ax[i_date, 0].pcolor(X, Y, results_Dk_common_grid[key_dates[i_date]][cam_name] + vertical_shift_Dk,
                                      cmap='jet_r', vmin=depth_lims[0], vmax=depth_lims[1], alpha=alphas[i])
        fig.colorbar(im0, ax=ax[i_date, 0])
        ax[i_date, 0].axis('equal')
        ax[i_date, 0].set_title(f'depth for each camera [m] on {key_dates[i_date]}')
        ax[i_date, 0].set_xlim([emprise[0], emprise[1]])
        ax[i_date, 0].set_ylim([emprise[2], emprise[3]])
    # plot bathy differences
    # plot bathy diff for central cam first
    diff = results_Dk_common_grid[key_dates[1]][cam_name_central] - results_Dk_common_grid[key_dates[0]][
        cam_name_central]
    im0 = ax[0, 1].pcolor(X, Y, diff, cmap='Spectral', vmin=diff_depth_lims[0], vmax=diff_depth_lims[1],
                          alpha=alphas[i_central])
    for i, cam_name in enumerate(cam_names):
        if cam_name != cam_name_central:
            diff = results_Dk_common_grid[key_dates[1]][cam_name] - results_Dk_common_grid[key_dates[0]][cam_name]
            im1 = ax[0, 1].pcolor(X, Y, diff, cmap='Spectral', vmin=diff_depth_lims[0], vmax=diff_depth_lims[1],
                                  alpha=alphas[i])
    fig.colorbar(im1, ax=ax[0, 1])
    ax[0, 1].axis('equal')
    ax[0, 1].set_title(f'diff depth [m], {key_dates[1]} vs {key_dates[0]}')
    ax[0, 1].set_xlim([emprise[0], emprise[1]])
    ax[0, 1].set_ylim([emprise[2], emprise[3]])
    plt.show()


    fig.colorbar(im0, ax=ax[0])
    # fig.colorbar(im1, ax=ax[1])
    #
    # ax[0].axis('equal')
    # ax[0].set_title('depth for each camera [m]')
    # ax[0].set_ylabel('y [m]')
    # ax[0].set_xlabel('x [m]')
    # ax[0].set_xlim([emprise[0], emprise[1]])
    # ax[0].set_ylim([emprise[2], emprise[3]])
    # ax[1].set_title('depth gathered [m]')
    # ax[1].axis('equal')
    # ax[1].set_title('depth gathered [m]')
    # ax[1].set_ylabel('y [m]')
    # ax[1].set_xlabel('x [m]')
    # ax[1].set_xlim([emprise[0], emprise[1]])
    # ax[1].set_ylim([emprise[2], emprise[3]])
    #
    # fig.savefig(Path(output_dir_plot).joinpath(
    #     f'results_gathered_k_{k}_cpu_speed_{cpu_speed}_vertical_ref_{vertical_ref}_res_{resolution}' + '.jpg'))
    # plt.close('all')


# execution option
k = -1

vertical_ref = 'IGN69'#'WL' or 'IGN69'
fieldsite = 'wavecams_palavas_cristal'
cam_names = ['cristal_1', 'cristal_2', 'cristal_3']
cam_name_central = 'cristal_1'
# cpu_speeds = ['fast', 'normal', 'slow', 'accurate', 'exact'] #'fast','normal','slow', 'accurate', 'exact'
cpu_speeds = ['accurate']
calcdmd = 'standard' # standard or robust

# dates and hours of respective computed bathymetries
dates = ['20220314', '20220314']
hours = ['07h', '08h']
results = {}

# cam_names_ordered so as to have cam_central in last position
cam_names_ordered = copy(cam_names)
cam_names_ordered.append(cam_names_ordered.pop(cam_names_ordered.index(cam_name_central)))

# load results
for cpu_speed in cpu_speeds:
    key_dates = []
    results_Dk_common_grid_gathered = {}
    results_Dk_common_grid = {}
    for i in range(len(dates)):
        key_date = '_'.join([dates[i], hours[i]])
        key_dates.append(key_date)
        # define WL_ref_IGN69
        if dates[i] == '20220314':
            WL_ref_IGN69 = 0.60 - 0.307
        elif dates[i] == '20220323':
            WL_ref_IGN69 = 0.19 - 0.307
        results[key_date] = {}
        for cam_name in cam_names:
            output_dir = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{dates[i]}/{hours[i]}/'
            f_results = glob(output_dir + f'/results_CPU_speed_{cpu_speed}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
            results[key_date][cam_name] = np.load(f_results)
            # merge results
            X, Y, resolution, resolutions = compute_gathered_grid(results[key_date])
        # merging
        results_Dk_common_grid_gathered[key_date], results_Dk_common_grid[key_date] = \
            interpolate_results_on_same_grid(results[key_date], X, Y, k, cam_names_ordered)

    # affichage
    if fieldsite == 'wavecams_palavas_cristal':
        xmin = 575680
        xmax = 576410
        ymin = 4819600
        ymax = 4820200
    if fieldsite == 'wavecams_palavas_stpierre':
        xmin = 574300
        xmax = 575120
        ymin = 4818900
        ymax = 4819600
    emprise_plot = [xmin, xmax, ymin, ymax]

    depth_lims = [0, 6]
    diff_depth_lims = [-0.5, 0.5]

    output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/gathered/comparison_between_2_dates/' \
                      f'{key_dates[0]}_vs_{key_dates[1]}/'
    Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
    plot_gathered_diff_results(cpu_speed, X, Y, results_Dk_common_grid_gathered, results_Dk_common_grid, depth_lims,
                               diff_depth_lims, output_dir_plot, emprise_plot, k, cam_name_central, vertical_ref,
                               WL_ref_IGN69, resolutions, resolution, key_dates)


# for cpu_speed in cpu_speeds:
#     print(f'cpu_speed: {cpu_speed}')
#     # load results
#     results = {}
#     for cam_name in cam_names:
#         output_dir = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{date}/{hour}/'
#         try:
#             f_results = glob(output_dir + f'/results_CPU_speed_{cpu_speed}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
#             results[cam_name] = np.load(f_results)
#             results_exist = True
#         except IndexError:
#             results_exist = False
#             continue
#     if not results_exist:
#         print(f'no result for cpu_speed {cpu_speed}')
#         continue
#     basename = Path(f_results).stem
#
#     # merge results
#     X, Y, resolution, resolutions = compute_gathered_grid(results)
#     # cam_names_ordered so as to have cam_central in last position
#     cam_names_ordered = copy(cam_names)
#     cam_names_ordered.append(cam_names_ordered.pop(cam_names_ordered.index(cam_name_central)))
#     # merging
#     results_Dk_common_grid_gathered, results_Dk_common_grid = interpolate_results_on_same_grid(results, X, Y, k, cam_names_ordered)
#
#     # check if ground truth exists:
#     ground_truth_exists = False
#     # ground_truth_exists = type(results[cam_names[-1]]['Dgt']).__module__ == np.__name__
#
#     # define WL_ref_IGN69
#     if date == '20220314':
#         WL_ref_IGN69 = 0.60 - 0.307
#     elif date == '20220323':
#         WL_ref_IGN69 = 0.19 - 0.307
#
#     # interpolate lidar data on gathered grid
#     mask = np.isnan(results_Dk_common_grid_gathered)
#     if ground_truth_exists:
#         if  fieldsite == 'wavecams_palavas_cristal':
#             f_litto3d = '/home/florent/Projects/Palavas-les-flots/Bathy/litto3d/cristal/litto3d_Palavas_epsg_32631_775_776_6271.pk'
#         elif fieldsite == 'wavecams_palavas_stpierre':
#             f_litto3d = '/home/florent/Projects/Palavas-les-flots/Bathy/litto3d/st_pierre/litto3d_Palavas_st_pierre_epsg_32631_774_775_6270.pk'
#         litto3d = pickle.load(open(f_litto3d, 'rb'))
#         z_lidar = interpolate_lidar_data_on_gathered_grid(litto3d, X, Y, mask)
#
#         # convert lidar topo to depth and change vertical ref if necessary
#         if vertical_ref == 'WL':
#             z_lidar = - z_lidar + WL_ref_IGN69
#         elif vertical_ref == 'IGN69':
#             z_lidar = - z_lidar
#
#     # plot bathy results only
#     output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/gathered/bathy_results_only/{date}/{hour}/'
#     Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
#     plot_gathered_results(cpu_speed, X, Y, results_Dk_common_grid_gathered, results_Dk_common_grid, depth_lims,
#                           output_dir_plot, emprise_plot, k, cam_name_central, vertical_ref, WL_ref_IGN69, resolutions,
#                           resolution)
