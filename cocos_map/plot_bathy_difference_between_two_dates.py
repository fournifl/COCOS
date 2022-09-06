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



def plot_gathered_diff_results(cpu_speed, X, Y, results_Dk_common_grid, depth_lims, diff_depth_lims,
                          output_dir_plot, emprise, k, cam_name_central, vertical_ref, WL_ref_IGN69, resolutions,
                          resolution, key_dates):
    fig, ax = plt.subplots(2, 2, figsize=(25, 15), frameon=False)

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
    dates = f'{key_dates[1]}_vs_{key_dates[0]}'
    fig.delaxes(ax[1, 1])
    plt.tight_layout()
    fig.savefig(Path(output_dir_plot).joinpath(f'diff_bathy_{dates}_cpu_speed_{cpu_speed}_k_{k}_res_{resolution}' + '.jpg'))



# execution option
k = -1

vertical_ref = 'IGN69'#'WL' or 'IGN69'
# fieldsite = 'wavecams_palavas_cristal'
# cam_names = ['cristal_1', 'cristal_2', 'cristal_3']
# cam_name_central = 'cristal_1'
fieldsite = 'wavecams_palavas_stpierre'
cam_name_central = 'st_pierre_3'
cam_names = ['st_pierre_1', 'st_pierre_2', 'st_pierre_3']
# cpu_speeds = ['fast', 'normal', 'slow', 'accurate', 'exact'] #'fast','normal','slow', 'accurate', 'exact'
cpu_speeds = ['accurate']
calcdmd = 'standard' # standard or robust

# dates and hours of respective computed bathymetries
# dates = ['20220314', '20220314']
# hours = ['07h', '08h']
dates = ['20220314', '20220323']
hours = ['08h', '15h']
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
    diff_depth_lims = [-1.0, 1.0]

    output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/gathered/comparison_between_2_dates/'
    Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
    plot_gathered_diff_results(cpu_speed, X, Y, results_Dk_common_grid, depth_lims, diff_depth_lims, output_dir_plot,
                               emprise_plot, k, cam_name_central, vertical_ref, WL_ref_IGN69, resolutions, resolution,
                               key_dates)
