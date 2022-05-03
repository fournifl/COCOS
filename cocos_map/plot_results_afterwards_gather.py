"""
Created on Tue Mar 29 13:46:40 2022

@author: ffournier
"""

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from glob import glob
import numpy as np
import pickle
import pdb
import collections


def plot_bathy(results, output_dir_plot, basename, depth_lims, diff_depth_lims, ground_truth_exists=None):
    n_iter = results['t_iter'].size
    # results['Dgt'][results['Dgt'] > 35] = np.nan
    for k in range(n_iter):
        if ground_truth_exists:
            fig, ax = plt.subplots(1, 3, figsize=(22, 5))
            mask = np.isnan(results['Dk'][k])
            diff_depth_ground_truth = results['Dk'][k] - results['Dgt']

            im1 = ax[0].pcolor(results['grid_X'], results['grid_Y'], results['Dk'][k], cmap='jet_r')
            im2 = ax[1].pcolor(results['grid_X'], results['grid_Y'], np.ma.array(results['Dgt'], mask=mask), cmap='jet_r')
            im3 = ax[2].pcolor(results['grid_X'], results['grid_Y'], diff_depth_ground_truth, cmap='Spectral')

            contours = ax[2].contour(results['grid_X'], results['grid_Y'], diff_depth_ground_truth,
                                     levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',), linewidths=(1.5,))
            ax[2].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])
            fig.colorbar(im3, ax=ax[2])
            im1.set_clim([depth_lims[0], depth_lims[1]])
            im2.set_clim([depth_lims[0], depth_lims[1]])
            im3.set_clim([diff_depth_lims[0], diff_depth_lims[1]])

            ax[0].axis('equal')
            ax[0].set_title('depth [m]')
            ax[0].set_ylabel('y [m]')
            ax[0].set_xlabel('x [m]')
            ax[1].axis('equal')
            ax[1].set_title('ground truth depth [m]')
            ax[1].set_ylabel('y [m]')
            ax[1].set_xlabel('x [m]')
            ax[2].axis('equal')
            ax[2].set_title('diff depth - ground truth depth [m]')
            ax[2].set_ylabel('y [m]')
            ax[2].set_xlabel('x [m]')
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' %k + '.jpg'))
            plt.close('all')
        else:
            fig, ax = plt.subplots(figsize=(20, 10))
            im1 = ax.pcolor(results['grid_X'], results['grid_Y'], results['Dk'][k], cmap='jet_r')
            im1.set_clim([0, 6])
            fig.colorbar(im1, ax=ax)
            ax.axis('equal')
            ax.set_title('depth [m]')
            ax.set_ylabel('y [m]')
            ax.set_xlabel('x [m]')
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' % k + '.jpg'))
            plt.close('all')


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


def interpolate_results_on_same_grid(results, X, Y, k):
    results_Dk_common_grid = {}
    for i, cam_name in enumerate(results.keys()):
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


def plot_gathered_results_save(X, Y, results_Dk_gathered, lidar_data, results, cam_names, depth_lims, diff_depth_lims,
                          output_dir_plot, emprise, k):
    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    diff_depth_ground_truth = results_Dk_gathered - lidar_data
    im1 = ax[0].pcolor(X, Y, results_Dk_gathered, cmap='jet_r')
    im2 = ax[1].pcolor(X, Y, lidar_data, cmap='jet_r')
    im3 = ax[2].pcolor(X, Y, diff_depth_ground_truth, cmap='Spectral')
    contours = ax[2].contour(X, Y, diff_depth_ground_truth, levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',),
                             linewidths=(1.5,))
    ax[2].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])
    im1.set_clim([depth_lims[0], depth_lims[1]])
    im2.set_clim([depth_lims[0], depth_lims[1]])
    im3.set_clim([diff_depth_lims[0], diff_depth_lims[1]])

    ax[0].set_title('depth gathered [m]')
    ax[0].axis('equal')
    ax[0].set_title('depth gathered [m]')
    ax[0].set_ylabel('y [m]')
    ax[0].set_xlabel('x [m]')
    ax[0].set_xlim([emprise[0], emprise[1]])
    ax[0].set_ylim([emprise[2], emprise[3]])
    ax[1].axis('equal')
    ax[1].set_title('ground truth depth [m]')
    ax[1].set_ylabel('y [m]')
    ax[1].set_xlabel('x [m]')
    ax[1].set_xlim([emprise[0], emprise[1]])
    ax[1].set_ylim([emprise[2], emprise[3]])
    ax[2].axis('equal')
    ax[2].set_title('diff depth - ground truth depth [m]')
    ax[2].set_ylabel('y [m]')
    ax[2].set_xlabel('x [m]')
    ax[2].set_xlim([emprise[0], emprise[1]])
    ax[2].set_ylim([emprise[2], emprise[3]])
    fig.savefig(Path(output_dir_plot).joinpath(basename + f'_gathered_k_{k}' + '.jpg'))
    plt.close('all')


def plot_gathered_results_ground_truth_exists(cpu_speed, X, Y, results_Dk_gathered, lidar_data, results_Dk_common_grid, depth_lims,
                          diff_depth_lims, output_dir_plot, emprise, k, cam_name_central, vertical_ref, WL_ref_IGN69,
                          resolutions, resolution):
    fig, ax = plt.subplots(2, 2, figsize=(22, 8), frameon=False)
    diff_depth_ground_truth = results_Dk_gathered - lidar_data
    # make sure to order cam_names so as to have central camera plotted first
    cam_names = list(results_Dk_common_grid.keys())
    print(cam_names)
    i_central = np.where(np.array(cam_names) == cam_name_central)[0][0]
    # cam_names.pop(i_central)
    # cam_names.insert(0, cam_name_central)
    alphas = np.ones(len(cam_names)) * 0.5
    alphas[i_central] = 1.0
    print(cam_names)
    print(alphas)

    # vertical_shift
    if vertical_ref == 'WL':
        vertical_shift_Dk = 0.0
    elif vertical_ref == 'IGN69':
        vertical_shift_Dk = - WL_ref_IGN69

    # plot bathy from central cam first
    im0 = ax[0, 0].pcolor(X, Y, results_Dk_common_grid[cam_name_central] + vertical_shift_Dk, cmap='jet_r',
                          vmin=depth_lims[0], vmax=depth_lims[1], alpha=alphas[i_central])
    for i, cam_name in enumerate(cam_names):
        # text resolution of each camera
        ax[0, 0].text(np.max(X) - 100, np.max(Y) - 100 - i * 30, f'resolution {cam_name}: {resolutions[cam_name]} m')
        if cam_name != cam_name_central:
            # plot bathy from each camera
            im0 = ax[0, 0].pcolor(X, Y, results_Dk_common_grid[cam_name]  + vertical_shift_Dk, cmap='jet_r',
                                  vmin=depth_lims[0], vmax=depth_lims[1], alpha=alphas[i])
    im1 = ax[1, 0].pcolor(X, Y, results_Dk_gathered +  + vertical_shift_Dk, cmap='jet_r', vmin=depth_lims[0],
                          vmax=depth_lims[1])
    im2 = ax[1, 1].pcolor(X, Y, lidar_data, cmap='jet_r', vmin=depth_lims[0], vmax=depth_lims[1])
    im3 = ax[0, 1].pcolor(X, Y, diff_depth_ground_truth, cmap='Spectral', vmin=diff_depth_lims[0], vmax=diff_depth_lims[1])
    contours = ax[0, 1].contour(X, Y, diff_depth_ground_truth, levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',),
                             linewidths=(1.5,))
    ax[0, 1].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
    fig.colorbar(im0, ax=ax[0, 0])
    fig.colorbar(im1, ax=ax[1, 0])
    fig.colorbar(im2, ax=ax[1, 1])
    fig.colorbar(im3, ax=ax[0, 1])

    ax[1, 0].set_title('depth gathered [m]')
    ax[1, 0].axis('equal')
    ax[1, 0].set_title('depth gathered [m]')
    ax[1, 0].set_ylabel('y [m]')
    ax[1, 0].set_xlabel('x [m]')
    ax[1, 0].set_xlim([emprise[0], emprise[1]])
    ax[1, 0].set_ylim([emprise[2], emprise[3]])
    ax[1, 1].axis('equal')
    ax[1, 1].set_title('ground truth depth [m]')
    ax[1, 1].set_ylabel('y [m]')
    ax[1, 1].set_xlabel('x [m]')
    ax[1, 1].set_xlim([emprise[0], emprise[1]])
    ax[1, 1].set_ylim([emprise[2], emprise[3]])
    ax[0, 1].axis('equal')
    ax[0, 1].set_title('diff depth - ground truth depth [m]')
    ax[0, 1].set_ylabel('y [m]')
    ax[0, 1].set_xlabel('x [m]')
    ax[0, 1].set_xlim([emprise[0], emprise[1]])
    ax[0, 1].set_ylim([emprise[2], emprise[3]])
    ax[0, 0].axis('equal')
    ax[0, 0].set_title('depth for each camera [m]')
    ax[0, 0].set_ylabel('y [m]')
    ax[0, 0].set_xlabel('x [m]')
    ax[0, 0].set_xlim([emprise[0], emprise[1]])
    ax[0, 0].set_ylim([emprise[2], emprise[3]])
    fig.savefig(Path(output_dir_plot).joinpath(f'results_gathered_k_{k}_cpu_speed_{cpu_speed}_vertical_ref_{vertical_ref}_res_{resolution}' + '.jpg'))
    plt.close('all')


def plot_gathered_results(cpu_speed, X, Y, results_Dk_gathered, results_Dk_common_grid, depth_lims,
                          output_dir_plot, emprise, k, cam_name_central, vertical_ref, WL_ref_IGN69, resolutions,
                          resolution):
    fig, ax = plt.subplots(2, 1, figsize=(15, 15), frameon=False)
    # make sure to order cam_names so as to have central camera plotted first
    cam_names = list(results_Dk_common_grid.keys())
    i_central = np.where(np.array(cam_names) == cam_name_central)[0][0]
    alphas = np.ones(len(cam_names)) * 0.5
    alphas[i_central] = 1.0

    # plot bathy from central cam first
    # vertical_shift
    if vertical_ref == 'WL':
        vertical_shift_Dk = 0.0
    elif vertical_ref == 'IGN69':
        vertical_shift_Dk = - WL_ref_IGN69

    im0 = ax[0].pcolor(X, Y, results_Dk_common_grid[cam_name_central] + vertical_shift_Dk, cmap='jet_r',
                    vmin=depth_lims[0], vmax=depth_lims[1], alpha=alphas[i_central])

    # text resolution of each camera
    for i, cam_name in enumerate(cam_names):
        ax[0].text(np.max(X) - 100, np.max(Y) - 100 - i * 30, f'resolution {cam_name}: {resolutions[cam_name]} m')
        if cam_name != cam_name_central:
            # plot bathy from each camera
            im0 = ax[0].pcolor(X, Y, results_Dk_common_grid[cam_name] + vertical_shift_Dk, cmap='jet_r',
                                  vmin=depth_lims[0], vmax=depth_lims[1], alpha=alphas[i])
    im1 = ax[1].pcolor(X, Y, results_Dk_gathered + vertical_shift_Dk, cmap='jet_r', vmin=depth_lims[0],
                          vmax=depth_lims[1])

    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])

    ax[0].axis('equal')
    ax[0].set_title('depth for each camera [m]')
    ax[0].set_ylabel('y [m]')
    ax[0].set_xlabel('x [m]')
    ax[0].set_xlim([emprise[0], emprise[1]])
    ax[0].set_ylim([emprise[2], emprise[3]])
    ax[1].set_title('depth gathered [m]')
    ax[1].axis('equal')
    ax[1].set_title('depth gathered [m]')
    ax[1].set_ylabel('y [m]')
    ax[1].set_xlabel('x [m]')
    ax[1].set_xlim([emprise[0], emprise[1]])
    ax[1].set_ylim([emprise[2], emprise[3]])

    fig.savefig(Path(output_dir_plot).joinpath(
        f'results_gathered_k_{k}_cpu_speed_{cpu_speed}_vertical_ref_{vertical_ref}_res_{resolution}' + '.jpg'))
    plt.close('all')



# execution option
k = -1
# k = 16 + 1 # kth calculated bathy, that has to be plotted

# configuration corresponding to given results
# date = '20220314'
# hour = '08h'
date = '20220323'
hour = '15h'
vertical_ref = 'IGN69'#'WL' or 'IGN69'

# fieldsite = 'wavecams_palavas_cristal'
# cam_names = ['cristal_1', 'cristal_2', 'cristal_3']
# cam_name_central = 'cristal_1'
fieldsite = 'wavecams_palavas_stpierre'
cam_name_central = 'st_pierre_3'
cam_names = ['st_pierre_1', 'st_pierre_2', 'st_pierre_3']

cpu_speeds = ['fast', 'normal', 'slow', 'accurate', 'exact'] #'fast','normal','slow', 'accurate', 'exact'
for cpu_speed in cpu_speeds:
    print(f'cpu_speed: {cpu_speed}')
    calcdmd = 'standard' # standard or robust

    # load results
    results = {}
    for cam_name in cam_names:
        output_dir = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{date}/{hour}/'
        try:
            f_results = glob(output_dir + f'/results_CPU_speed_{cpu_speed}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
            results[cam_name] = np.load(f_results)
            results_exist = True
        except IndexError:
            results_exist = False
            continue
    if not results_exist:
        print(f'no result for cpu_speed {cpu_speed}')
        continue
    basename = Path(f_results).stem

    # merge results
    X, Y, resolution, resolutions = compute_gathered_grid(results)
    results_Dk_common_grid_gathered, results_Dk_common_grid = interpolate_results_on_same_grid(results, X, Y, k)

    # check if ground truth exists:
    ground_truth_exists = False
    # ground_truth_exists = type(results[cam_names[-1]]['Dgt']).__module__ == np.__name__

    # define WL_ref_IGN69
    if date == '20220314':
        WL_ref_IGN69 = 0.60 - 0.307
    elif date == '20220323':
        WL_ref_IGN69 = 0.19 - 0.307

    # interpolate lidar data on gathered grid
    mask = np.isnan(results_Dk_common_grid_gathered)
    if ground_truth_exists:
        if  fieldsite == 'wavecams_palavas_cristal':
            f_litto3d = '/home/florent/Projects/Palavas-les-flots/Bathy/litto3d/cristal/litto3d_Palavas_epsg_32631_775_776_6271.pk'
        elif fieldsite == 'wavecams_palavas_stpierre':
            f_litto3d = '/home/florent/Projects/Palavas-les-flots/Bathy/litto3d/st_pierre/litto3d_Palavas_st_pierre_epsg_32631_774_775_6270.pk'
        litto3d = pickle.load(open(f_litto3d, 'rb'))
        z_lidar = interpolate_lidar_data_on_gathered_grid(litto3d, X, Y, mask)

        # convert lidar topo to depth and change vertical ref if necessary
        if vertical_ref == 'WL':
            z_lidar = - z_lidar + WL_ref_IGN69
        elif vertical_ref == 'IGN69':
            z_lidar = - z_lidar

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
    diff_depth_lims = [-1.5, 1.5]
    # plot with comparison with ground truth
    if ground_truth_exists:
        output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/gathered/comparison_with_ground_truth/' \
                          f'{date}/{hour}/'
        Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
        plot_gathered_results_ground_truth_exists(cpu_speed, X, Y, results_Dk_common_grid_gathered, z_lidar,
                                                  results_Dk_common_grid, depth_lims, diff_depth_lims, output_dir_plot,
                                                  emprise_plot, k, cam_name_central, vertical_ref, WL_ref_IGN69,
                                                  resolutions, resolution)
    # plot bathy results only
    output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/gathered/bathy_results_only/{date}/{hour}/'
    Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
    plot_gathered_results(cpu_speed, X, Y, results_Dk_common_grid_gathered, results_Dk_common_grid, depth_lims,
                          output_dir_plot, emprise_plot, k, cam_name_central, vertical_ref, WL_ref_IGN69, resolutions,
                          resolution)
