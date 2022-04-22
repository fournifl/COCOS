"""
Created on Tue Mar 29 13:46:40 2022

@author: ffournier
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
from glob import glob
import numpy as np
import pdb
import collections


def plot_bathy(results, output_dir_plot, basename, depth_lims, diff_depth_lims, emprise, ground_truth_exists=None):
    n_iter = results['t_iter'].size
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
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' %k + '.jpg'))
            plt.close('all')
        else:
            fig, ax = plt.subplots(figsize=(20, 10))
            im1 = ax.pcolor(results['grid_X'], results['grid_Y'], results['Dk'][k], cmap='jet_r')
            im1.set_clim([depth_lims[0], depth_lims[1]])
            fig.colorbar(im1, ax=ax)
            ax.axis('equal')
            ax.set_title('depth [m]')
            ax.set_ylabel('y [m]')
            ax.set_xlabel('x [m]')
            ax.set_xlim([emprise[0], emprise[1]])
            ax.set_ylim([emprise[2], emprise[3]])
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' % k + '.jpg'))
            plt.close('all')


def plot_all_diags(results, output_dir_plot, basename, depth_lims, diff_depth_lims, kalman_error_lims, freqlims):
    depth_rmse = np.array([])
    depth_debiased_rmse = np.array([])
    depth_bias = np.array([])
    depth_mae = np.array([])
    iteration_number = np.array([])
    iteration_time = np.array([])
    n_iter = results['t_iter'].size
    color1 = 'tab:blue'
    color2 = 'tab:red'
    color3 = 'tab:green'
    color4 = 'tab:brown'
    color5 = 'tab:orange'

    for k in range(n_iter):
        diff_depth = results['Dk'][k] - results['Dgt']
        depth_bias = np.append(depth_bias, np.nanmean(diff_depth))
        depth_rmse = np.append(depth_rmse, np.sqrt(np.nanmean(depth_bias**2)))
        depth_mae = np.append(depth_mae, np.abs(np.nanmean(depth_bias)))
        diff_depth_debiased = diff_depth - np.nanmean(diff_depth)
        depth_debiased_rmse = np.append(depth_debiased_rmse, np.sqrt(np.nanmean(diff_depth_debiased ** 2)))
        iteration_number = np.append(iteration_number, k)
        iteration_time = np.append(iteration_time, results['t_shift_k'][k])

        fig, ax = plt.subplots(2, 5, figsize=(25, 12))

        # depth
        im0 = ax[0, 0].pcolor(results['grid_X'], results['grid_Y'], results['Dk'][k], cmap='jet_r')
        im0.set_clim(depth_lims)
        contours_0 = ax[0, 0].contour(results['grid_X'], results['grid_Y'], results['Dgt'], levels=[0.5, 2, 3.5, 5, 7.5, 10, 12.5, 15],
                               colors=('k',), linewidths=(1.5,))
        ax[0, 0].clabel(contours_0, contours_0.levels, fontsize=14, fmt='%.1f')  # , inline=True
        plt.colorbar(im0, ax=ax[0, 0])

        # diff depth
        im1 = ax[0, 1].pcolor(results['grid_X'], results['grid_Y'], diff_depth, cmap='Spectral')
        contours = ax[0, 1].contour(results['grid_X'], results['grid_Y'], diff_depth,
                                 levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',), linewidths=(1.5,))
        ax[0, 1].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
        im1.set_clim(diff_depth_lims)
        plt.colorbar(im1, ax=ax[0, 1])

        # Kalman error
        im2 = ax[0, 2].pcolor(results['grid_X'], results['grid_Y'], results['d_K_errors'][k, :, :], cmap='Spectral')
        plt.colorbar(im2, ax=ax[0, 2])
        im2.set_clim(kalman_error_lims)

        # c(T) (m/s)
        density = 2
        norm = Normalize(vmin=0, vmax=len(results['Cxy_omega']))
        numrows = len(results['grid_Rows_ctr'])
        numcols = len(results['grid_Cols_ctr'])
        skip = (
            slice(None, None, round(numrows / 30 * density)),
            slice(None, None, round(numcols / 30 * density)))

        for ii in range(len(results['Cxy_omega'])):
            cc = cm.jet(norm(ii), bytes=True)
            im3 = ax[0, 3].quiver(results['grid_X'][skip], results['grid_Y'][skip],
                                  np.flipud(np.reshape(results['Cxk'][k, 0, :, ii], (numrows, numcols), order='F')[skip]),
                                  np.flipud(np.reshape(results['Cyk'][k, 0, :, ii], (numrows, numcols), order='F')[skip]),
                                  color=(cc[0] / 255, cc[1] / 255, cc[2] / 255, cc[3] / 255))
        plt.colorbar(im3, ax=ax[0, 3])

        # U (m/s)
        im4 = ax[0, 4].streamplot(results['grid_X'][0, :], results['grid_Y'][:, 0], results['Uk'][k], results['Vk'][k],
                                  color=np.sqrt(results['Uk'][k] ** 2 + results['Vk'][k] ** 2), linewidth=1,
                                  cmap='jet', density=2, arrowstyle='->', arrowsize=1.5)
        plt.colorbar(im4.lines, ax=ax[0, 4])

        # cpu time per update
        im5 = ax[1, 0].stem(iteration_number, iteration_time)

        # stats mrmse, bias
        im6 = ax[1, 1].plot(iteration_number, depth_debiased_rmse, color=color1)
        ax6b = ax[1, 1].twinx()
        ax6b.plot(iteration_number, depth_bias, color=color2)

        # scatter plot depth cocos vs depth ground truth
        im7 = ax[1, 2].scatter(results['Dgt'][~np.isnan(results['Dgt'])],
                               results['Dk'][k][~np.isnan(results['Dgt'])], 1.5, color2)
        ax[1, 2].plot(depth_lims, depth_lims)

        # gc kernel
        im8 = ax[1, 3].imshow(results['gc_kernel'], cmap='jet', origin='lower')

        # spectrum (fft, dmd)
        fft_freqlims_ID = (results['fft_spectrum_k'][k][1] / (2 * np.pi) > freqlims[1]) & (
            results['fft_spectrum_k'][k][1] / (2 * np.pi) < freqlims[0])
        im9 = ax[1, 4].plot(results['fft_spectrum_k'][k][1][fft_freqlims_ID],
                            results['fft_spectrum_k'][k][0][fft_freqlims_ID], 's-', color=(0.4, 0.9, 0))
        try:
            im9b = ax[1, 4].plot(results['dmd_spectrum_k'][k][1], results['dmd_spectrum_k'][k][0], '*-', color=(1.0, 0.5, 0))
        except ValueError:
            print('Object arrays cannot be loaded when allow_pickle=False')

        ax[0, 0].axis('equal')
        ax[0, 0].set_title('depth [m]')
        ax[0, 0].set_ylabel('y [m]')
        ax[0, 0].set_xlabel('x [m]')
        ax[0, 1].axis('equal')
        ax[0, 1].set_title('diff depth - ground truth depth [m]')
        ax[0, 1].set_ylabel('y [m]')
        ax[0, 1].set_xlabel('x [m]')
        ax[0, 2].axis('equal')
        ax[0, 2].set_title('Kalman d-error [m]')
        ax[0, 2].set_ylabel('y [m]')
        ax[0, 2].set_xlabel('x [m]')
        ax[0, 3].axis('equal')
        ax[0, 3].set_title('c(T) [m/s]')
        ax[0, 3].set_ylabel('y [m]')
        ax[0, 3].set_xlabel('x [m]')
        ax[0, 4].axis('equal')
        ax[0, 4].set_title('U [m/s]')
        ax[0, 4].set_ylabel('y [m]')
        ax[0, 4].set_xlabel('x [m]')
        ax[1, 1].set_title('med.bias(red) and IQR(blue)')
        ax[1, 1].set_ylabel('IQR [m]')
        ax6b.set_ylabel('bias [m]')
        ax[1, 0].set_title('CPU time per update');
        ax[1, 0].set_xlabel('update [#]')
        ax[1, 0].set_ylabel('t-CPU [s]')
        ax[1, 2].set_ylim(depth_lims)
        ax[1, 2].set_xlim(depth_lims)
        ax[1, 2].set_title('Direct comp.')
        ax[1, 2].set_ylabel('d_inv [m]')
        ax[1, 2].set_xlabel('d_meas [m]')
        ax[1, 3].axis('equal')
        ax[1, 3].set_title('SSPC sample locs.')
        ax[1, 3].set_ylabel('gc_y [-]')
        ax[1, 3].set_xlabel('gc_x [-]')
        ax[1, 4].set_title('FFT(green) vs. DMD(orange) spectra');
        ax[1, 4].set_ylabel('A [norm. intensity]');
        ax[1, 4].set_xlabel('omega [rad/s]')

        fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' % k + '.jpg'))
        plt.close('all')


# execution options
plot_only_bathy = True
plot_all_results = False
date = '20220323'
hour = '15h'

# configuration corresponding to given results
# fieldsite = 'wavecams_palavas_cristal'
# cam_name = 'cristal_3'
# fieldsite = 'wavecams_palavas_cristal_merged'
fieldsite = 'wavecams_palavas_stpierre'
cam_name = 'st_pierre_3'
# fieldsite = 'wavecams_palavas_cristal_merged'
# cam_name = 'cristal_merged'
cpu_speeds = ['fast', 'normal', 'slow', 'accurate'] #'fast','normal','slow', 'accurate', 'exact'
calcdmd = 'standard' # standard or robust

for cpu_speed in cpu_speeds:
    # load results
    output_dir = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{date}/{hour}/'
    try:
        f_results = glob(output_dir + f'/results_CPU_speed_{cpu_speed}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
    except IndexError:
        continue

    results = np.load(f_results)
    basename = Path(f_results).stem

    # plot options
    print(f'xmin = {int(results["grid_X"].min())}')
    print(f'xmax = {int(results["grid_X"].max())}')
    print(f'ymin = {int(results["grid_Y"].min())}')
    print(f'ymax = {int(results["grid_Y"].max())}')
    if cam_name == 'cristal_1':
        xmin = 575800
        xmax = 576140
        ymin = 4819730
        ymax = 4820080
    if cam_name == 'cristal_2':
        xmin = 575600
        xmax = 576100
        ymin = 4819600
        ymax = 4820100
    if cam_name == 'cristal_3':
        xmin = 575910
        xmax = 576430
        ymin = 4819750
        ymax = 4820200

    elif cam_name == 'st_pierre_2':
        xmin = 574300
        xmax = 574700
        ymin = 4818950
        ymax = 4819370
    elif cam_name == 'st_pierre_1':
        xmin = 574650
        xmax = 575112
        ymin = 4819178
        ymax = 4819514
    elif cam_name == 'st_pierre_3':
        xmin = 574500
        xmax = 574925
        ymin = 4819088
        ymax = 4819383
    elif cam_name == 'cristal_merged':
        xmin = -488
        xmax = 487
        ymin = 12
        ymax = 487
    emprise_plot = [xmin, xmax, ymin, ymax]

    # check if ground truth exists:
    ground_truth_exists = type(results['Dgt']).__module__ == np.__name__
    # ground_truth_exists = False

    # affichage
    depth_lims = [0, 8]
    diff_depth_lims = [-1.5, 1.5]

    if plot_only_bathy:
        dir_plot = 'plots_only_bathy'
        output_dir_plot = f'{output_dir}/{dir_plot}/'
        Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
        plot_bathy(results, output_dir_plot, basename, depth_lims, diff_depth_lims, emprise_plot, ground_truth_exists=ground_truth_exists)
    if plot_all_results * ground_truth_exists:
        kalman_error_lims = [0, 1]
        freqlims = [1 / 3, 1 / 15]
        dir_plot = 'plots_all_results'
        output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{dir_plot}/'
        Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
        plot_all_diags(results, output_dir_plot, basename, depth_lims, diff_depth_lims, kalman_error_lims, freqlims)
