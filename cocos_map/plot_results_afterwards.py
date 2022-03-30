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


def plot_bathy(results, output_dir_plot, basename, ground_truth_exists=None):
    n_iter = results['t_iter'].size
    for k in range(n_iter):
        if ground_truth_exists:
            fig, ax = plt.subplots(1, 3, figsize=(22, 5))
            mask = np.flipud(np.isnan(results['Dk'][k]))
            diff_depth_ground_truth = np.flipud(results['Dk'][k]) - results['Dgt']
            im1 = ax[0].imshow(results['Dk'][k], extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                   results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='jet_r')
            im2 = ax[1].imshow(np.ma.array(results['Dgt'], mask=mask), extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                          results['grid_Y'][0, 0], results['grid_Y'][-1, 0]],
                         cmap='jet_r', origin = 'lower')
            im3 = ax[2].imshow(diff_depth_ground_truth , extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                   results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='Spectral',
                               origin = 'lower')
            contours = ax[2].contour(results['grid_X'], results['grid_Y'], diff_depth_ground_truth,
                                     levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',), linewidths=(1.5,))
            ax[2].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])
            fig.colorbar(im3, ax=ax[2])
            im1.set_clim([0, 6])
            im2.set_clim([0, 6])
            im3.set_clim([-1.5, 1.5])

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
            # plt.show()
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' %k + '.jpg'))
            plt.close('all')
        else:
            fig, ax = plt.subplots(figsize=(20, 15))
            # depth = np.reshape(results['Dk'][k][0, :], (numrows, numcols), order='F')
            im1 = ax.imshow(results['Dk'][k], extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                             results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='jet_r')
            im1.set_clim([0, 6])
            im2.set_clim([-1.5, 1.5])
            fig.colorbar(im1, ax=ax)
            ax.axis('equal')
            ax.set_title('depth [m]')
            ax.set_ylabel('y [m]')
            ax.set_xlabel('x [m]')
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' % k + '.jpg'))
            plt.close('all')

def plot_all_diags(results, output_dir_plot, basename):
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
    depth_lims = [0, 6]
    diff_depth_lims = [-1.5, 1.5]
    kalman_error_lims = [0, 1]
    freqlims = [1 / 3, 1 / 15]
    for k in range(n_iter):
        diff_depth = np.flipud(results['Dk'][k]) - results['Dgt']
        depth_bias = np.append(depth_bias, np.nanmean(diff_depth))
        depth_rmse = np.append(depth_rmse, np.sqrt(np.nanmean(depth_bias**2)))
        depth_mae = np.append(depth_mae, np.abs(np.nanmean(depth_bias)))
        diff_depth_debiased = diff_depth - np.nanmean(diff_depth)
        depth_debiased_rmse = np.append(depth_debiased_rmse, np.sqrt(np.nanmean(diff_depth_debiased ** 2)))
        iteration_number = np.append(iteration_number, k)
        iteration_time = np.append(iteration_time, results['t_shift_k'][k])

        fig, ax = plt.subplots(2, 5, figsize=(25, 12))

        # depth
        im0 = ax[0, 0].imshow(results['Dk'][k], extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                     results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='jet_r')
        contours_0 = ax[0, 0].contour(results['grid_X'], results['grid_Y'], results['Dgt'], levels=[0.5, 2, 3.5, 5, 7.5, 10, 12.5, 15],
                               colors=('k',), linewidths=(1.5,))
        ax[0, 0].clabel(contours_0, contours_0.levels, fontsize=14, fmt='%.1f')  # , inline=True
        plt.colorbar(im0, ax=ax[0, 0])

        # diff depth
        im1 = ax[0, 1].imshow(diff_depth, extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                               results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='Spectral',
                           origin = 'lower')
        contours = ax[0, 1].contour(results['grid_X'], results['grid_Y'], diff_depth,
                                 levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',), linewidths=(1.5,))
        ax[0, 1].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
        im1.set_clim(diff_depth_lims)
        plt.colorbar(im1, ax=ax[0, 1])

        # Kalman error
        im2 = ax[0, 2].imshow(results['d_K_errors'][k, :, :], extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                             results['grid_Y'][0, 0], results['grid_Y'][-1, 0]],
                              cmap='Spectral')
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
                               np.flipud(results['Dk'][k])[~np.isnan(results['Dgt'])], 1.5, color2)
        ax[1, 2].plot(depth_lims, depth_lims)

        # gc kernel
        im8 = ax[1, 3].imshow(results['gc_kernel'], cmap='jet', origin='lower')

        # spectrum (fft, dmd)
        fft_freqlims_ID = (results['fft_spectrum_k'][k][1] / (2 * np.pi) > freqlims[1]) & (
            results['fft_spectrum_k'][k][1] / (2 * np.pi) < freqlims[0])
        im9 = ax[1, 4].plot(results['fft_spectrum_k'][k][1][fft_freqlims_ID],
                            results['fft_spectrum_k'][k][0][fft_freqlims_ID], 's-', color=(0.4, 0.9, 0))
        im9b = ax[1, 4].plot(results['dmd_spectrum_k'][k][1], results['dmd_spectrum_k'][k][0], '*-', color=(1.0, 0.5, 0))

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
        ax6b.set_ylabel('med.bias [m]')
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
plot_only_bathy = False
plot_all_results = True

# configuration corresponding to given results
fieldsite = 'wavecams_palavas_cristal'
cam_name = 'cristal_1'
cpu_speed = 'fast' #'fast','normal','slow', 'accurate', 'exact'
calcdmd = 'standard' # standard or robust

# load results
output_dir = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/'
f_results = glob(output_dir + f'/results_CPU_speed_{cpu_speed}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
results = np.load(f_results)
basename = Path(f_results).stem

# check if ground truth exists:
ground_truth_exists = type(results['Dgt']).__module__ == np.__name__
# ground_truth_exists = False

# affichage
if plot_only_bathy:
    dir_plot = 'plots_only_bathy'
    output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{dir_plot}/'
    Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
    plot_bathy(results, output_dir_plot, basename, ground_truth_exists=ground_truth_exists)
if plot_all_results * ground_truth_exists:
    dir_plot = 'plots_all_results'
    output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{dir_plot}/'
    Path(output_dir_plot).mkdir(parents=True, exist_ok=True)
    plot_all_diags(results, output_dir_plot, basename)









