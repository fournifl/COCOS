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


def plot_only_bathy(results, output_dir_plot, basename):
    n_iter = results['t_iter'].size
    ground_truth_exists = type(results['Dgt']).__module__ == np.__name__
    ground_truth_exists = False
    for k in range(n_iter):
        numrows = len(results['grid_Rows_ctr'])
        numcols = len(results['grid_Cols_ctr'])
        if ground_truth_exists:
            fig, ax = plt.subplots(1, 3, figsize=(22, 5))
            depth = np.reshape(results['Dk'][k][0, :], (numrows, numcols),order = 'F')
            mask = np.flipud(np.isnan(depth))
            diff_depth_ground_truth = np.flipud(depth) - results['Dgt']
            im1 = ax[0].imshow(depth, extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                   results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='jet_r')
            im2 = ax[1].imshow(np.ma.array(results['Dgt'], mask=mask), extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                          results['grid_Y'][0, 0], results['grid_Y'][-1, 0]],
                         cmap='jet_r', origin = 'lower')
            im3 = ax[2].imshow(diff_depth_ground_truth , extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                                   results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='Spectral_r',
                               origin = 'lower')
            contours = ax[2].contour(results['grid_X'], results['grid_Y'], diff_depth_ground_truth,
                                     levels=[-1.5, -0.5, 0.5, 1.5], colors=('k',), linewidths=(1.5,))
            ax[2].clabel(contours, contours.levels, fontsize=14, fmt='%.1f')  # , inline=True
            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar3 = fig.colorbar(im3, ax=ax[2])
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
            depth = np.reshape(results['Dk'][k][0, :], (numrows, numcols), order='F')
            im1 = ax.imshow(depth, extent=[results['grid_X'][0, 0], results['grid_X'][0, -1],
                                             results['grid_Y'][0, 0], results['grid_Y'][-1, 0]], cmap='jet_r')
            im1.set_clim([0, 6])
            cbar1 = fig.colorbar(im1, ax=ax)
            ax.axis('equal')
            ax.set_title('depth [m]')
            ax.set_ylabel('y [m]')
            ax.set_xlabel('x [m]')
            fig.savefig(Path(output_dir_plot).joinpath(basename + '_' + '%02d' % k + '.jpg'))
            plt.close('all')

# execution options
plot_bathy = True
plot_whole_results = False

# configuration corresponding to given results
fieldsite = 'wavecams_palavas_cristal'
cam_name = 'cristal_1'
cpu_speed = 'fast' #'fast','normal','slow', 'accurate', 'exact'
calcdmd = 'standard' # standard or robust

# load results
output_dir = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/'
if plot_bathy:
    dir_plot = 'plots_only_bathy'
elif plot_whole_results:
    dir_plot = 'plots_whole_results'
output_dir_plot = f'/home/florent/dev/COCOS/results/{fieldsite}/{cam_name}/{dir_plot}/'
Path(output_dir_plot).mkdir(parents=True, exist_ok=True)

f_results = glob(output_dir + f'/results_CPU_speed_{cpu_speed}_calcdmd_{calcdmd}_exec_time_*.npz')[0]
results = np.load(f_results)
basename = Path(f_results).stem

# affichage
plot_only_bathy(results, output_dir_plot, basename)








