# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:26:08 2020

@author: gawehn
"""

#Python modules
import pdb
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt
#COCOS modules
from Data      import Data
from Options   import Options
from DMD       import ExactDMD, OptDMD
from Inversion import Inversion
from Kalman    import Kalman
from Grid      import Grid
from Plot      import Plot
from pathlib import Path
from simple_utils import optional_print as op_print

# execution options
t0 = time.time()
plot_results = False
save_results = True
# cpu_speeds = ['fast', 'normal', 'slow', 'accurate'] #'fast','normal','slow', 'accurate', 'exact'
cpu_speeds = ['fast'] #'fast','normal','slow', 'accurate', 'exact'
calcmdmd = 'standard' # standard or robust

# Fieldsite
fieldsite = 'wavecams_palavas_cristal'
# fieldsite = 'wavecams_palavas_cristal_merged'
# fieldsite = 'wavecams_palavas_stpierre'
# fieldsite = 'chicama'
# ~ fieldsite = 'narrabeen'
cam_name = 'cristal_2'
# cam_name = 'st_pierre_3'
# cam_name = 'cristal_merged'
# cam_name = 'cam'

# output directory
output_dir = f'../results/{fieldsite}/{cam_name}'
Path(output_dir).mkdir(parents=True, exist_ok=True)

# load video data
Video, PlotLims = Data.get_Video(fieldsite, cam_name)

# set options
for cpu_speed in cpu_speeds:
    opts = Options(Video, CPU_speed=cpu_speed, calcmdmd=calcmdmd, parallel_flag=True, gc_kernel_sampnum=80, f_scale=0.012)
    # opts = Options(Video, CPU_speed='fast', parallel_flag=True, gc_kernel_sampnum=80, f_scale=0.012)
    # opts = Options(Video, CPU_speed='fast', parallel_flag=True, gc_kernel_sampnum=80, f_scale=0.012, calcdmd='robust')
    # opts = Options(Video, CPU_speed = 'slow', parallel_flag = True, gc_kernel_sampnum = 80, f_scale    = 0.012)
    # opts = Options(Video, CPU_speed='accurate', parallel_flag = True, gc_kernel_sampnum = 80, f_scale = 0.012,
    #                calcdmd='robust')

    # INITIALIZE
    # prepare parallel computation of grid cells
    def gc_walk_unwrap_self(arg, **kwarg):      #__main__ function for inversion
        return Inversion.gc_walk(*arg, **kwarg)
    # initialize grid
    grid = Grid(Video, opts)
    # initialize inversion
    InvObj  = Inversion()
    # initialize spectral storage
    InvStg  = Inversion.get_storage(grid)
    # initialize SSPC sampling kernel
    InvObj.get_gridKernel(grid,opts.gc_kernel_rad)
    # initialize optimized DMD
    dmd     = OptDMD(opts, alpha0 = None)
    # initialize Kalman filter
    KalObj  = Kalman(opts, grid)
    # initialize plotting
    if plot_results:
        plot = Plot(opts, Video, grid, step = None)

    # preallocation: OPTIONAL: for saving results
    # -------------
    t_iter = []
    Dk = []
    Uk = []
    Vk = []
    Cxk = []
    Cyk = []
    n_used_layers_k = []
    results_c_omega_k = []
    fft_spectrum_k = []
    dmd_spectrum_k = []
    t_shift_k = []
    d_K_errors = []

    # PROCESS
    frame_start = 0
    cnt = 0
    while frame_start+opts.Nt <= Video.ImgSequence.shape[2]: #(remove <= tt for unlimited analysis)
        op_print('\n --------------- START Update #{:} ---------------\n'.format(cnt+1))
        t_real_start    = time.time()
        t               = (frame_start + np.round(opts.Nt/2))*Video.dt
        # make timestamps of frame sequence
        dmd.get_times(Video, frame_start, frame_start+opts.Nt)
        # build video matrix
        dmd.get_Xvid(Video, frame_start, frame_start+opts.Nt)
        try:
            # get Dynamic Modes
            dmd.get_dynamic_modes()
        except:
            op_print('previous local minimizer and re-initialized minimizer failed')
            op_print('try next image sequence?')
            t_real_end = time.time()
            t_shift = t_real_end - t_real_start
            op_print('iteration time {:} sec \n '.format(t_shift))
            cnt += 1
            if opts.frame_int == 'OnTheFly':
                frame_start = frame_start+int(t_shift/Video.dt)
            else:
                frame_start = frame_start+opts.frame_int
            continue
        # frequency filter Dynamic Modes
        dmd.filter_frequencybounds()
        # delete weak Dynamic Modes
        dmd.del_weak_modes()
        # get Fourier compliant spectral amplitudes
        dmd.get_b_fourier()
        # convert Dynamic Modes to phase images
        dmd.transform_phi2phaseIm()
        # stack Dynamic Mode layers
        dmd.stack_phi()
        # OPTIONAL: smoothe Dynamic Modes
        #dmd.clean_phi()
        # get subdomain size and resolution per mode and location
        mask = np.reshape(dmd.badpix_IX,(Video.m,Video.n), order="F")# could be any mask
        grid.get_gcSizes(Video, opts, dmd.omega, mask)
        # invert d,u,v,cx,cy
        Results, InvStg     = InvObj.get_maps(Video, opts, dmd, grid, gc_walk_unwrap_self, InvStg, t)
        # Kalman filter d,u,v,cx,cy
        KalObj.Filter(opts, Results, t)
        # get current timestamp and shift to next image sequence
        t_real_end = time.time()
        t_shift = t_real_end - t_real_start
        cnt += 1
        if opts.frame_int == 'OnTheFly':
            frame_start = frame_start+int(t_shift/Video.dt)
        else:
            frame_start = frame_start+opts.frame_int
        # simple visualization of results
        if plot_results:
            if cnt > 0:
                try:
                    plot.results(opts, grid, Results, KalObj, InvStg, PlotLims.d_lims, PlotLims.diff_lims, PlotLims.err_lims, InvObj.kernel_samp, (dmd.A_fft, dmd.omegas_fft), (dmd.b_fourier,dmd.omega), t_shift)
                    # plot.results_only_bathy(grid, KalObj, t_shift)
                except:
                    op_print('no plot. probably empty results')

        if save_results:
            # save data from update for postprocessing
            t_iter.append(t)
            # Dk.append(np.copy(KalObj.derrt_prev))
            Dk.append(np.flipud(np.reshape(np.copy(KalObj.derrt_prev)[0, :], (grid.Numrows, grid.Numcols), order='F')))
            # Uk.append(np.copy(KalObj.uerrt_prev))
            Uk.append(np.flipud(np.reshape(np.copy(KalObj.uerrt_prev)[0, :], (grid.Numrows, grid.Numcols), order="F")))
            # Vk.append(np.copy(KalObj.verrt_prev))
            Vk.append(np.flipud(np.reshape(np.copy(KalObj.verrt_prev)[0, :], (grid.Numrows,grid.Numcols), order="F")))
            Cxk.append(np.copy(KalObj.cxerrt_prev))
            Cyk.append(np.copy(KalObj.cyerrt_prev))

            n_used_layers = [np.mean([len(lis) for lis in line]) for line in InvStg.omega_store]
            n_used_layers_k.append(np.reshape(n_used_layers, (grid.Numrows, grid.Numcols), order='F'))
            results_c_omega_k = Results.c_omega
            fft_spectrum_k.append((dmd.A_fft, dmd.omegas_fft))
            dmd_spectrum_k.append((dmd.b_fourier,dmd.omega))
            t_shift_k.append(t_shift)
            d_K_errors_tmp = np.copy(KalObj.derrt_prev[1, :])
            d_K_errors_tmp = np.reshape(d_K_errors_tmp, (grid.Numrows, grid.Numcols), order='F')
            d_K_errors.append(np.flipud(d_K_errors_tmp))

    t1 = time.time()
    exec_time = int(t1 - t0)

    if save_results:
        # save other
        gc_kernel = InvObj.kernel_samp
        gc_kernel = gc_kernel.astype(float) * -1
        gc_kernel[int(gc_kernel.shape[0] / 2), int(gc_kernel.shape[1] / 2)] = 1
        gc_kernel[gc_kernel == 0] = np.nan
        Cxy_omega   = Results.c_omega
        Dgt         = Data.get_GroundTruth(opts, Video, grid, step=None)
        numrows = np.asarray(grid.Numrows)
        numcols = np.asarray(grid.Numcols)
        # save in npz compressed format
        np.savez_compressed(f'{output_dir}/results_CPU_speed_{opts.CPU_speed}_calcdmd_{opts.calcdmd}_exec_time_'
                            f'{exec_time}_s', t_iter=t_iter, Dk=Dk, Uk=Uk, Vk=Vk, Cxk=Cxk, Cyk=Cyk, Cxy_omega=Cxy_omega,
                            Dgt=Dgt, grid_dx=grid.dx, grid_X=grid.X, grid_Y=grid.Y, grid_Rows_ctr=grid.Rows_ctr,
                            grid_Cols_ctr=grid.Cols_ctr,
                            n_used_layers_k=n_used_layers_k, results_c_omega_k=results_c_omega_k,
                            fft_spectrum_k=fft_spectrum_k, dmd_spectrum_k=dmd_spectrum_k, t_shift_k=t_shift_k,
                            d_K_errors=d_K_errors, gc_kernel=gc_kernel)
