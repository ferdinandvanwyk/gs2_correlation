
# Standard packages
import os
import sys
import operator #enumerate list
import time
import gc #garbage collector
import configparser

# External Packages
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
import scipy.interpolate as interp
from scipy.io import netcdf
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset

# Local methods
import fit
import film
import wk

# Define shape variable for use by analysis procedures
shape = field.shape
nt = shape[0]
nx = shape[1]
nky = shape[2]
ny = (nky-1)*2

#################
# Time Analysis #
#################
# The time correlation analysis involves looking at each radial location
# separately, and calculating C(dy, dt). Depending on whether there is 
# significant flow, can fit a decaying exponential to either the central peak 
# or the envelope of peaks of different dy's
if analysis == 'time':
    diag_file.write("Started 'time' analysis.\n")

    # Need to IFFT in x so that x index represents radial locations
    field_real_space = np.empty([nt,nx,ny],dtype=float)
    for it in range(nt):
        field_real_space[it,:,:] = np.fft.irfft2(real_to_complex_2d(
                                        field[it,:,:,:]), axes=[0,1])
        field_real_space[it,:,:] = np.roll(field_real_space[it,:,:], 
                                             nx/2, axis=0)

    #Clear memory
    field = None; gc.collect();

    #mask terms which are zero to not skew standard deviations
    #tau_mask = np.ma.masked_equal(tau, 0)
    time_window = 200
    tau_v_r = np.empty([nt/time_window-1, nx], dtype=float)
    for it in range(nt/time_window - 1): 
        tau_v_r[it, :] = tau_vs_radius(field_real_space[it*time_window:(it+1)*
                        time_window,:,:], t[it*time_window:(it+1)*time_window],
                        out_dir, diag_file)

    np.savetxt(out_dir + '/time_fit.csv', (tau_v_r), delimiter=',', fmt='%1.3f')

    #Plot correlation time as a function of radius
    plt.clf()
    plt.plot(np.mean(tau_v_r, axis=0))
    plt.savefig(out_dir + '/time_corr.pdf')

    #End timer
    t_end = time.clock()
    print('Total Time = ', t_end-t_start, ' s')

#######################
# Zonal Flow Analysis #
#######################
elif analysis == 'zf':
    diag_file.write("Started zonal flow correlation analysis.\n")

    # phi = phi[t,kx,ky,ri]
    # Need to multiply by nx since ifft contains 1/nx implicitly but 
    # spectral->real for GS2 variables require no factor. Finally, zf_vel is in 
    # units of (1/kxfac vth) since technically: zf_vel = kxfac*IFT[(kx*phi_imag)] 
    # however kxfac calculation is nontrivial.
    zf_vel = np.empty([nt,nx],dtype=float)
    for it in range(nt):
        zf_vel[it,:] = np.fft.ifft(real_to_complex_1d(field[it,:,0,:])*kx).imag*nx

    #ZF vs x and t
    plt.clf()
    plt.contourf(zf_vel)
    plt.title('$v_{ZF}(x, t))$', fontsize=25)
    plt.colorbar()
    plt.xlabel(r'$ x (\rho_i)$', fontsize=25)
    plt.ylabel(r'$t (a / v_{thi})$', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(out_dir + '/zf_2d.pdf')

    # Mean ZF vs x
    plt.clf()
    plt.plot(np.mean(zf_vel, axis=0))
    plt.title('$v_{ZFi, mean}(x))$', fontsize=25)
    plt.xlabel(r'$ x (\rho_i)$', fontsize=25)
    plt.ylabel(r'$v_{ZF} (v_{thi}/kxfac)$', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(out_dir + '/zf_mean.pdf')

diag_file.write(analysis + " analysis finished succesfully.\n")
plt.close()
diag_file.close()
