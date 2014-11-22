
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

# Check config parameter and interpolate in time if specified. A regular time
# grid is required by the FFT routines.
if interpolate:
    t_reg = np.linspace(min(t), max(t), len(t))
    shape = cdf_field.shape
    # Create empty array and squeeze axes of length 1
    field = np.empty(shape)
    # Swap ky and kx axes such that index = [t, kx, ky, ri]
    field = np.array(np.swapaxes(field, 1, 2)) 
    for i in range(shape[1]):
        for j in range(shape[2]):
            for k in range(shape[3]):
                f = interp.interp1d(t, cdf_field[:, i, j, k])
                # Transpose: ntot(t,kx,ky,theta,ri)
                field[:, j, i, k] = f(t_reg)
    t = t_reg
else:
    diag_file.write("User chose not to interpolate time onto a regular grid.\n")
    t = np.array(t)
    shape = cdf_field.shape
    # Transpose field and squeeze axes length 1 
    field = np.array(np.swapaxes(np.squeeze(cdf_field), 1, 2)) 

# Zero out density fluctuations which are larger than the BES
if zero_bes_scales:
    diag_file.write("User chose to zero out k-modes larger than the approx "
                    "size of the BES.\n")
    shape = field.shape
    for iky in range(shape[2]):
        for ikx in range(shape[1]):
            # Roughly the size of BES (160x80mm)
            if abs(kx[ikx]) < 0.25 and ky[iky] < 0.5: 
                field[:,ikx,iky,:] = 0.0

# End timer
t_end = time.clock()
diag_file.write('Interpolation Time = ' + str(t_end-t_start) + ' s' + "\n")

# Clear NetCDF field from memory
cdf_field = None; f = None; gc.collect();

# Define shape variable for use by analysis procedures
shape = field.shape
nt = shape[0]
nx = shape[1]
nky = shape[2]
ny = (nky-1)*2

#################
# Perp Analysis #
#################
if analysis == 'perp':
    diag_file.write("Started 'perp' analysis.\n")
    corr_fn = np.empty([nt,nx,ny-1],dtype=float)
    # Perform inverse FFT to real space ntot[kx, ky, th]
    for it in range(0,nt):
        corr_fn[it,:,:] = wk_thm_2d(real_to_complex_2d(field[it,:,:,:]))

        # Shift the zeros to the middle of the domain (only in x and y dirns)
        corr_fn[it,:,:] = np.fft.fftshift(corr_fn[it,:,:], axes=[0,1])
        # Normalize the correlation function
        corr_fn[it,:,:] = corr_fn[it,:,:]/np.max(corr_fn[it,:,:])

    xpts = np.linspace(-2*np.pi/kx[1], 2*np.pi/kx[1], nx)
    ypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], ny-1)
    # film.film_2d(xpts, ypts, corr_fn[:,:,:], 100, 'corr')

    # Fit correlation function and get fitting parameters for time slices
    time_window = 20
    avg_fit_par = np.empty([int(nt/time_window)-1, 4], dtype=float)
    avg_fit_par[-1,:] = [100,10,0.01,0.1]
    for it in range(int(nt/time_window) - 1): 
        avg_fit_par[it, :] = fit.perp_fit(corr_fn[it*time_window:(it+1)*time_window, 
            nx/2-20:nx/2+20, (ny-1)/2-20:(ny-1)/2+20], xpts[nx/2-20:nx/2+20], 
            ypts[(ny-1)/2-20:(ny-1)/2+20], avg_fit_par[it-1,:])
    avg_fit_par = np.array(avg_fit_par)
    # Write the fitting parameters to a file
    # Order is: [lx, ly, kx, ky]
    np.savetxt(out_dir + '/perp_fit.csv', (avg_fit_par), delimiter=',', 
               fmt='%1.3f')

    # Calculate average correlation function over time
    plt.clf()
    avg_corr = np.mean(corr_fn[:,:,:], axis=0)
    plt.contourf(xpts, ypts, np.transpose(avg_corr), 10)
    plt.colorbar()
    plt.xlabel(r'$\Delta x (\rho_i)$')
    plt.ylabel(r'$\Delta y (\rho_i)$')
    plt.savefig(out_dir + '/averaged_correlation.pdf')

    # Plot avg corr function and fit with average fit parameters on same graph
    x,y = np.meshgrid(xpts, ypts)
    x = np.transpose(x); y = np.transpose(y)
    xpts = xpts*rhoref # change to meters
    ypts = ypts*rhoref*np.tan(pitch_angle) # change to meters and pol plane
    data_fitted = fit.tilted_gauss((x, y), *np.mean(avg_fit_par, axis=0))
    plt.clf()
    plt.contourf(xpts[nx/2-20:nx/2+20], ypts[(ny-1)/2-20:(ny-1)/2+20], 
                 np.transpose(avg_corr[nx/2-20:nx/2+20,(ny-1)/2-20:(ny-1)/2+20]), 
                 11, levels=np.linspace(-1, 1, 11))
    cbar = plt.colorbar(ticks=np.linspace(-1, 1, 11))
    cbar.ax.tick_params(labelsize=25)
    plt.title('$C(\Delta x, \Delta y)$', fontsize=25)
    plt.xlabel(r'$\Delta x (m)$', fontsize=25)
    plt.ylabel(r'$\Delta y (m)$', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(out_dir + '/sim_perp.pdf')

    plt.clf()
    plt.contourf(xpts[nx/2-20:nx/2+20], ypts[(ny-1)/2-20:(ny-1)/2+20], 
                 np.transpose(data_fitted.reshape(nx,ny-1)[nx/2-20:nx/2+20,
                 (ny-1)/2-20:(ny-1)/2+20]), 11, levels=np.linspace(-1, 1, 11))
    plt.title('$C_{fit}(\Delta x, \Delta y)$', fontsize=25)
    cbar = plt.colorbar(ticks=np.linspace(-1, 1, 11))
    cbar.ax.tick_params(labelsize=25)
    plt.xlabel(r'$\Delta x (m)$', fontsize=25)
    plt.ylabel(r'$\Delta y (m)$', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig(out_dir + '/fit_perp.pdf')

    #End timer
    t_end = time.clock()
    diag_file.write('Total Time = ' + str(t_end-t_start) + ' s' + "\n")

#################
# Time Analysis #
#################
# The time correlation analysis involves looking at each radial location
# separately, and calculating C(dy, dt). Depending on whether there is 
# significant flow, can fit a decaying exponential to either the central peak 
# or the envelope of peaks of different dy's
elif analysis == 'time':
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

##############
# BES Output #
##############
elif analysis == 'bes':
    diag_file.write("Started 'bes' film making and density fluctuations write "
                    "out to NetCDF.\n")

    field_real_space = np.empty([nt,nx,ny],dtype=float)
    for it in range(nt):
        field_real_space[it,:,:] = np.fft.irfft2(real_to_complex_2d(
                                            field[it,:,:,:]), axes=[0,1])
        field_real_space[it,:,:] = np.roll(field_real_space[it,:,:], 
                                             int(nx/2), axis=0)

    xpts = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref 
    ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle)  
    tpts = np.array(t*amin/vth)

    #Write out density fluctuations in real space to be analyzed
    nc_file = netcdf.netcdf_file(out_dir + '/density.nc', 'w')
    nc_file.createDimension('x',nx)
    nc_file.createDimension('y',ny)
    nc_file.createDimension('t',nt)
    nc_x = nc_file.createVariable('x','d',('x',))
    nc_y = nc_file.createVariable('y','d',('y',))
    nc_t = nc_file.createVariable('t','d',('t',))
    nc_ntot = nc_file.createVariable('n','d',('t', 'x', 'y',))
    nc_x[:] = xpts[:]
    nc_y[:] = ypts[:]
    nc_t[:] = tpts[:] - tpts[0]
    nc_ntot[:,:,:] = field_real_space[:,:,:]
    nc_file.close()

    #Export film
    print ('Exporting film...')
    film.real_space_film_2d(xpts, ypts, field_real_space[:,:,:],
                            in_field, out_dir)

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
