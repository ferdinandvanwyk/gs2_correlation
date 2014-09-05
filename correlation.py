# This program reads in GS2 density fluctuation data in Fourier space, converts to real space
# and calculates the perpendicular correlation function using the Wiener-Kinchin theorem. 
# The program takes in a command line argument that specifies the GS2 NetCDF file that 
# contains the variable ntot(t, spec, ky, kx, theta, ri).
# Use as follows:
#
#     python correlation.py <time/perp analysis/bes> <location of .nc file>

import os, sys
import operator #enumerate list
import time
import gc #garbage collector
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.signal as sig
from scipy.io import netcdf
import fit #local method
import film #local method
import cPickle as pkl
from mpl_toolkits.mplot3d import Axes3D
from netCDF4 import Dataset

#Make folder which will contain all the correlation analysis
os.system("mkdir analysis")

#Open a diagnostic output file.
diag_file = open("analysis/diag.out", "w")

# Read command line argument specifying location of NetCDF file
analysis =  str(sys.argv[1]) #specify analysis (time or perp)
if analysis != 'perp' and analysis != 'time' and analysis != 'bes':
  raise Exception('Please specify analysis: time or perp.')
in_file =  str(sys.argv[2])
diag_file.write("User specified the following analysis: " + str(analysis) + "\n")
diag_file.write("User specified the following GS2 output file: " + str(in_file) + "\n")

#Normalization parameters
amin = 0.58044 # m
vth = 1.4587e+05 # m/s
rhoref = 6.0791e-03 # m
pitch_angle = 0.6001 # in radians
diag_file.write("The following normalization parameters were used: [amin, vth, rhoref, pitch_angle] = " + str([amin, vth, rhoref, pitch_angle]) + "\n")

#########################
# Function Declarations #
#########################

# Function which converts from GS2 field to complex field which can be passed to fft routines
def real_to_complex_1d(field):
  n1 = field.shape[0]
  cplx_field = np.empty([n1],dtype=complex)
  cplx_field.real = field[:,0]
  cplx_field.imag = field[:,1]
  # fix fft normalisation that is appropriate for numpy fft package
  cplx_field = cplx_field
  return cplx_field

def real_to_complex_2d(field):
  # convert fgs2 to numpy 2D real Fourier transform fr(x,y,theta)
  # nx,ny are number of points within a period
  # => periodic points are nx+1, ny+1 apart
  [nk1, nk2] = [ field.shape[0], field.shape[1]]
  n1 = nk1; n2 = (nk2-1)*2 #assuming second index is half complex
  cplx_field = np.empty([nk1,nk2],dtype=complex)
  cplx_field.real = field[:,:,0]
  cplx_field.imag = field[:,:,1]
  # fix fft normalisation that is appropriate for numpy fft package
  cplx_field = cplx_field
  return cplx_field

def wk_thm_1d(field_1, field_2):
  field = np.conjugate(field_1)*field_2
  corr = np.fft.ifft(field)
  return corr.real*field.shape[0]

# Function which applies WK theorem to a real 2D field field(x,y,ri) where y is assumed to be
# half complex and 'ri' indicates the real/imaginary axis (0 real, 1 imag). 
# The output is the correlation function C(dx, dy).
def wk_thm_2d(c_field):
  #The Wiener-Khinchin thm states that the autocorrelation function is the FFT of the power spectrum.
  #The power spectrum is defined as abs(A)**2 where A is a COMPLEX array. In this case f.
  c_field = np.abs(c_field**2)
  #option 's' below truncates by ny by 1 such that an odd number of y pts are output => need to have corr fn at (0,0)
  corr = np.fft.irfft2(c_field,axes=[0,1], s=[c_field.shape[0], 2*(c_field.shape[1]-1)-1]) #need to use irfft2 since the original signal is real in real space
  return corr*c_field.shape[0]*c_field.shape[1]/2

#
#Fit the 2D correlation function with a tilted Gaussian and extract the fitting parameters
def perp_fit(corr_fn, xpts, ypts, guess):
  shape = corr_fn.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
  x,y = np.meshgrid(xpts, ypts)
  x = np.transpose(x); y = np.transpose(y)

  #Average corr fn over time
  avg_corr = np.empty([nx, ny], dtype=float)
  avg_corr = np.mean(corr_fn, axis=0)

  # lx, ly, kx, ky
  popt, pcov = opt.curve_fit(fit.tilted_gauss, (x, y), avg_corr.ravel(), p0=guess)

  #plt.contourf(xpts, ypts, np.transpose(avg_corr))
  #plt.hold(True)
  #data_fitted = fit.tilted_gauss((x, y), *popt)
  #plt.contour(xpts, ypts, np.transpose(data_fitted.reshape(nx,ny)), 8, colors='w')
  #plt.show()

  return popt

#Fit the peaks of the correlation functions of different dy with decaying exponential
def time_fit(corr_fn, t):
  shape = corr_fn.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];

  #define delta t range
  dt = np.linspace(-max(t)+t[0], max(t)-t[0], nt)
  
  peaks = np.empty([nx, 5]); peaks[:,:] = 0.0;
  max_index = np.empty([nx, 5], dtype=int);
  popt = np.empty([nx], dtype=float)
  mid_idx = 61
  os.system("mkdir analysis/corr_fns")
  for ix in range(0,nx): # loop over all radial points
    for iy in range(mid_idx,mid_idx+5): # only do first 5 functions since rest may be noise
      max_index[ix, iy-mid_idx], peaks[ix, iy-mid_idx] = max(enumerate(corr_fn[:,ix,iy]), key=operator.itemgetter(1))

    if (strictly_increasing(max_index[ix,:]) == True or strictly_increasing(max_index[ix,::-1]) == True):
      #Perform fitting of decaying exponential to peaks
      init_guess = (10.0)
      if max_index[ix, 4] > max_index[ix, 0]:
        popt[ix], pcov = opt.curve_fit(fit.decaying_exp, (dt[max_index[ix,:]]), peaks[ix,:].ravel(), p0=init_guess)
        fit.plot_fit(ix, dt, corr_fn[:,ix,:], max_index[ix,:], peaks[ix,:], mid_idx, popt[ix], 'decaying_exp', amin, vth)
        diag_file.write("Index " + str(ix) + " was fitted with decaying exponential. tau = " + str(popt[ix]) + "\n")
      else:
        popt[ix], pcov = opt.curve_fit(fit.growing_exp, (dt[max_index[ix,::-1]]), peaks[ix,::-1].ravel(), p0=init_guess)
        fit.plot_fit(ix, dt, corr_fn[:,ix,:], max_index[ix, :], peaks[ix,:], mid_idx, popt[ix], 'growing_exp', amin, vth)
        diag_file.write("Index " + str(ix) + " was fitted with growing exponential. tau = " + str(popt[ix]) + "\n")
    else:
      #If abs(max_index) is not monotonically increasing, this usually means
      #that there is no flow and that the above method cannot be used to calculate the correlation time. Try
      #fitting a decaying oscillating exponential to the central peak.
        corr_fn[:,ix,mid_idx] = corr_fn[:,ix,mid_idx]/max(corr_fn[:,ix,mid_idx])
        init_guess = (10.0, 1.0)
        tau_and_omega, pcov = opt.curve_fit(fit.osc_exp, (dt[nt/2-100:nt/2+100]), (corr_fn[nt/2-100:nt/2+100,ix,mid_idx]).ravel(), p0=init_guess)
        popt[ix] = tau_and_omega[0]
        fit.plot_fit(ix, dt, corr_fn[:,ix,:], max_index[ix,:], peaks[ix,:], mid_idx, tau_and_omega, 'osc_exp', amin, vth)
        diag_file.write("Index " + str(ix) + " was fitted with an oscillating Gaussian to the central peak. [tau, omega] = " + str(tau_and_omega) + "\n")
  
  #return correlation time in seconds
  return abs(popt)*1e6*amin/vth 

#Function which checks monotonicity. Returns True or False.
def strictly_increasing(L):
      return all(x<y for x, y in zip(L, L[1:]))

#Function which takes in density fluctuations and outputs the correlation time as a 
#function of the minor radius
def tau_vs_radius(ntot, t):
  shape = ntot.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
  ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  dypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  corr_fn = np.empty([2*nt-1, nx, 2*ny-1], dtype=float); corr_fn[:,:,:] = 0.0

  for ix in range(nx):
    print 'ix = ', ix, ' of ', nx
    corr_fn[:,ix,:] = sig.correlate(ntot[:,ix,:], ntot[:,ix,:])

    #Normalize correlation function by the max
    corr_fn[:,ix,:] = corr_fn[:,ix,:] / np.max(corr_fn[:,ix,:])

  #Fit exponential decay to peaks of correlation function in dt for a few dy's
  tau = time_fit(corr_fn, t) #tau in seconds

  return tau


#############
# Main Code #
#############

#Start timer
t_start = time.clock()

ncfile = netcdf.netcdf_file(in_file, 'r')
density = ncfile.variables['ntot_t'][:,0,:,:,10,:] #index = (t, spec, ky, kx, theta, ri)
th = ncfile.variables['theta'][10]
kx = ncfile.variables['kx'][:]
ky = ncfile.variables['ky'][:]
t = ncfile.variables['t'][:]

#Ensure time is on a regular grid for uniformity
interp_inp = raw_input('Do you want to interpolate the input (y/n)?')
if interp_inp == 'y':
  diag_file.write("User chose to interpolate time onto a regular grid.\n")
  t_reg = np.linspace(min(t), max(t), len(t))
  shape = density.shape
  ntot_reg = np.empty([shape[0], shape[2], shape[1], shape[3]])
  for i in range(shape[1]):
    for j in range(shape[2]):
      for k in range(shape[3]):
        f = interp.interp1d(t, density[:, i, j, k])
        ntot_reg[:, j, i, k] = f(t_reg) #perform a transpose here: ntot(t,kx,ky,theta,ri)
elif interp_inp == 'n':
  diag_file.write("User chose not to interpolate time onto a regular grid.\n")
  t_reg = np.array(t)
  shape = density.shape
  ntot_reg = np.array(np.swapaxes(density, 1, 2)) #ensure ntot_reg[t, kx, ky, ri]


#Zero out density fluctuations which are larger than the BES
zero = raw_input('Do you want to zero out modes that are larger than the BES?')
if zero == 'y':
  diag_file.write("User chose to zero out k-modes larger than the approx size of the BES.\n")
  shape = ntot_reg.shape
  for iky in range(shape[2]):
    for ikx in range(shape[1]):
      if abs(kx[ikx]) < 0.25 and ky[iky] < 0.5: #Roughly the size of BES (160x80mm)
        ntot_reg[:,ikx,iky] = 0.0

#End timer
t_end = time.clock()
print 'Interpolation Time = ', t_end-t_start, ' s'

#Clear density from memory
density = None; f = None; gc.collect();

#################
# Perp Analysis #
#################

if analysis == 'perp':
  diag_file.write("Started 'perp' analysis.\n")
  # Perform inverse FFT to real space ntot[kx, ky, th]
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2
  corr_fn = np.empty([nt,nx,ny-1],dtype=float)
  for it in range(0,nt):
    corr_fn[it,:,:] = wk_thm_2d(real_to_complex_2d(ntot_reg[it,:,:,:]))

    #Shift the zeros to the middle of the domain (only in x and y directions)
    corr_fn[it,:,:] = np.fft.fftshift(corr_fn[it,:,:], axes=[0,1])
    #Normalize the correlation function
    corr_fn[it,:,:] = corr_fn[it,:,:]/np.max(corr_fn[it,:,:])

  xpts = np.linspace(-2*np.pi/kx[1], 2*np.pi/kx[1], nx)
  ypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], ny-1)
  #film.film_2d(xpts, ypts, corr_fn[:,:,:], 100, 'corr')

  #Fit correlation function and get fitting parameters for time slices of a given size
  time_window = 200
  avg_fit_par = np.empty([nt/time_window-1, 4], dtype=float)
  avg_fit_par[-1,:] = [100,10,0.01,0.1]
  for it in range(nt/time_window - 1): 
    avg_fit_par[it, :] = perp_fit(corr_fn[it*time_window:(it+1)*time_window, nx/2-20:nx/2+20, (ny-1)/2-20:(ny-1)/2+20], xpts[nx/2-20:nx/2+20], ypts[(ny-1)/2-20:(ny-1)/2+20], avg_fit_par[it-1,:])
  #avg_fit_par = perp_fit(corr_fn[:, :, :], xpts, ypts)
  avg_fit_par = np.array(avg_fit_par)
  #Write the fitting parameters to a file
  #Order is: [lx, ly, kx, ky]
  np.savetxt('analysis/perp_fit.csv', (avg_fit_par), delimiter=',', fmt='%1.3f')


  # Calculate average correlation function over time
  plt.clf()
  avg_corr = np.mean(corr_fn[:,:,:], axis=0)
  #plt.contourf(xpts[49:79], ypts[21:40], np.transpose(avg_corr[49:79,21:40]), 10)
  plt.contourf(xpts, ypts, np.transpose(avg_corr), 10)
  plt.colorbar()
  plt.xlabel(r'$\Delta x (\rho_i)$')
  plt.ylabel(r'$\Delta y (\rho_i)$')
  plt.savefig('analysis/averaged_correlation.pdf')

  # Plot avg corr function and fit with average fit parameters on same graph
  x,y = np.meshgrid(xpts, ypts)
  x = np.transpose(x); y = np.transpose(y)
  xpts = xpts*rhoref # change to meters
  ypts = ypts*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  data_fitted = fit.tilted_gauss((x, y), *np.mean(avg_fit_par, axis=0))
  plt.clf()
  plt.contourf(xpts[nx/2-20:nx/2+20], ypts[(ny-1)/2-20:(ny-1)/2+20], np.transpose(avg_corr[nx/2-20:nx/2+20,(ny-1)/2-20:(ny-1)/2+20]), 11, levels=np.linspace(-1, 1, 11))
  cbar = plt.colorbar(ticks=np.linspace(-1, 1, 11))
  cbar.ax.tick_params(labelsize=25)
  #plt.hold(True)
  #plt.contour(xpts, ypts, np.transpose(data_fitted.reshape(nx,ny-1)), 8, colors='w')
  plt.title('$C(\Delta x, \Delta y)$', fontsize=25)
  plt.xlabel(r'$\Delta x (m)$', fontsize=25)
  plt.ylabel(r'$\Delta y (m)$', fontsize=25)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.savefig('analysis/sim_perp.pdf')

  plt.clf()
  plt.contourf(xpts[nx/2-20:nx/2+20], ypts[(ny-1)/2-20:(ny-1)/2+20], np.transpose(data_fitted.reshape(nx,ny-1)[nx/2-20:nx/2+20,(ny-1)/2-20:(ny-1)/2+20]), 11, levels=np.linspace(-1, 1, 11))
  plt.title('$C_{fit}(\Delta x, \Delta y)$', fontsize=25)
  cbar = plt.colorbar(ticks=np.linspace(-1, 1, 11))
  cbar.ax.tick_params(labelsize=25)
  plt.xlabel(r'$\Delta x (m)$', fontsize=25)
  plt.ylabel(r'$\Delta y (m)$', fontsize=25)
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.savefig('analysis/fit_perp.pdf')

  #End timer
  t_end = time.clock()
  print 'Total Time = ', t_end-t_start, ' s'

#################
# Time Analysis #
#################
# The time correlation analysis involves looking at each radial location
# separately, and calculating C(dy, dt). Depending on whether there is significant
# flow, can fit a decaying exponential to either the central peak or the envelope
# of peaks of different dy's
elif analysis == 'time':
  diag_file.write("Started 'time' analysis.\n")
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2

  # Need to IFFT in x so that x index represents radial locations
  ntot_real_space = np.empty([nt,nx,ny],dtype=complex)
  for it in range(nt):
    ntot_real_space[it,:,:] = np.fft.irfft2(real_to_complex_2d(ntot_reg[it,:,:,:]), axes=[0,1])

  #Clear memory
  ntot_reg = None; gc.collect();
  
  #mask terms which are zero to not skew standard deviations
  #tau_mask = np.ma.masked_equal(tau, 0)
  time_window = 200
  tau_v_r = np.empty([nt/time_window-1, nx], dtype=float)
  for it in range(nt/time_window - 1): 
    tau_v_r[it, :] = tau_vs_radius(ntot_real_space[it*time_window:(it+1)*time_window,:,:], t[it*time_window:(it+1)*time_window])

  np.savetxt('analysis/time_fit.csv', (tau_v_r), delimiter=',', fmt='%1.3f')

  #Plot correlation time as a function of radius
  plt.clf()
  plt.plot(np.mean(tau_v_r, axis=0))
  plt.savefig('analysis/time_corr.pdf')

  #End timer
  t_end = time.clock()
  print 'Total Time = ', t_end-t_start, ' s'

##############
# BES Output #
##############
elif analysis == 'bes':
  diag_file.write("Started 'bes' film making and density fluctuations write out to NetCDF.\n")
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2

  real_space_density = np.empty([nt,nx,ny],dtype=float)
  for it in range(nt):
    real_space_density[it,:,:] = np.fft.irfft2(real_to_complex_2d(ntot_reg[it,:,:,:])*nx*ny/2, axes=[0,1])

  xpts = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref # change to meters
  ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  tpts = np.array(t*amin/vth)

  #Write out density fluctuations in real space to be analyzed
  nc_file = netcdf.netcdf_file('analysis/density.nc', 'w')
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
  nc_ntot[:,:,:] = real_space_density[:,:,:]
  nc_file.close()

  #Export film
  print 'Exporting film...'
  film.real_space_film_2d(xpts, ypts, real_space_density[:,:,:], 'density')

diag_file.write(str(analysis) + "analysis finished succesfully.\n")
plt.close()
diag_file.close()
