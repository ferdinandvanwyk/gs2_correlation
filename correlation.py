# This program reads in GS2 density fluctuation data in Fourier space, converts to real space
# and calculates the perpendicular correlation function using the Wiener-Kinchin theorem. 
# The program takes in a command line argument that specifies the GS2 NetCDF file that 
# contains the variable ntot(t, spec, ky, kx, theta, ri).
# Use as follows:
#
#     python correlation.py <time/perp analysis> <location of .nc file>

import os, sys
import operator #enumerate list
import time
import gc #garbage collector
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
from scipy.io import netcdf
import fit #local method
import film #local method

# Read command line argument specifying location of NetCDF file
analysis =  str(sys.argv[1]) #specify analysis (time or perp)
if analysis != 'perp' and analysis != 'time' and analysis != 'bes':
  raise Exception('Please specify analysis: time or perp.')
in_file =  str(sys.argv[2])

#Normalization parameters
amin = 0.58044 # m
vth = 1.4587e+05 # m/s
rhoref = 6.0791e-03 # m
pitch_angle = 0.6001 # in radians

#########################
# Function Declarations #
#########################

# Function which converts from GS2 field to complex field which can be passed to fft routines
def real_to_complex(field):
  # convert fgs2 to numpy 2D real Fourier transform fr(x,y,theta)
  # nx,ny are number of points within a period
  # => periodic points are nx+1, ny+1 apart
  [nk1, nk2] = [ field.shape[0], field.shape[1]]
  n1 = nk1; n2 = (nk2-1)*2 #assuming second index is half complex
  cplx_field = np.empty([nk1,nk2],dtype=complex)
  cplx_field.real = field[:,:,0]
  cplx_field.imag = field[:,:,1]
  # fix fft normalisation that is appropriate for numpy fft package
  cplx_field = cplx_field*n1*n2/2
  return cplx_field

# Function which applies WK theorem to a real 2D field field(x,y,ri) where y is assumed to be
# half complex and 'ri' indicates the real/imaginary axis (0 real, 1 imag). 
# The output is the correlation function C(dx, dy).
def wk_thm(field):
# create complex field out of 
  c_field = real_to_complex(field)
  #The Wiener-Khinchin thm states that the autocorrelation function is the FFT of the power spectrum.
  #The power spectrum is defined as abs(A)**2 where A is a COMPLEX array. In this case f.
  c_field = np.abs(c_field**2)
  #option 's' below truncates by ny by 1 such that an odd number of y pts are output => need to have corr fn at (0,0)
  corr = np.fft.irfft2(c_field,axes=[0,1], s=[c_field.shape[0], 2*(c_field.shape[1]-1)-1]) #need to use irfft2 since the original signal is real in real space
  return corr

#Fit the 2D correlation function with a tilted Gaussian and extract the fitting parameters
def perp_fit(corr_fn, xpts, ypts):
  shape = corr_fn.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
  x,y = np.meshgrid(xpts, ypts)
  x = np.transpose(x); y = np.transpose(y)

  #Average corr fn over time
  avg_corr = np.empty([nx, ny], dtype=float)
  avg_corr = np.mean(corr_fn, axis=0)

  # lx, ly, kx, ky
  init_guess = (10.0, 10.0, 0.01, 0.01)
  popt, pcov = opt.curve_fit(fit.tilted_gauss, (x, y), avg_corr.ravel(), p0=init_guess)

  return popt

#Fit the peaks of the correlation functions of different dy with decaying exponential
def time_fit(corr_fn, t):
  dt = t - t[0] #define delta t range
  
  shape = corr_fn.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
  peaks = np.empty([nx, 5]); peaks[:,:] = 0.0;
  max_index = np.empty([nx, 5], dtype=int);
  popt = np.empty([nx], dtype=float)
  for ix in range(0,nx): # loop over all radial points
    for iy in range(30,35): # only do first 5 functions since rest may be noise
      max_index[ix, iy-30], peaks[ix, iy-30] = max(enumerate(corr_fn[:,ix,iy]), key=operator.itemgetter(1))

    #Perform fitting of decaying exponential to peaks
    init_guess = (10.0)
    popt[ix], pcov = opt.curve_fit(fit.decaying_exp, (dt[max_index[ix,:]]), peaks[ix,:].ravel(), p0=init_guess)
  
    #Need to take care of cases where there is no flow. Test if max_index for first two peaks are
    #both at zero. If so just fit the central peak with a decaying exponential. 
    if max_index[ix, 0] == max_index[ix, 1]:
      popt[ix] = 0

  return popt

# Define general function which takes in density n = n(t, x, ky, ri) and gives correlation time
# as a function of raduis for that time window
def time_corr_vs_radius(ntot_reg, t):
  shape = ntot_reg.shape; nt = shape[0]; nx = shape[1]; nky = shape[2]; ny = (nky-1)*2

  # Pad the original data with an equal number of zeroes, following 
  # the B&P method of separating out the circular correlation terms.
  ntot_pad = np.empty([2*nt, nx, nky, shape[3]])
  ntot_pad[:,:,:,:] = 0.0
  ntot_pad[0:nt,:,:,:] = ntot_reg

  # Need to FFT in t so that WK thm can be used
  f = np.empty([2*nt, nx, nky], dtype=complex)
  f.real = ntot_pad[:, :, :, 0]
  f.imag = ntot_pad[:, :, :, 1]
  f = np.fft.fft(f,axis=0)  #fft(t)
  ntot_pad[:,:,:,0] = f.real #ntot_reg(t, x, ky, ri)
  ntot_pad[:,:,:,1] = f.imag

  #Clear memory
  ntot_reg = None, gc.collect();

  # For each x value in the outboard midplane (theta=0) calculate the function C(dt,dy)
  corr_fn = np.empty([nt,nx,ny-1],dtype=float)
  for ix in range(0,nx):
    #Do correlation analysis but only keep first half as per B&P
    corr_fn[:,ix,:] = wk_thm(ntot_pad[:,ix,:,:])[0:nt,:]

    #Shift the zeros to the middle of the domain (only in t and y directions)
    corr_fn[:,ix,:] = np.fft.fftshift(corr_fn[:,ix,:], axes=[1])

    #Normalize correlation function to number of products as per B&P (11.96)
    norm = nt/(nt-np.linspace(0,nt-1,nt))
    for iy in range(0,ny-1):
      corr_fn[:,ix,iy] = norm*corr_fn[:,ix,iy]

    #Normalize the correlation function
    corr_fn[:,ix,:] = corr_fn[:,ix,:]/np.max(corr_fn[:,ix,:])

  #Clear memory
  ntot_pad = None; gc.collect();

  # Plot one function to illustrate procedure
  xvalue = 30 
  plt.clf()
  plt.plot(dt[:], corr_fn[:,xvalue,30:35])
  plt.legend(p1, [r'$\exp[-|\Delta t_{peak} / \tau_c]$'])
  plt.xlabel(r'$\Delta t (a/v_{thr})$')
  plt.ylabel(r'$C_{\Delta y}(\Delta t)$')
  plt.savefig('analysis/time_fit.pdf')


  #Fit correlation function and get fitting parameters for time slices of a given size
  return time_fit(corr_fn, t)

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
t_reg = np.linspace(min(t), max(t), len(t))
shape = density.shape
ntot_reg = np.empty([shape[0], shape[2], shape[1], shape[3]])
for i in range(shape[1]):
  for j in range(shape[2]):
    for k in range(shape[3]):
      f = interp.interp1d(t, density[:, i, j, k])
      ntot_reg[:, j, i, k] = f(t_reg) #perform a transpose here: ntot(t,kx,ky,theta,ri)

#Zero out density fluctuations which are larger than the BES
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

#Make folder which will contain all the correlation analysis
os.system("mkdir analysis")

#################
# Perp Analysis #
#################

if analysis == 'perp':
  # Perform inverse FFT to real space ntot[kx, ky, th]
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2
  corr_fn = np.empty([nt,nx,ny-1],dtype=float)
  for it in range(0,nt):
    corr_fn[it,:,:] = wk_thm(ntot_reg[it,:,:,:])

    #Shift the zeros to the middle of the domain (only in x and y directions)
    corr_fn[it,:,:] = np.fft.fftshift(corr_fn[it,:,:], axes=[0,1])
    #Normalize the correlation function
    corr_fn[it,:,:] = corr_fn[it,:,:]/np.max(corr_fn[it,:,:])

  xpts = np.linspace(-2*np.pi/kx[1], 2*np.pi/kx[1], nx)
  ypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], ny-1)
  film.film_2d(xpts, ypts, corr_fn[:,:,:], 100, 'corr')

  #Fit correlation function and get fitting parameters for time slices of a given size
  time_window = 200
  avg_fit_par = np.empty([nt/time_window-1, 4], dtype=float)
  for it in range(nt/time_window - 1): 
    avg_fit_par[it, :] = perp_fit(corr_fn[it*time_window:(it+1)*time_window, :, :], xpts, ypts)
  avg_fit_par = np.array(avg_fit_par)
  #Write the fitting parameters to a file
  #Order is: [lx, ly, kx, ky]
  np.savetxt('analysis/perp_fit.csv', (np.mean(avg_fit_par, axis=0), np.std(avg_fit_par, axis=0)), delimiter=',', fmt='%1.3f')


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
  data_fitted = fit.tilted_gauss((x, y), *np.mean(avg_fit_par, axis=0))
  plt.clf()
  plt.contourf(xpts, ypts, np.transpose(avg_corr), 8)
  plt.colorbar()
  plt.hold(True)
  plt.contour(xpts, ypts, np.transpose(data_fitted.reshape(nx,ny-1)), 8, colors='w')
  plt.xlabel(r'$\Delta x (\rho_i)$')
  plt.ylabel(r'$\Delta y (\rho_i)$')
  plt.savefig('analysis/perp_fit.pdf')

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
  # As given in Bendat & Piersol, need to pad time series with zeros to separate parts
  # of circular correlation function 
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2

  # Need to IFFT in x so that x index represents radial locations
  f = np.empty([nt, nx, nky], dtype=complex)
  f.real = ntot_reg[:, :, :, 0]
  f.imag = ntot_reg[:, :, :, 1]
  f = np.fft.ifft(f,axis=1) #ifft(kx)
  ntot_reg[:,:,:,0] = f.real #ntot_reg(t, x, ky, ri)
  ntot_reg[:,:,:,1] = f.imag

  #Clear memory
  f = None; gc.collect();
  
  ntot_reg =  ntot_reg[:,:,::-1,:]
  time_window = 500
  tau = np.empty([nt/time_window-1, nx], dtype=float)
  for it in range(nt/time_window - 1): 
    tau[it, :] = time_corr_vs_radius(ntot_reg[it*time_window:(it+1)*time_window, :, :, :], t_reg[it*time_window:(it+1)*time_window])

  tau = np.array(tau)
  plt.plot(tau[0, :])
  plt.show()

#  #Write correlation times to file
#  np.savetxt('analysis/time_fitting.csv', (popt,), delimiter=',', fmt='%1.4e')
#
#  #Plot correlation time as a function of radius
#  plt.clf()
#  plt.plot(popt)
#  plt.savefig('analysis/time_corr.pdf')
#
  #End timer
  t_end = time.clock()
  print 'Total Time = ', t_end-t_start, ' s'

##############
# BES Output #
##############
elif analysis == 'bes':
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2

  real_space_density = np.empty([nt,nx,ny],dtype=float)
  for it in range(nt):
    real_space_density[it,:,:] = np.fft.irfft2(real_to_complex(ntot_reg[it,:,:,:]), axes=[0,1])

  #Export film
  print 'Exporting film...'
  xpts = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref # change to meters
  ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  film.real_space_film_2d(xpts, ypts, real_space_density, nt, 'density')









