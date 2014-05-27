# This program reads in GS2 density fluctuation data in Fourier space, converts to real space
# and calculates the perpendicular  and time correlation function using the Wiener-Kinchin theorem. 
# The program takes in a command line argument that specifies the GS2 NetCDF file that 
# contains the variable ntot(t, spec, ky, kx, theta, ri).
# Use as follows:
#
#     python wk_corr.py <location of .nc file>

import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
from scipy.io import netcdf

#Start timer
t_start = time.clock()

# Read command line argument specifying location of NetCDF file
in_file =  str(sys.argv[1])

#########################
# Function Declarations #
#########################

# Function which converts from Fourier space to real space using 2D FFT
def gs2toreal(f):
  #The Wiener-Khinchin thm states that the autocorrelation function is the FFT of the power spectrum.
  #The power spectrum is defined as abs(A)**2 where A is a COMPLEX array. In this case f.
  f = np.abs(f**2)
  r = np.fft.irfftn(f,axes=[0,1,2], s=[f.shape[0], f.shape[1], 2*(f.shape[2]-1)-1]) #need to use irfft2 since the original signal is real in real space
  return r

#Model function to be fitted to data, as defined in Anthony's papers
def tilted_gauss((x,y), p, lx, ly, ky, Th):
  exp_term = np.exp(- (x/lx)**2 - ((y + Th*x) / ly)**2 )
  cos_term = np.cos(ky*Th*x + ky*y)
  fit_fn =  p + exp_term*cos_term
  return fit_fn.ravel() # fitting function only works on 1D data, reshape later to plot

#############
# Main Code #
#############

ncfile = netcdf.netcdf_file(in_file, 'r')
density = ncfile.variables['ntot_t'][:,0,:,:,10,:] #index = (t, spec, ky, kx, theta, ri)
th = ncfile.variables['theta'][:]
kx = ncfile.variables['kx'][:]
ky = ncfile.variables['ky'][:]
t = ncfile.variables['t'][:]

#Need to interpolate time onto a regular grid in order to do FFT
tnew = np.linspace(min(t), max(t), len(t))
shape = density.shape
ntot_reg = np.empty([shape[0], shape[1], shape[2], shape[3]])
for i in range(shape[1]):
  for j in range(shape[2]):
    for k in range(shape[3]):
      f = interp.interp1d(t, density[:, i, j, k])
      ntot_reg[:, i, j, k] = f(tnew)

#Clear density from memory
density = None

#Now need to FFT in time to obtain ntot as a function of freq, kx, and ky
ntot = np.empty([len(t), len(ky), len(kx)], dtype=complex)
ntot.real = ntot_reg[:,:,:,0]
ntot.imag = ntot_reg[:,:,:,1]
ntot = np.fft.fft(ntot, axis=0)

#Now normalize, transpose (so third index is is on half plane) and do inverse FFT
nx = shape[2]; ny = (len(ky) - 1)*2; nt = shape[0] 
ntot = ntot*nx*ny*nt
ntot = ntot.transpose(0,2,1)
corr = gs2toreal(ntot)
corr = np.fft.fftshift(corr, axes=[0,1,2])

################
# NetCDF Write #
################

#Write correlation function to NetCDF file for later plotting
xpts = np.linspace(-2*np.pi/kx[1], 2*np.pi/kx[1], nx)
ypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], ny-1)
tpts = np.linspace(-max(t), max(t), nt)

f = netcdf.netcdf_file('correlation_function.nc', 'w')

f.createDimension('dt', nt)
f.createDimension('dx', nx)
f.createDimension('dy', ny-1)
f.createDimension('theta', 1)

dt = f.createVariable('dt', 'd', ('dt',))
dx = f.createVariable('dx', 'd', ('dx',))
dy = f.createVariable('dy', 'd', ('dy',))
theta = f.createVariable('theta', 'd', ('theta',))
nc_corr = f.createVariable('correlation', 'd', ('dt', 'dx', 'dy'))

dt[:] = tpts
dx[:] = xpts
dy[:] = ypts
theta[:] = th[10]
nc_corr[:,:,:] = corr[:,:,:]
f.close()

#End timer and print time
print 'Time taken = ', time.clock() - t_start, 'seconds'
















