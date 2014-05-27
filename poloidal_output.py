# This program reads in GS2 density fluctuation data in Fourier space, converts to real space
# and calculates the perpendicular correlation function using the Wiener-Kinchin theorem. 
# The program takes in a command line argument that specifies the GS2 NetCDF file that 
# contains the variable ntot(t, spec, ky, kx, theta, ri).
# Use as follows:
#
#     python wk_corr.py <location of .nc file>

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt
from scipy.io import netcdf

# Read command line argument specifying location of NetCDF file
in_file =  str(sys.argv[1])

#Normalization parameters
amin = 0.58044 # m
vth = 1.4587e+05 # m/s
rhoref = 6.0791e-03 # m
pitch_angle = 0.6001 # in radians

#########################
# Function Declarations #
#########################

# Function which converts from GS2 field to complex field which can be passed to fft routines
def fgs2tofr( fgs2 ):
  # convert fgs2 to numpy 2D real Fourier transform fr(x,y,theta)
  # nx,ny are number of points within a period
  # => periodic points are nx+1, ny+1 apart
  # f(ix,iy)=f(ix+nx+1,iy)
  # f(ix,iy)=f(ix,iy+ny+1)
  [nky, nkx] = [ fgs2.shape[0], fgs2.shape[1]]
  ny = (nky-1)*2 ; nx=nkx #when ifft is done, one index will be larger, y idx in this case
  fr = np.empty([nkx,nky],dtype=complex)
  fgs2_tr = fgs2.transpose(1,0,2) #needs to be done since ifft assumes 2nd index is on the half plane
  fr.real = fgs2_tr[:,:,0]
  fr.imag = fgs2_tr[:,:,1]
  # fix fft normalisation that is appropriate for numpy fft package
  fr = fr*nx*ny/2
  return fr

# Function which converts from Fourier space to real space using 2D FFT
def fgs2toreal( fgs2 ):
  # convert fgs2 to real data in real space using numpy 2D real FFT
  f = fgs2tofr(fgs2)
  r = np.fft.irfft2(f,axes=[0,1]) #need to use irfft2 since the original signal is real in real space
  return r

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

# Perform inverse FFT to real space ntot[kx, ky, th]
nx = shape[2]
nky = shape[1]
ny = (nky-1)*2
nth = shape[3]
nt = shape[0]
ntot = np.empty([nt,nx,ny],dtype=float)
for it in range(0,density.shape[0]):
  ntot[it,:,:] = fgs2toreal(ntot_reg[it,:,:,:])

# Clear density to free memory
density = None
#Make film of correlation function in time
files = []
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
xpts = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref # change to meters
ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
os.system("mkdir film_frames")
for it in range(len(t)):
      ax.cla()
      ax.contourf(xpts, ypts, np.transpose(ntot[it,:,:]), levels=np.linspace(-20, 15, 11))
      plt.xlabel(r'Radial Direction x $(m)$')
      plt.ylabel(r'Poloidal Direction y $(m)$')
      fname = 'film_frames/ntot_tmp%04d.png'%it
      print 'Saving frame', fname
      fig.savefig(fname)
      files.append(fname)

print 'Making movie animation.mp4 - this make take a while'
os.system("ffmpeg -threads 2 -y -f image2 -r 40 -i 'film_frames/ntot_tmp%04d.png' density_animation.mp4")


################
# NetCDF Write #
################

#Write correlation function to NetCDF file for later plotting
tpts = np.linspace(min(t), max(t), nt)

f = netcdf.netcdf_file('poloidal_output.nc', 'w')

f.createDimension('t', nt)
f.createDimension('x', nx)
f.createDimension('y', ny)
f.createDimension('theta', 1)

dt = f.createVariable('t', 'd', ('t',))
dx = f.createVariable('x', 'd', ('x',))
dy = f.createVariable('y', 'd', ('y',))
theta = f.createVariable('theta', 'd', ('theta',))
nc_ntot = f.createVariable('n', 'd', ('t', 'x', 'y'))

# Recover SI units
dt[:] = tpts*amin/vth # seconds
dx[:] = xpts
dy[:] = ypts
theta[:] = th[10]
nc_ntot[:,:,:] = ntot[:,:,:]
f.close()
