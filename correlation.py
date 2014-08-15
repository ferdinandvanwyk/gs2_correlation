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
plt.rcParams.update({'figure.autolayout': True})
import scipy.optimize as opt
import scipy.interpolate as interp
from scipy.io import netcdf
import fit #local method
import film #local method
import cPickle as pkl
from mpl_toolkits.mplot3d import Axes3D

# Read command line argument specifying location of NetCDF file
analysis =  str(sys.argv[1]) #specify analysis (time or perp)
if analysis != 'perp' and analysis != 'time' and analysis != 'bes' and analysis!='test':
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
  corr = np.fft.ifft(field)*field.shape[0]
  return corr.real

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
  for ix in range(0,nx): # loop over all radial points
    for iy in range(30,35): # only do first 5 functions since rest may be noise
      max_index[ix, iy-30], peaks[ix, iy-30] = max(enumerate(corr_fn[:,ix,iy]), key=operator.itemgetter(1))

    #Perform fitting of decaying exponential to peaks
    init_guess = (10.0)
    if max_index[ix, 1] > max_index[ix, 0]:
      popt[ix], pcov = opt.curve_fit(fit.decaying_exp, (dt[max_index[ix,:]]), peaks[ix,:].ravel(), p0=init_guess)
    else:
      #print dt[max_index[ix,::-1]], peaks[ix,::-1]
      popt[ix], pcov = opt.curve_fit(fit.growing_exp, (dt[max_index[ix,::-1]]), peaks[ix,::-1].ravel(), p0=init_guess)
  
    #Need to take care of cases where there is no flow. Test if max_index for first two peaks are
    #both at zero. If so just fit the central peak with a decaying exponential. 
    if max_index[ix, 0] == max_index[ix, 1]:
      popt[ix] = 0

    #Need to check monotonicity of max indices. If abs(max_index) is not monotonically increasing, this usually means
    #that there is no flow and that the above method cannot be used to calculate the correlation time. Just ignore
    #these cases by setting correlation time  to zero here.
    if (strictly_increasing(max_index[ix,:]) == False and strictly_increasing(max_index[ix,::-1]) == False):
      corr_fn[:,ix,30] = corr_fn[:,ix,30]/max(corr_fn[:,ix,30])
      init_guess = (1.0, 1.0)
      tau_and_omega, pcov = opt.curve_fit(fit.osc_exp, (dt[nt/2-100:nt/2+100]), (corr_fn[nt/2-100:nt/2+100,ix,30]).ravel(), p0=init_guess)
      popt[ix] = tau_and_omega[0]
      print 'test: ', ix, tau_and_omega

  xvalue = 18
  print popt[xvalue]
  plt.clf()
  plt.plot(dt*1e6*amin/vth, corr_fn[:,xvalue,30:35])
  plt.hold(True)
  plt.plot(dt[max_index[xvalue,:]]*1e6*amin/vth, peaks[xvalue,:], 'ro')
  plt.hold(True)
  #p1 = plt.plot(dt*1e6*amin/vth, np.exp(-(dt / popt[xvalue])**2)*np.cos(tau_and_omega[1]*dt), 'b', lw=2)
  p1 = plt.plot(dt[nt/2:nt/2+100]*1e6*amin/vth, np.exp(-dt[nt/2:nt/2+100] / popt[xvalue]), 'b', lw=2)
  plt.xlabel(r'$\Delta t (\mu s)})$', fontsize=25)
  plt.ylabel(r'$C_{\Delta y}(\Delta t)$', fontsize=25)
  plt.legend(p1, [r'$\exp[-|\Delta t_{peak} / \tau_c]$'])
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.savefig('analysis/time_fit.pdf')

  #return correlation time in seconds
  return abs(popt)*1e6*amin/vth

#Function which checks monotonicity. Returns True or False.
def strictly_increasing(L):
      return all(x<=y for x, y in zip(L, L[1:]))

#Function which takes in density fluctuations and outputs the correlation time as a 
#function of the minor radius
def tau_vs_radius(ntot, t):
  shape = ntot.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
  ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  dypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  corr_fn = np.empty([nt, nx, ny], dtype=float); corr_fn[:,:,:] = 0.0
  count = np.empty([ny], dtype=int); count[:] = 0;

  ntot = np.fft.fft(ntot, axis=0) # (w, x, y) and complex

  for ix in range(nx):
    for iy1 in range(ny):
      for iy2 in range(iy1,ny):
        #separation in y
        dy = abs(ypts[iy1] - ypts[iy2])
        # calculate index based on y separation (assume ny bins)
        # y index: -dy_max, ..., 0, ..., dy_max
        y_index = int((dy - min(dypts))/(dypts[1] - dypts[0]))

        #Use WK theorem to calculate corr fn in time
        corr_fn[:, ix, y_index] += wk_thm_1d(ntot[:, ix, iy1], ntot[:, ix, iy2]) 
        # increment a count variable which will be used to normalize y bins 
        count[y_index] += 1 

        #Repeat analysis for negative separation
        dy = -dy
        y_index = int((dy - min(dypts))/(dypts[1] - dypts[0]))
        corr_fn[:, ix, y_index] += wk_thm_1d(ntot[:, ix, iy2], ntot[:, ix, iy1])
        count[y_index] += 1 

    #Normalize by count
    for iy in range(ny):
      if count[iy] > 0:
        corr_fn[:, ix, iy] = corr_fn[:, ix, iy] / count[iy]

    #Normalize correlation function by the max
    corr_fn[:,ix,:] = corr_fn[:,ix,:] / np.max(corr_fn[:,ix,:])
    #shift time zeros to centre
    corr_fn[:,ix,:] = np.fft.fftshift(corr_fn[:,ix,:], axes=[0])

  #Anthony plots
  #ofile = open('analysis/corr_fn.pkl', 'wb')
  #pkl.dump([np.linspace(-t_reg[750]+t_reg[0], t_reg[750]-t_reg[0], 750)*1e6*amin/vth,corr_fn], ofile)

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
print 'Starting interpolation...'
t_reg = np.linspace(min(t), max(t), len(t))
shape = density.shape
ntot_reg = np.empty([shape[0], shape[2], shape[1], shape[3]])
for i in range(shape[1]):
  for j in range(shape[2]):
    for k in range(shape[3]):
      f = interp.interp1d(t, density[:, i, j, k])
      ntot_reg[:, j, i, k] = f(t_reg) #perform a transpose here: ntot(t,kx,ky,theta,ri)
print 'Finished interpolation...'

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
  avg_fit_par[-1,:] = [10,10,1,0.1]
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
  plt.contourf(xpts[44:84], ypts[20:40], np.transpose(avg_corr[44:84,20:40]), 11, levels=np.linspace(-1, 1, 11))
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
  plt.contourf(xpts[44:84], ypts[20:40], np.transpose(data_fitted.reshape(nx,ny-1)[44:84,20:40]), 11, levels=np.linspace(-1, 1, 11))
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
  shape = ntot_reg.shape
  nt = shape[0]
  nx = shape[1]
  nky = shape[2]
  ny = (nky-1)*2

  real_space_density = np.empty([nt,nx,ny],dtype=float)
  for it in range(nt):
    real_space_density[it,:,:] = np.fft.irfft2(real_to_complex_2d(ntot_reg[it,:,:,:]), axes=[0,1])

  #Export film
  print 'Exporting film...'
  xpts = np.linspace(0, 2*np.pi/kx[1], nx)*rhoref # change to meters
  ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) # change to meters and poloidal plane
  film.real_space_film_2d(xpts, ypts, real_space_density[:,:,:], 'density')

