#This program takes the output from the correlation calculation program 
#and plots and fits the data to produce correlation times and lengths. It
#is called using the following command:
#
#     python correlation_plot.py <netcdf filename> 

import os, sys
import time
import operator
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
from scipy.io import netcdf


# Read command line argument specifying location of NetCDF file
in_file =  str(sys.argv[1])

#Model function to be fitted to data, as defined in Anthony's papers
def tilted_gauss((x,y), p, lx, ly, ky, Th):
  exp_term = np.exp(- (x/lx)**2 - ((y + Th*x) / ly)**2 )
  cos_term = np.cos(ky*Th*x + ky*y)
  fit_fn =  p + exp_term*cos_term
  return fit_fn.ravel() # fitting function only works on 1D data, reshape later to plot

#Exponential decay function for time correlation
def exp_decay((t), tau, omega):
  exp_term = np.exp(- (np.abs(t)/tau) )
  cos_term = np.cos(omega*t)
  return exp_term*cos_term.ravel() # fitting function only works on 1D data, reshape later to plot

#############
# Main Code #
#############

ncfile = netcdf.netcdf_file(in_file, 'r')
corr = ncfile.variables['correlation'][:,:,:]
th = ncfile.variables['theta'][:]
dx = ncfile.variables['dx'][:]
dy = ncfile.variables['dy'][:]
dt = ncfile.variables['dt'][:]

####################
# Time Correlation #
####################

dx0 = 42; dy0 = 31; dt0 = 500;

plt.contourf(dt, dy, np.transpose(corr[:,dx0,:]))
plt.show()

init_guess = (10, 0.001)
popt, pcov = opt.curve_fit(exp_decay, (dt), corr[:,dx0,dy0]/np.max(corr[:,dx0,dy0]), p0=init_guess)
data_fitted = exp_decay((dt), *popt)
print popt

plt.plot(dt, corr[:,dx0,dy0]/np.max(corr[:,dx0,dy0]))
plt.hold(True)
plt.plot(dt, data_fitted)
plt.show()

####################
# Perp Correlation #
####################


