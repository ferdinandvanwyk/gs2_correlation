#This file contains various fitting functions called during the main correlation analysis
import os, sys
import numpy as np

#Model function to be fitted to data, as defined in Anthony's papers
def tilted_gauss((x,y), lx, ly, kx, ky):
  exp_term = np.exp(- (x/lx)**2 - (y/ly)**2 )
  cos_term = np.cos(kx*x + ky*y)
  fit_fn =  exp_term*cos_term
  return fit_fn.ravel() # fitting function only works on 1D data, reshape later to plot

#Decaying exponential for time correlation with positive flow
def decaying_exp((t), tau_c):
  exp_fn = np.exp(- abs(t) / tau_c)
  return exp_fn.ravel() # fitting function only works on 1D data, reshape later to plot

#Growing exponential for time correlation with negative flow
def growing_exp((t), tau_c):
  exp_fn = np.exp(t / tau_c)
  return exp_fn.ravel() # fitting function only works on 1D data, reshape later to plot

#Growing exponential for time correlation with negative flow
def osc_exp((t), tau_c, omega):
  fit_fn = np.exp(- (t / tau_c)**2) * np.cos(omega * t)
  return fit_fn.ravel() # fitting function only works on 1D data, reshape later to plot
