#This file contains various fitting functions called during the main correlation analysis
import os, sys
import numpy as np

#Model function to be fitted to data, as defined in Anthony's papers
def tilted_gauss((x,y), lx, ly, ky, Th):
  exp_term = np.exp(- (x/lx)**2 - ((y + Th*x) / ly)**2 )
  cos_term = np.cos(ky*Th*x + ky*y)
  fit_fn =  exp_term*cos_term
  return fit_fn.ravel() # fitting function only works on 1D data, reshape later to plot
