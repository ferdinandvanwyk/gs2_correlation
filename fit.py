############################
# gs2_correlation_analysis #
#    Ferdinand van Wyk     #
############################

###############################################################################
# This file is part of gs2_correlation_analysis.
#
# gs2_correlation_analysis is free software: you can redistribute it and/or 
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gs2_correlation_analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gs2_correlation_analysis.  
# If not, see <http://www.gnu.org/licenses/>.
###############################################################################

# This file contains various fitting functions called during the main 
# correlation analysis

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

#Model function to be fitted to data, as defined in Anthony's papers
def tilted_gauss(xdata_tuple, lx, ly, kx, ky):
    (x,y) = xdata_tuple
    exp_term = np.exp(- (x/lx)**2 - (y/ly)**2 )
    cos_term = np.cos(kx*x + ky*y)
    fit_fn =  exp_term*cos_term
    # fitting function only works on 1D data, reshape later to plot
    return fit_fn.ravel()

#Decaying exponential for time correlation with positive flow
def decaying_exp(t, tau_c):
    exp_fn = np.exp(- abs(t) / tau_c)
    # fitting function only works on 1D data, reshape later to plot
    return exp_fn.ravel() 

#Growing exponential for time correlation with negative flow
def growing_exp(t, tau_c):
    exp_fn = np.exp(t / tau_c)
    # fitting function only works on 1D data, reshape later to plot
    return exp_fn.ravel()

#Growing exponential for time correlation with negative flow
def osc_exp(t, tau_c, omega):
    fit_fn = np.exp(- (t / tau_c)**2) * np.cos(omega * t)
    # fitting function only works on 1D data, reshape later to plot
    return fit_fn.ravel() 

# General plotting function called by the various fitting methods to plot the 
# correlation function along with its fit. This is done for every radial value.
# plot_type can be either 'decaying_exp', 'growing_exp', or 'osc_exp', to plot 
# either an exponential fit to the correlation function peaks, or an 
# oscillating Gaussian to the central peak if there is no flow in which case 
# tau = [tau, omega].
def plot_fit(ix, dt, corr_fn, max_index, peaks, mid_idx, tau, 
             plot_type, amin, vth):
    nt = len(dt)
    plt.clf()
    plt.plot(dt*1e6*amin/vth, corr_fn[:,mid_idx:mid_idx+5])
    plt.hold(True)
    plt.plot(dt[max_index[:]]*1e6*amin/vth, peaks[:], 'ro')
    plt.hold(True)

    if plot_type == 'decaying_exp':
        p1 = plt.plot(dt[nt/2:nt/2+100]*1e6*amin/vth, 
                      np.exp(-dt[nt/2:nt/2+100] / tau), 'b', lw=2)
        plt.legend(p1, [r'$\exp[-|\Delta t_{peak} / \tau_c|]$'])
    if plot_type == 'growing_exp':
        p1 = plt.plot(dt[nt/2-100:nt/2]*1e6*amin/vth, 
                      np.exp(dt[nt/2-100:nt/2] / tau), 'b', lw=2)
        plt.legend(p1, [r'$\exp[|\Delta t_{peak} / \tau_c|]$'])
    if plot_type == 'osc_exp':
        p1 = plt.plot(dt*1e6*amin/vth, 
                      np.exp(-(dt / tau[0])**2)*np.cos(tau[1]*dt), 'b', lw=2)
        plt.legend(p1, [r'$\exp[- (\Delta t_{peak} / \tau_c)^2] '
                         '\cos(\omega \Delta t) $'])

    plt.xlabel(r'$\Delta t (\mu s)})$', fontsize=25)
    plt.ylabel(r'$C_{\Delta y}(\Delta t)$', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.savefig('analysis/corr_fns/time_fit_ix_' + str(ix) + '.pdf')

