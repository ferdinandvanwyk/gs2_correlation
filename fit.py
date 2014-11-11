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
import scipy.optimize as opt
import scipy.signal as sig

######################
# Fitting Procedures #
######################

# Fit the 2D correlation function with a tilted Gaussian and extract the 
# fitting parameters
def perp_fit(corr_fn, xpts, ypts, guess):
    shape = corr_fn.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
    x,y = np.meshgrid(xpts, ypts)
    x = np.transpose(x); y = np.transpose(y)

    # Average corr fn over time
    avg_corr = np.empty([nx, ny], dtype=float)
    avg_corr = np.mean(corr_fn, axis=0)

    # lx, ly, kx, ky
    popt, pcov = opt.curve_fit(tilted_gauss, (x, y), avg_corr.ravel(), 
                               p0=guess)

    #plt.contourf(xpts, ypts, np.transpose(avg_corr))
    #plt.hold(True)
    #data_fitted = fit.tilted_gauss((x, y), *popt)
    #plt.contour(xpts, ypts, np.transpose(data_fitted.reshape(nx,ny)), 8, 
    #            colors='w')
    #plt.show()

    return popt

#Fit the peaks of the correlation functions of different dy with decaying exp
def time_fit(corr_fn, t, out_dir):
    shape = corr_fn.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];

    #define delta t range
    dt = np.linspace(-max(t)+t[0], max(t)-t[0], nt)

    peaks = np.empty([nx, 5]); peaks[:,:] = 0.0;
    max_index = np.empty([nx, 5], dtype=int);
    popt = np.empty([nx], dtype=float)
    mid_idx = 61
    os.system("mkdir " + out_dir + "/corr_fns")
    for ix in range(0,nx): # loop over all radial points
        #only read fit first 5 since rest may be noise
        for iy in range(mid_idx,mid_idx+5):
            max_index[ix, iy-mid_idx], peaks[ix, iy-mid_idx] = \
                max(enumerate(corr_fn[:,ix,iy]), key=operator.itemgetter(1))

    if (strictly_increasing(max_index[ix,:]) == True or 
        strictly_increasing(max_index[ix,::-1]) == True):
        # Perform fitting of decaying exponential to peaks
        init_guess = (10.0)
        if max_index[ix, 4] > max_index[ix, 0]:
            popt[ix], pcov = opt.curve_fit(decaying_exp, 
                    (dt[max_index[ix,:]]), peaks[ix,:].ravel(), p0=init_guess)
            plot_fit(ix, dt, corr_fn[:,ix,:], max_index[ix,:], peaks[ix,:], 
                    mid_idx, popt[ix], 'decaying_exp', amin, vth)
            diag_file.write("Index " + str(ix) + " was fitted with decaying" 
                    "exponential. tau = " + str(popt[ix]) + "\n")
        else:
            popt[ix], pcov = opt.curve_fit(fit.growing_exp, 
                    (dt[max_index[ix,::-1]]), peaks[ix,::-1].ravel(), 
                    p0=init_guess)
            fit.plot_fit(ix, dt, corr_fn[:,ix,:], max_index[ix, :], peaks[ix,:],
                    mid_idx, popt[ix], 'growing_exp', amin, vth)
            diag_file.write("Index " + str(ix) + " was fitted with growing "
                    "exponential. tau = " + str(popt[ix]) + "\n")
    else:
        # If abs(max_index) is not monotonically increasing, this usually means
        # that there is no flow and that the above method cannot be used to 
        # calculate the correlation time. Try fitting a decaying oscillating 
        # exponential to the central peak.
        corr_fn[:,ix,mid_idx] = corr_fn[:,ix,mid_idx]/max(corr_fn[:,ix,mid_idx])
        init_guess = (10.0, 1.0)
        tau_and_omega, pcov = opt.curve_fit(fit.osc_exp, (dt[nt/2-100:nt/2+100]), 
                (corr_fn[nt/2-100:nt/2+100,ix,mid_idx]).ravel(), p0=init_guess)
        popt[ix] = tau_and_omega[0]
        fit.plot_fit(ix, dt, corr_fn[:,ix,:], max_index[ix,:], peaks[ix,:], 
                     mid_idx, tau_and_omega, 'osc_exp', amin, vth)
        diag_file.write("Index " + str(ix) + " was fitted with an oscillating "
                    "Gaussian to the central peak. [tau, omega] = " 
                    + str(tau_and_omega) + "\n")

    # Return correlation time in seconds
    return abs(popt)*1e6*amin/vth 

# Function which checks monotonicity. Returns True or False.
def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

# Function which takes in density fluctuations and outputs the correlation time
# as a function of the minor radius
def tau_vs_radius(ntot, t, out_dir, diag_file):
    shape = ntot.shape; nt = shape[0]; nx = shape[1]; ny = shape[2];
    # change to meters and poloidal plane
    ypts = np.linspace(0, 2*np.pi/ky[1], ny)*rhoref*np.tan(pitch_angle) 
    dypts = np.linspace(-2*np.pi/ky[1], 2*np.pi/ky[1], 
                        ny)*rhoref*np.tan(pitch_angle)
    corr_fn = np.empty([2*nt-1, nx, 2*ny-1], dtype=float); corr_fn[:,:,:] = 0.0

    for ix in range(nx):
        print('ix = ', ix, ' of ', nx)
        corr_fn[:,ix,:] = sig.correlate(ntot[:,ix,:], ntot[:,ix,:])

        # Normalize correlation function by the max
        corr_fn[:,ix,:] = corr_fn[:,ix,:] / np.max(corr_fn[:,ix,:])

    # Fit exponential decay to peaks of correlation function in dt for few dy's
    tau = time_fit(corr_fn, t, out_dir, diag_file) #tau in seconds

    return tau

###########################
# Model Fitting Functions #
###########################

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

