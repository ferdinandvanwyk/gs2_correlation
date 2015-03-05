#########################
#   gs2_correlation     #
#   Ferdinand van Wyk   #
#########################

###############################################################################
# This file is part of gs2_correlation.
#
# gs2_correlation_analysis is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gs2_correlation is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gs2_correlation.
# If not, see <http://www.gnu.org/licenses/>.
###############################################################################

"""
.. module:: middle_box_analysis
   :platform: Unix, OSX
   :synopsis: Class which analyzes the middle of the box

.. moduleauthor:: Ferdinand van Wyk <ferdinandvwyk@gmail.com>

"""

# Standard
import os
import sys
import gc #garbage collector
import configparser
import logging
import operator #enumerate list
import multiprocessing

# Third Party
import numpy as np
from scipy.io import netcdf
import scipy.interpolate as interp
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
import seaborn as sns

# Local
import gs2_correlation.fitting_functions as fit
from gs2_correlation.simulation import Simulation

class MiddleBox(Simulation):
    """
    Class containing methods which assume analysis is done on the middle part
    of the domain.

    Analysis done on this domain can no longer assume periodicity and so all
    analysis has to be done in real space, i.e. without taking advantage of
    Fourier space calculations, most importantly the Wiener-Khinchin theorem.

    Attributes
    ----------

    r : array_like
        Radial coordinate *x*, centered at the major radius *rmaj*.
    z : array_like
        Poloidal coordinate *z* centered at 0.
    """

    def __init__(self, config_file):
        """
        Initialization consists of: 
        
        * Calling Simulation.__init__ to read config file, NetCDF file etc.
        * Calculating radial and poloidal coordinates r, z.
        * Using input parameter *box_size* to determine the index range to 
          perform the correlation analysis on.
        * Reduce extent of real space field using this index.
        * Recalculate some real space arrays such as x, y, dx, dy, etc.
        """
        Simulation.__init__(self, config_file)

        self.x = np.linspace(0, 2*np.pi/self.kx[1], self.nx)*self.rhoref
        self.y = np.linspace(0, 2*np.pi/self.ky[1], self.ny-1)*self.rhoref \
                             *np.tan(self.pitch_angle)
        
        # Calculate coords r, z
        self.r = self.x[:] - self.x[-1]/2 + self.rmaj
        self.z = self.y[:] - self.y[-1]/2 

        # Find index range
        r_min_idx, r_min = min(enumerate(abs(self.r - (self.box_size[0] + 
                                                       self.rmaj))), 
                               key=operator.itemgetter(1))
        r_box_idx = r_min_idx-int(self.nx/2) + 1

        z_min_idx, z_min = min(enumerate(abs(self.z - self.box_size[1])), 
                               key=operator.itemgetter(1))
        z_box_idx = z_min_idx-int(self.ny/2) + 1

        # Reduce extent
        self.r = self.r[int(self.nx/2)-r_box_idx+1:int(self.nx/2)+r_box_idx]
        self.z = self.z[int(self.ny/2)-z_box_idx+1:int(self.ny/2)+z_box_idx]
        self.field_real_space = self.field_real_space[
                :,int(self.nx/2)-r_box_idx+1:int(self.nx/2)+r_box_idx,
                int(self.ny/2)-z_box_idx+1:int(self.ny/2)+z_box_idx]

        # Recalculate real space arrays
        self.nx = len(self.r)
        self.ny = len(self.z)

        self.x = np.linspace(0, self.r[-1] - self.r[0], self.nx)
        self.y = np.linspace(0, 2*self.z[-1], self.ny)
        self.dx = np.linspace(-self.x[-1], self.x[-1], self.nx)
        self.dy = np.linspace(-self.y[-1], self.y[-1], self.ny)
        self.fit_dx = self.dx
        self.fit_dy = self.dy
        self.fit_dx_mesh, self.fit_dy_mesh = np.meshgrid(self.fit_dx, self.fit_dy)
        self.fit_dx_mesh = np.transpose(self.fit_dx_mesh)
        self.fit_dy_mesh = np.transpose(self.fit_dy_mesh)

    def perp_analysis(self):
        """
        Performs a perpendicular correlation analysis on the field.

        Notes
        -----

        * Uses a 2D Wiener-Khinchin theorem to calculate the correlation
          function.
        * Splits correlation function into time slices and fits each time
          slice with a tilted Gaussian using the perp_fit function.
        * The fit parameters for the previous time slice is used as the initial
          guess for the next time slice.
        """

        logging.info('Start perpendicular correlation analysis...')

        if 'perp' not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/perp')

        self.calculate_perp_corr()
        self.perp_fit_params = np.empty([self.nt_slices, 4], dtype=float)

        for it in range(self.nt_slices):
            self.perp_corr_fit(it)

        np.savetxt(self.out_dir + '/perp/perp_fit_params.csv', (self.perp_fit_params),
                   delimiter=',', fmt='%1.3f')

        self.perp_plots()
        self.perp_analysis_summary()

        logging.info('Finished perpendicular correlation analysis.')

    def calculate_perp_corr(self):
        """
        Calculates the perpendicular correlation function from the real space
        field.
        """
        logging.info("Calculating perpendicular correlation function...")

        nr = len(self.r)
        nz = len(self.z)
        self.perp_corr = np.empty([self.nt, nr, nz], dtype=float)
        for it in range(self.nt):
            self.perp_corr[it,:,:] = sig.fftconvolve(self.field_real_space[it,:,:], 
                                                     self.field_real_space[it,::-1,::-1],
                                                     mode='same')
            self.perp_corr[it,:,:] = (self.perp_corr[it,:,:] /  
                                        np.max(self.perp_corr[it,:,:]))

        logging.info("Findished calculating perpendicular correlation " 
                      "function...")

    def perp_corr_fit(self, it):
        """
        Fits tilted Gaussian to perpendicular correlation function.

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being fitted.
        """
        logging.info('Fitting time window %d'%it)
        
        # Average corr_fn over time
        avg_corr = np.mean(self.perp_corr, axis=0)

        if len(self.perp_guess) == 4:
            popt, pcov = opt.curve_fit(fit.tilted_gauss, (self.fit_dx_mesh,
                                                          self.fit_dy_mesh),
                                       avg_corr.ravel(), p0=self.perp_guess)

            self.perp_fit_params[it, :] = popt

        elif len(self.perp_guess) == 3:
            popt, pcov = opt.curve_fit(fit.tilted_gauss_ky_fixed, 
                                                            (self.fit_dx_mesh,
                                                             self.fit_dy_mesh),
                                       avg_corr.ravel(), p0=self.perp_guess)

            self.perp_fit_params[it, :] = np.append(popt, 2*np.pi/popt[1])

        self.perp_guess = popt

    def perp_plots(self):
        """
        Function which plots various things relevant to perpendicular analysis.

        * Time-averaged correlation function
        * Tilted Gaussian using time-averaged fitting parameters
        * The above two graphs overlayed
        """
        logging.info("Writing perp_analysis plots...")

        sns.set_style('darkgrid', {'axes.axisbelow':False, 'legend.frameon': True})
        #Time averaged correlation
        plt.clf()
        avg_corr = np.mean(self.perp_corr, axis=0) # Average over time
        plt.contourf(self.fit_dx, self.fit_dy, np.transpose(avg_corr), 11,
                     levels=np.linspace(-1, 1, 11), cmap='coolwarm')
        plt.colorbar(ticks=np.linspace(-1, 1, 11))
        plt.xlabel(r'$\Delta x (m)$')
        plt.ylabel(r'$\Delta y (m)$')
        plt.savefig(self.out_dir + '/perp/time_avg_correlation.pdf')

        # Tilted Gaussian using time-averaged fitting parameters
        data_fitted = fit.tilted_gauss((self.fit_dx_mesh, self.fit_dy_mesh),
                                        *np.mean(self.perp_fit_params, axis=0))
        plt.clf()
        plt.contourf(self.fit_dx, self.fit_dy,
                     np.transpose(data_fitted.reshape(len(self.fit_dx),
                                                      len(self.fit_dy))),
                                  11, levels=np.linspace(-1, 1, 11), cmap='coolwarm')
        plt.title('$C_{fit}(\Delta x, \Delta y)$')
        plt.colorbar(ticks=np.linspace(-1, 1, 11))
        plt.xlabel(r'$\Delta x (m)$')
        plt.ylabel(r'$\Delta y (m)$')
        plt.savefig(self.out_dir + '/perp/perp_corr_fit.pdf')

        # Avg correlation and fitted function overlayed
        plt.clf()
        plt.contourf(self.fit_dx, self.fit_dy, np.transpose(avg_corr), 10,
                     levels=np.linspace(-1, 1, 11), cmap='coolwarm')
        plt.colorbar(ticks=np.linspace(-1, 1, 11))
        plt.contour(self.fit_dx, self.fit_dy,
                     np.transpose(data_fitted.reshape(len(self.fit_dx),
                                                      len(self.fit_dy))),
                                  11, levels=np.linspace(-1, 1, 11), colors='k')
        plt.xlabel(r'$\Delta x (m)$')
        plt.ylabel(r'$\Delta y (m)$')
        plt.savefig(self.out_dir + '/perp/perp_fit_comparison.pdf')

        logging.info("Finished writing perp_analysis plots...")














