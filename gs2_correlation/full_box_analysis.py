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
.. module:: full_box_analysis
   :platform: Unix, OSX
   :synopsis: Class describing the GS2 simulation.

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

class FullBox(Simulation):
    """
    Class containing methods which assume analysis is done on the full domain.
    """

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

        self.wk_2d()
        self.perp_fit_params = np.empty([self.nt_slices, 4], dtype=float)

        for it in range(self.nt_slices):
            self.perp_fit(it)
            self.perp_guess = self.perp_fit_params[it,:]

        np.savetxt(self.out_dir + '/perp/perp_fit_params.csv', (self.perp_fit_params),
                   delimiter=',', fmt='%1.3f')

        self.perp_plots()
        self.perp_analysis_summary()

        logging.info('Finished perpendicular correlation analysis.')














