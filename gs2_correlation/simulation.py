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
.. module:: Simulation
   :platform: Unix, OSX
   :synopsis: Class describing the GS2 simulation.
                                                                                
.. moduleauthor:: Ferdinand van Wyk <ferdinandvwyk@gmail.com>                   
                                                                                
"""

# Standard
import os
import sys
import gc #garbage collector
import logging

# Third Party
import numpy as np
from scipy.io import netcdf
import scipy.interpolate as interp

# Local

class Simulation(object):
    """Class containing all simulation information.

    The class mainly reads from the simulation NetCDF file and operates on the
    field specified in the configuration file, such as performing correlations,
    FFTs, plotting, making films etc.

    Attributes
    ----------

    field : array_like
        Field read in from the NetCDF file.
    kx : array_like
        Values of the kx grid in the following order: 0,...,kx_max,-kx_max,...
        kx_min.
    ky : array_like
        Values of the ky grid.
    t : array_like
        Values of the time grid.

    """
    
    def read_netcdf(self, conf):
        """Read array from NetCDF file.

        Read array specified in configuration file as 'cdf_field'. Function 
        uses information from the configuration object passed to it. 

        Parameters
        ----------

        conf : object
            This is an instance of the Configuration class which contains 
            information read in from the configuration file, such as NetCDF 
            filename, location, field to read in etc.

        """
        logging.info('Start reading from NetCDf file...')

        # mmap=False does not read directly from cdf file. Copies are created.
        # This prevents seg faults when cdf file is closed after function exits
        ncfile = netcdf.netcdf_file(conf.in_file, 'r', mmap=False)

        # NetCDF order is [t, species, ky, kx, theta, r]
        self.field = ncfile.variables[conf.in_field][:,conf.spec_idx,:,:,conf.theta_idx,:]
        self.field = np.squeeze(self.field) 
        self.field = np.swapaxes(self.field, 1, 2)

        self.kx = ncfile.variables['kx'][:]
        self.ky = ncfile.variables['ky'][:]
        self.t = ncfile.variables['t'][:]

        logging.info('Finished reading from NetCDf file.')

    def interpolate(self, conf):
        """Interpolates in time onto a regular grid

        Depending on whether the user specified to interpolate, the time grid
        is interpolated into a regular grid. This is required in order to do 
        FFTs in time. Interpolation is done by default if not specified.

        Parameters
        ----------

        conf : object
            This is an instance of the Configuration class which contains 
            information read in from the configuration file, such as NetCDF 
            filename, location, field to read in etc.
        """
        if conf.interpolate:
            t_reg = np.linspace(min(self.t), max(self.t), len(self.t))
            for i in range(len(self.kx)):
                for j in range(len(self.ky)):
                    for k in range(2):
                        f = interp.interp1d(self.t, self.field[:, i, j, k])
                        self.field[:, i, j, k] = f(t_reg)
            self.t = t_reg























