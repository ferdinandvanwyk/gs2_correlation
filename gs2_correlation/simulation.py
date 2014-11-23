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
    perp_corr : array_like
        Perpendicular correlation function calculated from the field array.
    kx : array_like
        Values of the kx grid in the following order: 0,...,kx_max,-kx_max,...
        kx_min.
    ky : array_like
        Values of the ky grid.
    t : array_like
        Values of the time grid.
    x : array_like
        Values of the real space x (radial) grid.
    y : array_like
        Values of the real space y (poloidal) grid. This has been transformed 
        from the toroidal plane to the poloidal plane by using the pitch-angle
        of the magnetic field lines.
    nkx : int
        Number of kx values. Also the number of real space x points.
    nky : int
        Number of ky values.
    ny : int
        Number of real space y. This is ny = 2*(nky - 1). 
    nt : int
        Number of time points.

    """

    def __init__(self, conf):
        """Initialize by object using information from configuration file.

        Initialization does the following, depending on the parameters in the 
        configuration file:

        * Calculates sizes of kx, ky, x, and y.
        * Reads data from the NetCDF file.
        * Interpolates onto a regular time grid.
        * Zeros out BES scales.
        * Zeros out ZF scales.
        """

        self.read_netcdf(conf)

        self.nt = len(self.t)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.ny = 2*(self.nky - 1)

        self.x = np.linspace(-2*np.pi/self.kx[1], 2*np.pi/self.kx[1], 
                             self.nkx)*conf.rhoref
        self.y = np.linspace(-2*np.pi/self.ky[1], 2*np.pi/self.ky[1], 
                             self.ny)*conf.rhoref*np.tan(conf.pitch_angle)

        if conf.interpolate:
            self.interpolate(conf)

        if conf.zero_bes_scales:
            self.zero_bes_scales(conf)

        if conf.zero_zf_scales:
            self.zero_zf_scales(conf)

        self.to_complex()

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
        logging.info('Started interpolating onto a regular grid...')

        t_reg = np.linspace(min(self.t), max(self.t), len(self.t))
        for i in range(len(self.kx)):
            for j in range(len(self.ky)):
                for k in range(2):
                    f = interp.interp1d(self.t, self.field[:, i, j, k])
                    self.field[:, i, j, k] = f(t_reg)
        self.t = t_reg

        logging.info('Finished interpolating onto a regular grid.')

    def zero_bes_scales(self, conf):
        """Sets modes larger than the BES to zero.

        The BES is approximately 160x80mm(rad x pol), so we would set kx < 0.25
        and ky < 0.5 to zero, since k = 2 pi / L. 

        Parameters
        ----------

        conf : object
            This is an instance of the Configuration class which contains 
            information read in from the configuration file, such as NetCDF 
            filename, location, field to read in etc.
        """
        for ikx in range(len(self.kx)):
            for iky in range(len(self.ky)):
                # Roughly the size of BES (160x80mm)
                if abs(self.kx[ikx]) < 0.25 and self.ky[iky] < 0.5: 
                    self.field[:,ikx,iky,:] = 0.0

    def zero_zf_scales(self, conf):
        """Sets ZF (ky = 0) modes to zero.

        Parameters
        ----------

        conf : object
            This is an instance of the Configuration class which contains 
            information read in from the configuration file, such as NetCDF 
            filename, location, field to read in etc.
        """
        self.field[:,:,0,:] = 0.0

    def to_complex(self):
        """Converts field to a complex array.

        Field is in the following format: field[t, kx, ky, ri] where ri 
        represents a dimension of length 2. 

        * ri = 0 - Real part of the field. 
        * ri = 1 - Imaginary part of the field. 
        """
        self.field = self.field[:,:,:,0] + 1j*self.field[:,:,:,1] 
    
    def perp_analysis(self, conf):

        logging.info('Start perpendicular correlation analysis...')

        self.wk_2d(conf) 

        logging.info('Finished perpendicular correlation analysis.')
        
    def wk_2d(self):
        """Calculates perpendicular correlation function for each time step.

        Using the Wiener-Khinchin theorem, the 2D perpendicular correlation 
        function os calculated for each time step. The zeros are then shifted 
        to the centre of the domain and the correlation function is normalized.
        The result is saved as a new Simulation attribute: perp_corr.

        Notes
        -----
        The Wiener-Khinchin theorem states the following:

        .. math:: C(\Delta x, \Delta y) = IFFT2[|f(k_x, k_y)|^2]

        where C is the correlation function, *f* is the field, and IFFT2 is the 
        2D inverse Fourier transform.

        """

        logging.info("Performing 2D WK theorem on field...")

        # ny-1 below since to ensure odd number of y points and that zero is in
        # the middle of the y domain.
        self.perp_corr = np.empty([self.nt, self.nkx, self.ny-1], dtype=float) 
        print(self.perp_corr.shape, self.field.shape)
        for it in range(self.nt):
            sq = np.abs(self.field[it,:,:])**2  
            self.perp_corr[it,:,:] = np.fft.irfft2(sq, s=[self.nkx, self.ny-1])
            self.perp_corr[it,:,:] = np.fft.fftshift(self.perp_corr[it,:,:])
            self.perp_corr[it,:,:] = (self.perp_corr[it,:,:] / 
                                      np.max(self.perp_corr[it,:,:]))

        logging.info("Finished 2D WK theorem.")






















