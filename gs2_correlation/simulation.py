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
import configparser
import logging
import operator #enumerate list
import multiprocessing
import warnings

# Third Party
import numpy as np
from scipy.io import netcdf
import scipy.interpolate as interp
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import f90nml as nml
import lmfit as lm
import pyfftw
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus']=False
pal = sns.color_palette('deep')  

# Local
import gs2_correlation.fitting_functions as fit
import gs2_correlation.plot_style as plot_style

class Simulation(object):
    """
    Class containing all simulation information.

    The class mainly reads from the simulation NetCDF file and operates on the
    field specified in the configuration file, such as performing correlations,
    FFTs, plotting, making films etc.

    An exhaustive list of instance variables is provided separately in the 
    documentation since it is very long.
    """

    def __init__(self, config_file):
        """
        Initialized by object using information from configuration file.

        Initialization does the following, depending on the parameters in the
        configuration file:

        * Calculates sizes of kx, ky, x, and y.
        * Reads data from the NetCDF file.
        * Interpolates onto a regular time grid.
        * Zeros out BES scales.
        * Zeros out ZF scales.
        * Transforms to lab frame.
        * Calculates the real space field.
        * Reduce domain size to according to *box_size* if domain is specified
          as 'middle', otherwise do nothing.
        * Ensures real space field has odd points.

        Parameters
        ----------
        config_file : str
            Filename of configuration file and path if not in the same
            directory.

        Notes
        -----

        The configuration file should contain the following namelists:

        * 'general': information such as the analysis to be performed and
          where to find the NetCDF file.
        * 'perp' : parameters relating to the perpendicular correlation analysis.
        * 'time' : parameters relating to the time analysis.
        * 'par' : parameters relating to the parallel analysis.
        * 'normalization': normalization parameters for the simulation/experiment.
        * 'output': parameters which relate to the output produced by the code.

        The exact parameters read in are documented in the documentation.
        """

        self.config_file = config_file
        self.read_config()

        # Set plot options
        sns.set_context(self.seaborn_context)

        self.rho_star = self.rho_ref/self.amin

        self.read_input_file()
        self.read_geometry_file()
        self.read_netcdf()

        if self.out_dir not in os.listdir():
            os.system("mkdir " + self.out_dir)
        self.nt = len(self.t)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.nx = self.nkx
        self.ny = 2*(self.nky - 1)
        self.ntheta = self.field.shape[3]

        self.config_checks()

        self.R = self.geometry[:,1]*self.amin
        self.Z = self.geometry[:,2]*self.amin
        self.alpha = self.geometry[:,3]
        self.tor_phi = -self.geometry[:,3]
        self.dR_drho = self.geometry[:,4]*self.amin
        self.dZ_drho = self.geometry[:,5]*self.amin
        self.dalpha_drho = self.geometry[:,6]*self.amin
        theta_len = len(self.R)

        self.rmaj = self.R[int(theta_len/2)]*self.amin 
        self.pitch_angle = np.arctan(self.Z[int(theta_len/2)+1]/ \
                           (self.R[int(theta_len/2)] * \
                           self.tor_phi[int(theta_len/2)+1]))

        self.t = self.t*self.amin/self.vth
        self.x = np.linspace(0, 2*np.pi/self.kx[1], self.nx, endpoint=False)* \
                     self.rho_ref
        self.y = np.linspace(0, 2*np.pi/self.ky[1], self.ny, endpoint=False)* \
                     self.rho_ref * np.abs(np.tan(self.pitch_angle))* \
                     (self.rmaj/self.amin) * (self.drho_dpsi) * \
                     np.cos(self.pitch_angle)
        
        self.r_geo = self.input_file['theta_grid_parameters']['R_geo']*self.amin

        self.btor = self.bref*self.r_geo/self.R[int(self.ntheta/2)]
        self.bmag = np.sqrt(self.btor**2 + self.bpol**2)  

        self.field_to_complex()
        self.fourier_correction()

        if self.time_interpolate_bool or self.lab_frame:
            self.time_interpolate()
        self.nt_slices = int(self.nt/self.time_slice)

        if self.zero_bes_scales_bool:
            self.zero_bes_scales()

        if self.zero_zf_scales_bool:
            self.zero_zf_scales()

        if self.lab_frame:
            self.to_lab_frame()

        self.field_to_real_space()

        if self.domain == 'middle':
            self.domain_reduce()

        self.field_odd_pts()

        if self.analysis != 'par':
            self.field_real_space = self.field_real_space[:,:,:,0]

    def read_config(self):
        """
        Reads analysis and normalization parameters from self.config_file.

        The full list of possible configuration parameters is provided 
        separately in the documentation since it is quite long and are 
        technically not parameters of this function.
        """
        logging.info('Started read_config...')

        config_parse = configparser.ConfigParser()
        config_parse.read(self.config_file)

        ##########################
        # Normalization Namelist #
        ##########################

        self.amin = float(config_parse['normalization']['a_minor'])
        self.vth = float(config_parse['normalization']['vth_ref'])
        self.rho_ref = float(config_parse['normalization']['rho_ref'])
        self.bref = float(config_parse['normalization']['bref'])
        self.nref = float(config_parse.get('normalization', 'nref', fallback=1))
        self.tref = float(config_parse.get('normalization', 'tref', fallback=1))
        self.omega = float(config_parse.get('normalization', 'omega', fallback=0))
        self.dpsi_da = float(config_parse.get('normalization', 'dpsi_da', 
                                              fallback=0))

        ####################
        # General Namelist #
        ####################

        self.domain = config_parse.get('general', 'domain', 
                                        fallback='full')

        if self.domain == 'full':
            self.out_dir = 'full_analysis'
        elif self.domain == 'middle':
            self.out_dir = 'middle_analysis'

        self.out_dir = config_parse.get('general', 'out_dir',
                                         fallback=self.out_dir)

        self.file_ext = config_parse.get('general', 'file_ext',
                                         fallback='.cdf')

        # Automatically find .out.nc file if only directory specified
        self.run_folder = str(config_parse['general']['run_folder'])
        self.cdf_file = config_parse.get('general', 'cdf_file', fallback='None')
        if self.cdf_file == "None":
            dir_files = os.listdir(self.run_folder)
            found = False
            for s in dir_files:
                if s.find(self.file_ext) != -1:
                    self.cdf_file = self.run_folder + s
                    found = True
                    break

            if not found:
                raise NameError('No file found ending in ' + self.file_ext)

        self.g_file = config_parse.get('general', 'g_file', fallback='None')
        if self.g_file == 'None':
            dir_files = os.listdir(self.run_folder)
            found = False
            for s in dir_files:
                if s.find('.g') != -1:
                    self.g_file = self.run_folder + s
                    found = True
                    break

            if not found:
                raise NameError('No file found ending in .g')

        self.in_file = config_parse.get('general', 'in_file', fallback='None')
        if self.in_file == 'None':
            dir_files = os.listdir(self.run_folder)
            found = False
            for s in dir_files:
                if s.find('.in') != -1:
                    self.in_file = self.run_folder + s
                    found = True
                    break

            if not found:
                raise NameError('No file found ending in .in')

        self.in_field = str(config_parse['general']['field'])

        self.analysis = config_parse.get('general', 'analysis',
                                         fallback='all')
        if self.analysis not in ['all', 'perp', 'par', 'time', 'write_field', 
                                 'film']:
            raise ValueError('Analysis must be one of (perp, time, par, '
                             'write_field, make_film)')

        self.time_interpolate_bool = config_parse.getboolean('general', 
                                                             'time_interpolate',
                                                             fallback=True)

        self.time_interp_fac = int(config_parse.get('general', 
                                                    'time_interp_fac',
                                                    fallback=1))

        self.zero_bes_scales_bool = config_parse.getboolean('general',
                                   'zero_bes_scales', fallback=False)

        self.zero_zf_scales_bool = config_parse.getboolean('general',
                                   'zero_zf_scales', fallback=False)

        self.lab_frame = config_parse.getboolean('general',
                                   'lab_frame', fallback=False)

        self.spec_idx = str(config_parse['general']['species_index'])
        if self.spec_idx == "None":
            self.spec_idx = None
        else:
            self.spec_idx = int(self.spec_idx)

        self.theta_idx = str(config_parse['general']['theta_index'])
        if self.theta_idx == "None":
            self.theta_idx = None
        elif self.theta_idx == "-1":
            self.theta_idx = [0, None]
        elif type(eval(self.theta_idx)) == int:
            self.theta_idx = [int(self.theta_idx), int(self.theta_idx)+1] 
        else:
            raise ValueError("theta_idx can only be one of: None, -1(all), int")
        if self.analysis == 'par' and self.theta_idx != [0, None]:
            warnings.warn('Analysis = par, change to reading full theta grid.')
            self.theta_idx = [0, None]

        self.time_slice = int(config_parse.get('general', 'time_slice',
                                               fallback=49))

        self.box_size = str(config_parse.get('general',
                                               'box_size', fallback='[0.2,0.2]'))
        self.box_size = self.box_size[1:-1].split(',')
        self.box_size = [float(s) for s in self.box_size]

        self.time_range = str(config_parse.get('general',
                                               'time_range', fallback='[0,-1]'))
        self.time_range = self.time_range[1:-1].split(',')
        self.time_range = [float(s) for s in self.time_range]
        if self.time_range[1] == -1:
            self.time_range[1] = None

        #################
        # Perp Namelist #
        #################

        self.ky_free = config_parse.getboolean('perp','ky_free', fallback=False)

        perp_guess = str(config_parse.get('perp',
                                          'perp_guess', fallback='[0.05,0.1,1]'))
        perp_guess = perp_guess[1:-1].split(',')
        perp_guess = [float(s) for s in perp_guess]
        self.perp_guess_x = perp_guess[0]
        self.perp_guess_y = perp_guess[1]
        if self.ky_free:
            try:
                self.perp_guess_ky = perp_guess[2]
            except IndexError:
                warnings.warn('ky guess not specified in perp_guess, '
                              'setting perp_guess_ky = 1')
                self.perp_guess_ky = 1.0


        #################
        # Time Namelist #
        #################

        self.npeaks_fit = int(config_parse.get('time',
                                               'npeaks_fit', fallback=5))
        self.time_guess = str(config_parse.get('time',
                                          'time_guess', fallback='[1e-5,100]'))
        self.time_guess = self.time_guess[1:-1].split(',')
        self.time_guess = [float(s) for s in self.time_guess]

        self.time_guess_dec = self.time_guess[0]
        self.time_guess_grow = self.time_guess[0]
        self.time_guess_osc = np.array([self.time_guess[0], self.time_guess[1], 
                                        0.0])

        self.time_max = float(config_parse.get('time',
                                               'time_max', fallback=1))

        ################
        # Par Namelist #
        ################

        self.par_guess = str(config_parse.get('par',
                                              'par_guess', fallback='[1,0.1]'))
        self.par_guess = self.par_guess[1:-1].split(',')
        self.par_guess = [float(s) for s in self.par_guess]

        ###################
        # Output Namelist #
        ###################

        self.seaborn_context = str(config_parse.get('output', 'seaborn_context',
                                                    fallback='talk'))
        self.film_fps = int(config_parse.get('output', 'film_fps', fallback=40))
        self.film_contours = int(config_parse.get('output', 'film_contours', 
                                                  fallback=30))
        self.film_lim = str(config_parse.get('output', 'film_lim', 
                                                  fallback='None'))
        if self.film_lim == "None":
            self.film_lim = None
        else:
            self.film_lim = self.film_lim[1:-1].split(',')
            self.film_lim = [float(s) for s in self.film_lim]

        self.write_field_interp_x = config_parse.getboolean('output', 
                                                         'write_field_interp_x',
                                                         fallback=True)

        # Log the variables
        logging.info('The following values were read from ' + self.config_file)
        logging.info(vars(self))
        logging.info('Finished read_config.')

    def config_checks(self):
        """
        This function contains consistency checks of configurations parameters.
        """

        if self.time_slice%2 != 1:
            warnings.warn('time_slice should be odd, reducing by one...')
            self.time_slice -= 1

        if self.lab_frame and self.omega == 0:
            warnings.warn('Changing to lab frame but omega = 0 (default).')

        if self.lab_frame and self.dpsi_da == 0:
            warnings.warn('Changing to lab frame but dpsi_da = 0 (default).')

        if self.lab_frame and self.time_interp_fac == 1:
            warnings.warn('Transforming to lab frame, but time_interp_fac = 1. '
                          'This is probably not high enough. Recommend 4.')

        if not self.lab_frame and self.time_interp_fac > 1:
            warnings.warn('Not transforming to lab frame, but time_interp_fac > 1. '
                          'This is probably not needed.')
        
        if self.theta_idx == None and self.in_field[-2:] == '_t':
            raise ValueError('You have specified a field with theta info but '
                             'left theta_idx=None. Specify theta_idx as -1 '
                             'for full theta info or pick a specific theta.')

        if self.analysis != 'par' and self.ntheta > 1:
            raise ValueError('You are not doing parallel analysis and have more '
                             'than one theta point. Can only handle one theta '
                             'value at the moment!')

        if self.analysis == 'perp' and self.zero_zf_scales_bool == False:
            warnings.warn('Doing perp analysis but not zeroing ZF scales. This '
                          'is required for radial correlation. Changing '
                          'zero_zf_scales_bool to True')
            self.zero_zf_scales_bool = True

    def read_netcdf(self):
        """
        Read array from NetCDF file.

        Read array specified in configuration file as 'in_field'. Function
        uses information from the configuration object passed to it.

        Notes
        -----

        * An additional axis is added in the case of no theta_idx. This allows
          usual loops in theta to be kept but have no effect. If only one 
          element in dimension initialization will be performed regardless,
          however dimension will be removed for perp and time analysis.
        """
        logging.info('Start reading from NetCDf file...')

        self.ncfile = netcdf.netcdf_file(self.cdf_file, 'r')

        # NetCDF order is [t, species, ky, kx, theta, r]
        # ncfile.variable returns netcdf object - convert to array
        if self.theta_idx == None:
            self.field = np.array(self.ncfile.variables[self.in_field]
                                            [self.time_range[0]:self.time_range[1],
                                             self.spec_idx,:,:,self.theta_idx,:])
        else:
            self.field = np.array(self.ncfile.variables[self.in_field]
                                            [self.time_range[0]:self.time_range[1],
                                             self.spec_idx,:,:,
                                             self.theta_idx[0]:self.theta_idx[1],
                                             :])
        self.t = np.array(self.ncfile.variables['t'][self.time_range[0]:
                                                     self.time_range[1]])


        self.field = np.squeeze(self.field)
        self.field = np.swapaxes(self.field, 1, 2)
        if len(self.field.shape) < 5:
            self.field = self.field[:,:,:,np.newaxis,:]

        self.kx = np.array(self.ncfile.variables['kx'][:])
        self.ky = np.array(self.ncfile.variables['ky'][:])
        self.theta = np.array(self.ncfile.variables['theta'][:])
        self.drho_dpsi = float(self.ncfile.variables['drhodpsi'].data)
        self.gradpar = np.array(self.ncfile.variables['gradpar'][:])/self.amin
        try:
            self.bpol = np.array(self.ncfile.variables['bpol'][:])*self.bref
        except KeyError:
            self.bpol = self.geometry[:,7]*self.bref

        self.ncfile.close()

        logging.info('Finished reading from NetCDf file.')

    def read_geometry_file(self):
        """
        Read the geometry file for the GS2 run.
        """
        logging.info('Reading geometry file...')
        
        self.geometry = np.loadtxt(self.g_file)

        logging.info('Finished reading geometry file.')

    def read_input_file(self):
        """
        Read the input '.inp' file for the GS2 run.
        """
        logging.info('Reading input file...')
        
        self.input_file = nml.read(self.in_file)

        logging.info('Finished reading input file.')

    def field_to_complex(self):
        """
        Converts field to a complex array.

        Field is in the following format: field[t, kx, ky, ri] where ri
        represents a dimension of length 2.

        * ri = 0 - Real part of the field.
        * ri = 1 - Imaginary part of the field.
        """
        self.field = self.field[:,:,:,:,0] + 1j*self.field[:,:,:,:,1]

    def fourier_correction(self):
        """
        Correction to GS2s Fourier components to regular Fourier components.

        Notes
        -----

        GS2 fourier components are NOT simple fourier components obtained from 
        regular FFT functions. Due to GS2's legacy as a linear code, instead
        of just getting fourier components g_k the output is:

        G_k = {g_k for ky = 0, 2g_k for ky > 0}

        Therfore converting to regular fourier components simply means dividing
        all non-zonal components by 2.
        """
        self.field[:,:,1:,:] = self.field[:,:,1:,:]/2

    def time_interpolate(self):
        """
        Interpolates in time onto a regular grid

        Depending on whether the user specified time_interpolate, the time grid
        is interpolated into a regular grid. This is required in order to do
        FFTs in time. Interpolation is done by default if not specified. 
        time_interp_fac sets the multiple of interpolation.
        """
        logging.info('Started interpolating onto a regular time grid...')

        t_reg = np.linspace(min(self.t), max(self.t), self.time_interp_fac*self.nt)
        tmp_field = np.empty([self.time_interp_fac*self.nt, self.nkx, self.nky,
                              self.ntheta], dtype=complex)
        for ikx in range(self.nkx):
            for iky in range(self.nky):
                for ith in range(self.ntheta):
                    f = interp.interp1d(self.t, self.field[:, ikx, iky, ith])
                    tmp_field[:, ikx, iky, ith] = f(t_reg)
        self.t = t_reg
        self.nt = len(self.t)
        self.field = tmp_field

        tmp_field = None
        f = None
        gc.collect()

        logging.info('Finished interpolating onto a regular time grid.')

    def zero_bes_scales(self):
        """
        Sets modes larger than the BES to zero.

        The BES is approximately 160x80mm(rad x pol), so we would set kx < 0.25
        and ky < 0.5 to zero, since k = 2 pi / L.
        """
        for ikx in range(self.nkx):
            for iky in range(self.nky):
                for ith in range(self.ntheta):
                    # Roughly the size of BES (160x80mm)
                    if abs(self.kx[ikx]) < 0.25 and self.ky[iky] < 0.5:
                        self.field[:,ikx,iky,ith] = 0.0

    def zero_zf_scales(self):
        """
        Sets zonal flow (ky = 0) modes to zero.
        """
        self.field[:,:,0,:] = 0.0

    def to_lab_frame(self):
        """
        Transforms from the rotating frame to the lab frame.

        Notes
        -----

        The important thing here is that ky is NOT the toroidal wavenumber *n0*. 
        It is related to n0 by [1]_:

        ky_gs2 = n0*(rho_ref/a_min)*dpsi_da

        .. [1] C. M. Roach, "Equilibrium flow shear implementation in GS2",
               http://gyrokinetics.sourceforge.net/wiki/index.php/Documents,
               http://svn.code.sf.net/p/gyrokinetics/code/wikifiles/CMR/ExB_GS2.pdf
        """
        n0 = int(self.ky[1]*(self.amin/self.rho_ref)*self.dpsi_da)
        for ix in range(self.nkx):
            for iy in range(self.nky):
                for ith in range(self.ntheta):
                    self.field[:,ix,iy,ith] = self.field[:,ix,iy,ith]*np.exp(1j * \
                                                n0 * iy * self.omega * self.t)

    def field_to_real_space(self):
        """
        Converts field from (kx, ky) to (x, y) and saves as new array attribute.

        Notes
        -----

        * Since python defines x = IFFT[FFT(x)] need to undo the implicit 
          normalization by multiplying by the size of the arrays.
        * GS2 fluctuations are O(rho_star) and must be multiplied by rho_star
          to get their true values.
        * In order to avoid memory overloads, the fourier space field is
          cleared after the real space field is calculated.
        """
        logging.info('Calculating real space field...')

        self.field_real_space = np.empty([self.nt,self.nx,self.ny,self.ntheta],
                                         dtype=float)
        pyfftw.n_byte_align(self.field, 16)
        self.field_real_space = pyfftw.interfaces.numpy_fft.irfft2(self.field, 
                                                                   axes=[1,2])

        if self.analysis == 'par':
            self.field = None
            gc.collect()

        self.field_real_space = np.roll(self.field_real_space,
                                                int(self.nx/2), axis=1)

        self.field_real_space = self.field_real_space*self.nx*self.ny
        self.field_real_space = self.field_real_space*self.rho_star
        
        logging.info('Finished calculating real space field.')

    def domain_reduce(self):
        """
        Initialization consists of: 
        
        * Calculating radial and poloidal coordinates r, z.
        * Using input parameter *box_size* to determine the index range to 
          perform the correlation analysis on.
        * Reduce extent of real space field using this index.
        * Recalculate some real space arrays such as x, y, dx, dy, etc.
        """
        logging.info('Reducing domain size to %f x %f m'%(self.box_size[0],
                                                          self.box_size[1]))

        # Switch box size to length either side of 0
        self.box_size = np.array(self.box_size)/2

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
                int(self.ny/2)-z_box_idx+1:int(self.ny/2)+z_box_idx,:]

        # Recalculate real space arrays
        self.nx = len(self.r)
        self.ny = len(self.z)

        self.x = np.linspace(0, self.r[-1] - self.r[0], self.nx)
        self.y = np.linspace(0, 2*self.z[-1], self.ny)

        logging.info('Finished reducing domain size.')

    def field_odd_pts(self):
        """
        Ensures real space field has odd number of points in x and y.

        This is done so that when ``sig.fftconvolve`` is called on the field
        with the 'same' option, the resulting correlation function also has an 
        odd number of points. If this wasn't done, one axis *might* have an 
        even number of points meaning the correlation function will not have a 
        convenient middle point to use as the zero separation point, which 
        would then require more general fitting functions. This way we can 
        force dx=0, dy=0 at the self.nx/2, self.ny/2 location and avoid extra 
        parameters in the fitting functions.
        """
        logging.info('Ensuring field has odd points in space')

        if self.nx%2 != 1:
            self.field_real_space = self.field_real_space[:,:-1,:,:]
            self.x = self.x[:-1] 
        if self.ny%2 != 1:
            self.field_real_space = self.field_real_space[:,:,:-1,:]
            self.y = self.y[:-1] 

        self.nx = len(self.x)
        self.ny = len(self.y)

        self.dx = np.linspace(-self.x[-1]/2, self.x[-1]/2, self.nx)
        self.dy = np.linspace(-self.y[-1]/2, self.y[-1]/2, self.ny)

    def perp_analysis(self):
        """
        Performs a perpendicular correlation analysis on the field.

        Notes
        -----

        * Remove theta dimension before doing analysis to avoid pointless code
          which accounts for a theta information. Will produce error if more 
          than one element during config_checks.
        * Uses sig.fftconvolve to calculate the 2D perp correlation function.
        * Splits correlation function into time slices and fits each time
          slice with a tilted Gaussian using the perp_fit function.
        * The fit parameters for the previous time slice is used as the initial
          guess for the next time slice.
        * Also writes information on the mean fluctuation levels
        """

        logging.info('Start perpendicular correlation analysis...')

        if not self.ky_free:
            self.perp_dir = 'perp/ky_fixed'
        else:
            self.perp_dir = 'perp/ky_free'
        if self.perp_dir not in os.listdir(self.out_dir):
            os.system("mkdir -p " + self.out_dir + '/' + self.perp_dir)
        if 'corr_fns_x' not in os.listdir(self.out_dir + '/' + self.perp_dir):
            os.system("mkdir -p " + self.out_dir+'/'+self.perp_dir+'/corr_fns_x')
        if 'corr_fns_y' not in os.listdir(self.out_dir + '/' + self.perp_dir):
            os.system("mkdir -p " + self.out_dir+'/'+self.perp_dir+'/corr_fns_y')
        os.system('rm ' + self.out_dir + '/' + self.perp_dir + '/corr_fns_x/*')
        os.system('rm ' + self.out_dir + '/' + self.perp_dir + '/corr_fns_y/*')

        self.field_normalize_perp()
        self.calculate_perp_corr()
        self.perp_norm_mask()

        self.perp_fit_x = np.empty([self.nt_slices], dtype=float)
        self.perp_fit_x_err = np.empty([self.nt_slices], dtype=float)
        self.perp_fit_y = np.empty([self.nt_slices], dtype=float)
        self.perp_fit_y_err = np.empty([self.nt_slices], dtype=float)
        if self.ky_free:
            self.perp_fit_ky = np.empty([self.nt_slices], dtype=float)
            self.perp_fit_ky_err = np.empty([self.nt_slices], dtype=float)

        for it in range(self.nt_slices):
            self.perp_corr_fit(it)

        if not self.ky_free:
            np.savetxt(self.out_dir + '/' + self.perp_dir + '/perp_fit_params.csv', 
                       np.array([self.perp_fit_x, self.perp_fit_x_err, 
                        self.perp_fit_y, self.perp_fit_y_err]).T, 
                       delimiter=',', fmt='%1.4f', 
                       header='lx, std(lx), ly, std(ly)')
        else:
            np.savetxt(self.out_dir + '/' + self.perp_dir + '/perp_fit_params.csv', 
                       np.array([self.perp_fit_x, self.perp_fit_x_err, 
                        self.perp_fit_y, self.perp_fit_y_err, 
                        self.perp_fit_ky, self.perp_fit_ky]).T, 
                       delimiter=',', fmt='%1.4f', 
                       header='lx, std(lx), ly, std(ly), ky, std(ky)')

        self.perp_analysis_summary()

        self.fluctuation_levels()

        logging.info('Finished perpendicular correlation analysis.')

    def field_normalize_perp(self):
        """
        Defines normalized field for the perpandicular correlation by 
        subtracting the mean and dividing by the RMS value.
        """
        logging.info('Normalizing the real space field...')

        self.field_real_space_norm_x = \
                                np.empty([self.nt,self.nx,self.ny],dtype=float)
        self.field_real_space_norm_y = \
                                np.empty([self.nt,self.nx,self.ny],dtype=float)
        for it in range(self.nt):
            for iy in range(self.ny):
                self.field_real_space_norm_x[it,:,iy] = \
                                    self.field_real_space[it,:,iy] - \
                                    np.mean(self.field_real_space[it,:,iy])
                self.field_real_space_norm_x[it,:,iy] /= \
                                    np.std(self.field_real_space_norm_x[it,:,iy])

            for ix in range(self.nx):
                self.field_real_space_norm_y[it,ix,:] = \
                                    self.field_real_space[it,ix,:] - \
                                    np.mean(self.field_real_space[it,ix,:])
                self.field_real_space_norm_y[it,ix,:] /= \
                                    np.std(self.field_real_space_norm_y[it,ix,:])

        logging.info('Finished normalizing the real space field.')

    def calculate_perp_corr(self):
        """
        Calculates the perpendicular correlation function from the real space
        field.
        """
        logging.info("Calculating perpendicular correlation function...")

        self.perp_corr_x = np.empty([self.nt, self.nx, self.ny], dtype=float)
        self.perp_corr_y = np.empty([self.nt, self.nx, self.ny], dtype=float)
        for it in range(self.nt):

            for iy in range(self.ny):
                self.perp_corr_x[it,:,iy] = \
                            sig.correlate(self.field_real_space_norm_x[it,:,iy], 
                                          self.field_real_space_norm_x[it,:,iy],
                                          mode='same')

            for ix in range(self.nx):
                self.perp_corr_y[it,ix,:] = \
                            sig.correlate(self.field_real_space_norm_y[it,ix,:], 
                                          self.field_real_space_norm_y[it,ix,:],
                                          mode='same')

        logging.info("Finished calculating perpendicular correlation " 
                     "function...")

    def perp_norm_mask(self):
        """
        Applies the appropriate normalization to the perpendicular correlation 
        function.

        Notes
        -----

        After calling ``sig.correlate`` to calculate ``perp_corr``, we are
        left with an unnormalized correlation function as a function of dx
        or dy. This function applies a nomalization mask to ``perp_corr_x`` and
        ``perp_corr_y`` which is dependent on the number of points that 
        ``field_real_space_norm`` has in common with itself for a given dx or dy. 
        ``field_real_space_norm`` is already normalized to the standard 
        deviation of the time signal, so the only difference between correlate 
        and np.corrcoef is the number of points in common in the convolution 
        (that aren't the zero padded values and after averaging over many time 
        steps).
        """
        logging.info('Applying perp normalization mask...')

        x = np.ones([self.nx]) 
        y = np.ones([self.ny]) 
        mask_x = sig.correlate(x,x,'same')
        mask_y = sig.correlate(y,y,'same')

        self.perp_corr_x /= mask_x[np.newaxis, :, np.newaxis] 
        self.perp_corr_y /= mask_y

        logging.info('Finised applying perp normalization mask...')

    def perp_corr_fit(self, it):
        """
        Fits the appropriate Gaussian to the radial and poloidal correlation 
        functions.

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being fitted.

        Notes
        -----

        * The radial correlation function is fitted with a Gaussian.
        * The poloidal correlation function is fitted with an oscillating 
          Gaussian.
        """

        corr_fn_x = \
            np.array(self.perp_corr_x[it*self.time_slice:(it+1)*self.time_slice,:,:])
        corr_fn_y = \
            np.array(self.perp_corr_y[it*self.time_slice:(it+1)*self.time_slice,:,:])
        
        # Average corr_fn over time
        corr_std_x = np.empty([self.nx])
        corr_std_y = np.empty([self.ny])
        for ix in range(self.nx):
            corr_std_x[ix] = np.std(corr_fn_x[:,ix,:])
        for iy in range(self.ny):
            corr_std_y[iy] = np.std(corr_fn_y[:,:,iy])
        avg_corr_x = np.mean(np.mean(corr_fn_x, axis=0), axis=1)
        avg_corr_y = np.mean(np.mean(corr_fn_y, axis=0), axis=0)

        gmod_gauss = lm.Model(fit.gauss)
        gmod_osc_gauss = lm.Model(fit.osc_gauss)

        params_x = lm.Parameters()                                                    
        params_x.add('l', value=self.perp_guess_x)                                                  
        params_x.add('p', value=0.0, vary=False)               
        fit_x = gmod_gauss.fit(avg_corr_x, params_x, x=self.dx)
        
        params_y = lm.Parameters()                                                    
        params_y.add('l', value=self.perp_guess_y)                                                  
        if not self.ky_free:
            params_y.add('k', value=1, expr='2*3.141592653589793/l')
        else:
            params_y.add('k', value=self.perp_guess_ky)
        params_y.add('p', value=0.0, vary=False) 
        fit_y = gmod_osc_gauss.fit(avg_corr_y, params_y, x=self.dy)

        self.perp_fit_x[it] = fit_x.best_values['l']
        self.perp_fit_y[it] = fit_y.best_values['l']
        if self.ky_free:
            self.perp_fit_ky[it] = fit_y.best_values['k']

        if fit_x.errorbars:
            self.perp_fit_x_err[it] = np.sqrt(fit_x.covar[0,0])
        else:
            self.perp_fit_x_err[it] = 0

        if fit_y.errorbars:
            self.perp_fit_y_err[it] = np.sqrt(fit_y.covar[0,0])
            if self.ky_free:
                self.perp_fit_ky_err[it] = np.sqrt(fit_y.covar[1,1])
        else:
            self.perp_fit_y_err[it] = 0
            if self.ky_free:
                self.perp_fit_ky_err[it] = 0

        self.perp_guess_x = fit_x.best_values['l']
        self.perp_guess_y = fit_y.best_values['l']
        if self.ky_free:
            self.perp_guess_ky = fit_y.best_values['k']

        self.perp_plots_x(it, avg_corr_x, corr_std_x, fit_x)
        self.perp_plots_y(it, avg_corr_y, corr_std_y, fit_y)

    def perp_plots_x(self, it, corr_fn, corr_std, corr_fit):
        """
        Plot radial correlation function and fitted Gaussian.
        """
        plot_style.white()

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.scatter(self.dx, corr_fn, c=pal[0], 
                     label=r'$C(\Delta x)$')
        plt.plot(self.dx, corr_fit.best_fit, c=pal[2], 
                 label=r'$\exp(-(\Delta x / \ell_x)^2)$')
        plt.fill_between(self.dx, corr_fn-corr_std, corr_fn+corr_std, 
                         alpha=0.3)
        plt.legend()
        plt.xlabel(r'$\Delta x$ (m)')
        plt.xlim([self.dx[0], self.dx[-1]])
        plt.ylabel(r'$C(\Delta x)$')
        plt.ylim(ymax=1)
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/' +self.perp_dir + 
                    '/corr_fns_x/corr_x_fit_it_' + str(it) + '.pdf')
        plt.close(fig)

    def perp_plots_y(self, it, corr_fn, corr_std, corr_fit):
        """
        Plot radial correlation function and fitted Gaussian.
        """
        plot_style.white()

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.scatter(self.dy, corr_fn, c=pal[0], label=r'$C(\Delta y)$')
        plt.plot(self.dy, np.exp(-(self.dy/corr_fit.best_values['l'])**2), 
                 'k--', label=r'$\exp(-(\Delta y / \ell_y)^2)$')
        if not self.ky_free:
            fit_label=r'$\exp(-(\Delta y / \ell_y)^2) \cos(2 \pi \Delta y/ \ell_y)$' 
        else:
            fit_label=r'$\exp(-(\Delta y / \ell_y)^2) \cos(k_y \Delta y)$'
        plt.plot(self.dy, corr_fit.best_fit, c=pal[2], label=fit_label)
        plt.fill_between(self.dy, corr_fn-corr_std, corr_fn+corr_std, 
                         alpha=0.3)
        plt.legend()
        plt.xlabel(r'$\Delta y$ (m)')
        plt.xlim([self.dy[0], self.dy[-1]])
        plt.ylabel(r'$C(\Delta y)$')
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/' +self.perp_dir + 
                    '/corr_fns_y/corr_y_fit_it_' + str(it) + '.pdf')
        plt.close(fig)

    def perp_analysis_summary(self):
        """
        Prints out a summary of the perpendicular analysis.

        * Plots fitting parameters as a function of time window.
        * Averages them in time and calculates a standard deviation.
        * Writes summary to a text file.
        """
        logging.info("Writing perp_analysis summary...")

        plt.clf()
        plot_style.white()
        fig, ax = plt.subplots(1, 1)
        plt.errorbar(range(self.nt_slices), np.abs(self.perp_fit_x), 
                     yerr=self.perp_fit_x_err)
        plt.xlabel('Time Window')
        plt.ylabel(r'$l_x$ (m)')
        plt.ylim(ymin=0, ymax=2*np.mean(np.abs(self.perp_fit_x[0])))
        plt.xticks(range(self.nt_slices))
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/'+self.perp_dir+'/perp_fit_x_vs_time_slice.pdf')
        plt.close(fig)

        plt.clf()
        plot_style.white()
        fig, ax = plt.subplots(1, 1)
        plt.errorbar(range(self.nt_slices), np.abs(self.perp_fit_y), 
                     yerr=self.perp_fit_y_err)
        plt.xlabel('Time Window')
        plt.ylabel(r'$l_y$ (m)')
        plt.ylim(ymin=0, ymax=2*np.mean(np.abs(self.perp_fit_y)))
        plt.xticks(range(self.nt_slices))
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/'+self.perp_dir+
                    '/perp_fit_y_vs_time_slice.pdf')
        plt.close(fig)

        if not self.ky_free:
            np.savetxt(self.out_dir + '/' + self.perp_dir + '/perp_fit_summary.csv', 
                       np.nanmean([self.perp_fit_x, 
                                   self.perp_fit_x_err, 
                                   self.perp_fit_y, 
                                   self.perp_fit_y_err], axis=1)[np.newaxis,:], 
                       delimiter=',', fmt='%1.4f', 
                       header='lx, std(lx), ly, std(ly)')
        else:
            plt.clf()
            plot_style.white()
            fig, ax = plt.subplots(1, 1)
            plt.errorbar(range(self.nt_slices), np.abs(self.perp_fit_ky), 
                         yerr=self.perp_fit_ky_err)
            plt.xlabel('Time Window')
            plt.ylabel(r'$k_y (m^{-1})$')
            plt.ylim(ymin=0, ymax=2*np.mean(np.abs(self.perp_fit_ky)))
            plt.xticks(range(self.nt_slices))
            plot_style.minor_grid(ax)
            plot_style.ticks_bottom_left(ax)
            plt.savefig(self.out_dir + '/'+self.perp_dir+
                        '/perp_fit_ky_vs_time_slice.pdf')
            plt.close(fig)

            np.savetxt(self.out_dir + '/' + self.perp_dir + '/perp_fit_summary.csv', 
                       np.mean([self.perp_fit_x, 
                                self.perp_fit_x_err, 
                                self.perp_fit_y, 
                                self.perp_fit_y_err,
                                self.perp_fit_ky,
                                self.perp_fit_ky_err], axis=1)[np.newaxis,:], 
                       delimiter=',', fmt='%1.4f', 
                       header='lx, std(lx), ly, std(ly)')

        logging.info("Finished writing perp_analysis summary...")

    def fluctuation_levels(self):
        """
        Caculates mean fluctuation level and standard deviation and writes 
        results.

        Notes
        -----

        More precisely the following is calculated

        * At each grid point the RMS value is calculated in time.
        * The mean and std of the fluctuation levels are then the mean and 
          std of these RMS values.

        This is done since the standard deviation of a quantity with a mean of
        zero will be the RMS value so a straighforward mean and std across the
        whole box cannot be done.
        """
        logging.info("Calculating fluctuation level...")

        rms = np.sqrt(np.mean(self.field_real_space**2, axis=0))
    
        self.fluc_level = np.mean(rms)
        self.fluc_level_std = np.std(rms)

        np.savetxt(self.out_dir + '/'+ self.perp_dir +'/fluctuation_summary.csv',
                   np.array([self.fluc_level, self.fluc_level_std])[np.newaxis,:],
                   header='dn/n, std(dn/n)', delimiter=',', fmt='%1.4f')

        logging.info("Finished calculating fluctuation level.")

    def time_analysis(self):
        """
        Performs a time correlation analysis on the field.

        Notes
        -----

        * Split into time windows and perform correlation analysis on each 
          window separately.
        """
        logging.info("Starting time_analysis...")

        if self.lab_frame:
            self.time_dir = 'time_lab_frame'
        elif not self.lab_frame:
            self.time_dir = 'time'

        if self.time_dir not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/' + self.time_dir)
        if 'corr_fns' not in os.listdir(self.out_dir+'/'+self.time_dir):
            os.system("mkdir " + self.out_dir + '/'+self.time_dir+'/corr_fns')
        os.system('rm ' + self.out_dir + '/'+self.time_dir+'/corr_fns/*')
        
        self.time_corr = np.empty([self.nt_slices, self.time_slice, self.nx,
                                   self.ny], dtype=float)
        self.corr_time = np.empty([self.nt_slices, self.nx], dtype=float)
        self.corr_time_err = np.empty([self.nt_slices, self.nx], dtype=float)

        self.field_normalize_time()
        for it in range(self.nt_slices):
            self.calculate_time_corr(it)
            self.time_norm_mask(it)
            self.time_corr_fit(it)

        self.time_analysis_summary()

        logging.info("Finished time_analysis...")

    def field_normalize_time(self):
        """
        Defines normalized field for the time correlation by subtracting the 
        mean and dividing by the RMS value.
        """
        logging.info('Normalizing the real space field...')

        self.field_real_space_norm = \
                                np.empty([self.nt,self.nx,self.ny],dtype=float)

        for it in range(self.nt_slices):
            t_min = it*self.time_slice
            t_max = (it+1)*self.time_slice
            field_window = self.field_real_space[t_min:t_max,:,:]
            for ix in range(self.nx):
                self.field_real_space_norm[t_min:t_max,ix,:] = \
                                        field_window[:,ix,:] - \
                                        np.mean(field_window[:,ix,:])
                self.field_real_space_norm[t_min:t_max,ix,:] /= \
                        np.std(self.field_real_space_norm[t_min:t_max,ix,:])

        logging.info('Finished normalizing the real space field.')

    def calculate_time_corr(self, it):
        """
        Calculate the time correlation for a given time window at each x.

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being calculated.
        """
        
        field_window = self.field_real_space_norm[it*self.time_slice:(it+1)*
                                                  self.time_slice,:,:]

        for ix in range(self.nx):
            self.time_corr[it,:,ix,:] = sig.fftconvolve(field_window[:,ix,:], 
                                                        field_window[::-1,ix,::-1],
                                                        'same')

    def time_norm_mask(self, it):
        """
        Applies the appropriate normalization to the time correlation function.

        Notes
        -----

        After calling ``sig.fftconvolve`` to calculate ``time_corr``, we are
        left with an unnormalized correlation function as a function of dt
        and dy. This function applies a 2D nomalization mask to ``time_corr``
        which is dependent on the number of points that ``field_real_space_norm``
        has in common with itself for a given dt, dy, and time window. 
        ``field_real_space_norm`` is already normalized to the standard 
        deviation of the time signal, so the only difference between fftconvolve 
        and np.corrcoef is the number of points in common in the convolution 
        (that aren't the zero padded values and after averaging over many time 
        steps).

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being calculated.
        """
        logging.info('Applying time normalization mask...')

        x = np.ones([self.time_slice, self.ny]) 
        mask = sig.fftconvolve(x,x,'same')

        for ix in range(self.nx):
            self.time_corr[it,:,ix,:] /= mask 

        logging.info('Finised applying time normalization mask.')

    def time_corr_fit(self, it):
        """
        Fit the time correlation function with exponentials to find correlation
        time.

        Notes
        -----

        The fitting procedure consists of the following steps:
        
        * Loop over radial points
        * Identify the time correlation function peaks
        * Determine what type of function to fit to the peaks
        * Fitting the appropriate function

        The peaks can be fitted with a growing/decaying exponential depending
        on the direction of decrease of the peaks, or an oscillating exponential
        if the peaks don't monotically increase or decrease.

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being fitted.
        """
        t = self.t[it*self.time_slice:(it+1)*self.time_slice]
        self.dt = np.linspace((-max(t)+t[0])/2, (max(t)-t[0])/2, self.time_slice)

        peaks = np.zeros([self.nx, self.npeaks_fit], dtype=float)
        max_index = np.empty([self.nx, self.npeaks_fit], dtype=int);
        mid_idx = int(self.ny/2)

        for ix in range(self.nx):
            for iy in range(mid_idx,mid_idx+self.npeaks_fit):
                max_index[ix, iy-mid_idx], peaks[ix, iy-mid_idx] = \
                    max(enumerate(self.time_corr[it,:,ix,iy]), 
                        key=operator.itemgetter(1))

            if (fit.strictly_increasing(max_index[ix,:]) == True or 
                fit.strictly_increasing(max_index[ix,::-1]) == True):
                if max_index[ix, self.npeaks_fit-1] > max_index[ix, 0]:
                    try:
                        gmod_decay = lm.Model(fit.decaying_exp)
                        params_t = lm.Parameters()
                        params_t.add('tau_c', value=self.time_guess_dec)
                        fit_t = gmod_decay.fit(peaks[ix,:], params_t, 
                                               t=self.dt[max_index[ix,:]]) 

                        if fit_t.best_values['tau_c'] > self.time_max:
                            self.corr_time[it,ix] = np.nan
                            self.corr_time_err[it,ix] = np.nan
                        else:
                            self.corr_time[it,ix] = fit_t.best_values['tau_c']
                            if fit_t.errorbars:
                                self.corr_time_err[it,ix] = fit_t.covar[0,0]
                            else:
                                self.corr_time_err[it,ix] = np.nan
                            self.time_guess_dec = self.corr_time[it,ix]
                            self.time_plot(it, ix, max_index, peaks, 'decaying')

                        logging.info("(" + str(it) + "," + str(ix) + ") was fitted "
                                     "with decaying exponential. tau = " 
                                     + str(self.corr_time[it,ix]) + " s\n")

                    except RuntimeError:
                        logging.info("(" + str(it) + "," + str(ix) + ") " 
                                "RuntimeError - max fitting iterations reached, "
                                "skipping this case with tau = NaN\n")
                        self.corr_time[it, ix] = np.nan
                else:
                    try:
                        gmod_grow = lm.Model(fit.growing_exp)
                        params_t = lm.Parameters()
                        params_t.add('tau_c', value=self.time_guess_grow)
                        fit_t = gmod_grow.fit(peaks[ix,:], params_t, 
                                              t=self.dt[max_index[ix,:]]) 

                        if fit_t.best_values['tau_c'] > self.time_max:
                            self.corr_time[it,ix] = np.nan
                            self.corr_time_err[it,ix] = np.nan
                        else:
                            self.corr_time[it,ix] = fit_t.best_values['tau_c']
                            if fit_t.errorbars:
                                self.corr_time_err[it,ix] = fit_t.covar[0,0]
                            else:
                                self.corr_time_err[it,ix] = np.nan
                            self.time_guess_grow = self.corr_time[it,ix]
                            self.time_plot(it, ix, max_index, peaks, 'growing')

                        logging.info("(" + str(it) + "," + str(ix) + ") was fitted "
                                     "with growing exponential. tau = " 
                                     + str(self.corr_time[it,ix]) + " s\n")

                    except RuntimeError:
                        logging.info("(" + str(it) + "," + str(ix) + ") "
                                "RuntimeError - max fitting iterations reached, "
                                "skipping this case with tau = NaN\n")
                        self.corr_time[it, ix] = np.nan
            else:
                try:
                    # If abs(max_index) is not monotonically increasing, this 
                    # usually means that there is no flow and that the above method
                    # cannot be used to calculate the correlation time. Try fitting
                    # a decaying oscillating exponential to the central peak.
                    gmod_osc = lm.Model(fit.osc_gauss)
                    params_t = lm.Parameters()
                    params_t.add('l', value=self.time_guess_osc[0])
                    params_t.add('k', value=self.time_guess_osc[1])
                    params_t.add('p', value=self.time_guess_osc[2], vary=False)
                    fit_t = gmod_osc.fit(self.time_corr[it,:,ix,mid_idx], 
                                         params_t, x=self.dt) 

                    # Note l = tau_c sinc fitting function specification is for
                    # general l, k, p.
                    if fit_t.best_values['l'] > self.time_max:
                        self.corr_time[it,ix] = np.nan
                        self.corr_time_err[it,ix] = np.nan
                        logging.info("(" + str(it) + "," + str(ix) + ") was "
                                     "fitted with an oscillating Gaussian to "
                                     "the central peak. (tau, omega) = " 
                                     + str([np.nan, np.nan]) + "\n")

                    else:
                        self.corr_time[it,ix] = fit_t.best_values['l']
                        if fit_t.errorbars:
                            self.corr_time_err[it,ix] = fit_t.covar[0,0]
                        else:
                            self.corr_time_err[it,ix] = np.nan
                        self.time_guess_osc = np.array([fit_t.best_values['l'],
                                                        fit_t.best_values['k'],
                                                        fit_t.best_values['p']])
                        self.time_plot(it, ix, max_index, peaks, 'oscillating', 
                                       omega=fit_t.best_values['k'])

                        logging.info("(" + str(it) + "," + str(ix) + ") was "
                                     "fitted with an oscillating Gaussian to "
                                     "the central peak. (tau, omega) = " 
                                     + str([fit_t.best_values['l'], 
                                            fit_t.best_values['k']]) + "\n")

                except RuntimeError:
                    logging.info("(" + str(it) + "," + str(ix) + ") "
                            "RuntimeError - max fitting iterations reached, "
                            "skipping this case with (tau, omega) = NaN\n")
                    self.corr_time[it, ix] = np.nan

    def time_plot(self, it, ix, max_index, peaks, plot_type, **kwargs):
        """
        Plots the time correlation peaks as well as the apprpriate fitting 
        function.

        Parameters
        ----------
        
        it : int
            Time slice currently being fitted.
        ix : int
            Radial index currently being fitted.
        max_index : array_like
            Array containing radial indices of the peaks. Size: (nx, npeaks_fit)
        peaks : array_like
            Array of peak values at the max_index values. Size: (nx, npeaks_fit)
        plot_type : str
            Type of fitting function to plot. One of: 'growing'/'decaying'/
            'oscillating'
        """
        plot_style.white()
            
        mid_idx = int(self.ny/2)
        plt.clf()
        fig, ax = plt.subplots(1, 1)

        if plot_type == 'decaying':
            plt.plot(self.dt*1e6, self.time_corr[it,:,ix,mid_idx:mid_idx+self.npeaks_fit])
            plt.hold(True)
            plt.plot(self.dt[max_index[ix,:]]*1e6, peaks[ix,:], 'o', color='#7A1919')
            plt.hold(True)
            plt.plot(self.dt[int(self.time_slice/2):]*1e6, 
                     fit.decaying_exp(self.dt[int(self.time_slice/2):],self.corr_time[it,ix]), 
                     'k--', lw=2, 
                     label=r'$\exp[-|\Delta t_{peak} / \tau_c|]$')
            plt.legend()
        if plot_type == 'growing':
            plt.plot(self.dt*1e6, self.time_corr[it,:,ix,mid_idx:mid_idx+self.npeaks_fit])
            plt.hold(True)
            plt.plot(self.dt[max_index[ix,:]]*1e6, peaks[ix,:], 'o', color='#7A1919')
            plt.hold(True)
            plt.plot(self.dt[:int(self.time_slice/2)]*1e6, 
                     fit.growing_exp(self.dt[:int(self.time_slice/2)],self.corr_time[it,ix]), 
                     'k--', lw=2, 
                     label=r'$\exp[|\Delta t_{peak} / \tau_c|]$')
            plt.legend()
        if plot_type == 'oscillating':
            plt.plot(self.dt*1e6, self.time_corr[it,:,ix,mid_idx])
            plt.hold(True)
            plt.plot(self.dt*1e6, fit.osc_gauss(self.dt,self.corr_time[it,ix], 
                     kwargs['omega'], 0), 'k--', lw=2, 
                     label=r'$\exp[- (\Delta t_{peak} / \tau_c)^2] '
                            '\cos(\omega \Delta t) $')
            plt.legend()

        plt.xlabel(r'$\Delta t (\mu s)})$')
        plt.ylabel(r'$C_{\Delta y}(\Delta t)$')
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/'+self.time_dir+'/corr_fns/time_fit_it_' +
                    str(it) + '_ix_' + str(ix) + '.pdf')
        plt.close(fig)

    def time_analysis_summary(self):
        """
        Prints out a summary of the time analysis.

        * Plots average correlation time as a function of radius.
        * Calculates standard deviation.
        * Writes summary to a text file.
        """
        logging.info("Writing time_analysis summary...")

        self.corr_time = np.abs(self.corr_time)

        np.savetxt(self.out_dir + '/'+self.time_dir+'/corr_time.csv', 
                   (self.corr_time), delimiter=',', fmt='%.4e', 
                   header='rows = radius, columns = time slice')
        np.savetxt(self.out_dir + '/'+self.time_dir+'/corr_time_err.csv', 
                   (self.corr_time_err), delimiter=',', fmt='%.4e', 
                   header='rows = radius, columns = time slice')

        # Plot corr_time as a function of radius, average over time window
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        t_error = np.nanstd(self.corr_time*1e6, axis=0)
        plt.errorbar(self.x, np.nanmean(self.corr_time*1e6, axis=0), yerr=t_error)
        plt.ylim(ymin=0)
        plt.xlabel("Radius (m)")
        plt.ylabel(r'Correlation Time $\tau_c$ ($\mu$ s)')
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/'+self.time_dir+'/corr_time.pdf')

        np.savetxt(self.out_dir + '/'+self.time_dir+'/time_fit_summary.csv',
                   (np.array([np.nanmean(self.corr_time), 
                   np.nanstd(self.corr_time)])*1e6)[np.newaxis,:],
                   header='tau_c, std(tau_c)')

        logging.info("Finished writing time_analysis summary...")

    def par_analysis(self):
        """
        Calculates the parallel correlation function and fits with a Gaussian
        to find the parallel correlation length.
        """
        logging.info("Starting par_analysis...")

        if 'parallel' not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/parallel')
        if 'corr_fns' not in os.listdir(self.out_dir + '/parallel'):
            os.system("mkdir " + self.out_dir + '/parallel/corr_fns')
        os.system('rm ' + self.out_dir + '/par/corr_fns/*')

        self.calculate_l_par()
        self.calculate_par_corr()

        self.par_fit_params = np.empty([self.nt_slices, 2], 
                                       dtype=float)
        self.par_fit_params_err = np.empty([self.nt_slices, 2], 
                                           dtype=float)

        for it in range(self.nt_slices):
            self.par_corr_fit(it)
        
        self.par_analysis_summary()

        logging.info("Finished par_analysis...")

    def calculate_l_par(self):
        """
        Calculates the real space parallel grid. 
        """
        logging.info('Start calculating parallel length...')

        dR_dtheta = np.gradient(self.R)/np.gradient(self.theta)
        dZ_dtheta = np.gradient(self.Z)/np.gradient(self.theta)

        dphi_dtheta = self.r_geo*self.bref / (self.R**2 * \
                        self.bmag[int(self.ntheta/2)] * self.gradpar)
        dl_dtheta = np.sqrt(dR_dtheta**2 + dZ_dtheta**2 + (self.R*dphi_dtheta)**2)
        self.l_par = np.append(0, integrate.cumtrapz(dl_dtheta, x=self.theta))

        logging.info('Finished calculating parallel length.')

    def calculate_par_corr(self):
        """
        Calculate the parallel correlation function and apply normalization mask.

        Interpolation onto a regular parallel grid, correlation calculation,
        and normalization are done in one function to avoid unnecessary looping 
        over x, y, and theta.
        """
        logging.info('Start calculating parallel correlation function...')

        x = np.ones([self.ntheta])
        mask = sig.correlate(x, x, 'same')

        self.par_corr = np.empty([self.nt, self.nx, self.ny, self.ntheta], 
                                 dtype=float)
        l_par_reg = np.linspace(0, self.l_par[-1], self.ntheta)
        for it in range(self.nt):
            logging.info('Parallel correlation timestep: %d of %d'%(it,self.nt))
            for ix in range(self.nx):
                for iy in range(self.ny):
                    f = interp.interp1d(self.l_par, 
                                        self.field_real_space[it,ix,iy,:])
                    self.field_real_space[it,ix,iy,:] = f(l_par_reg)

                    self.field_real_space[it,ix,iy,:] -= \
                            np.mean(self.field_real_space[it,ix,iy,:])
                    self.field_real_space[it,ix,iy,:] /= \
                            np.std(self.field_real_space[it,ix,iy,:])

                    self.par_corr[it,ix,iy,:] = \
                        sig.correlate(self.field_real_space[it,ix,iy,:], 
                                      self.field_real_space[it,ix,iy,:],
                                      'same')/mask

        self.l_par = l_par_reg
        self.dl_par = np.linspace(-self.l_par[-1]/2, self.l_par[-1]/2, self.ntheta)

        logging.info('Finished calculating parallel correlation function.')

    def par_corr_fit(self, it):
        """
        Fit the parallel correlation function with an oscillatory Gaussian 
        function for time slice it.

        Parameters
        ----------

        it : int
            Time slice to average over and fit.

        Before fitting average over time, x, and y.
        """     
        corr_fn = self.par_corr[it*self.time_slice:(it+1)*self.time_slice,:,:,:]
        corr_std = np.empty([self.ntheta]) 
        for i in range(self.ntheta):                                                        
                corr_std[i] = np.std(corr_fn[:,:,:,i])
        corr_fn = np.mean(np.mean(np.mean(corr_fn, axis=0), axis=0), axis=0)

        try:
            gmod_osc = lm.Model(fit.osc_gauss)
            params = lm.Parameters()
            params.add('l', value=self.par_guess[0], min=self.l_par[1], 
                       max=100)
            params.add('k', value=self.par_guess[1], 
                       max=(2*np.pi/self.dl_par[-1]*len(self.dl_par)/2))
            params.add('p', value=0, vary=False)
            par_fit = gmod_osc.fit(corr_fn, params, x=self.dl_par)
            
            self.par_fit_params[it, :] = np.abs([par_fit.best_values['l'], 
                                                 par_fit.best_values['k']])
            self.par_fit_params_err[it, :] = np.sqrt(np.diag(par_fit.covar))
            self.par_plot(it, corr_fn, corr_std)
        except RuntimeError:
            logging.info("(" + str(it) + ") RuntimeError - max fitting iterations reached, "
                    "skipping this case with (l_par, k_par) = NaN\n")
            self.par_fit_params[it, :] = np.nan
            self.par_fit_params_err[it, :] = np.nan

        self.par_guess = self.par_fit_params[it, :] 

    def par_plot(self, it, corr, corr_std):
        """
        Plots and saves the parallel correlation function and its fit for each
        time window.

        Parameters
        ----------
        it : int
            Time window being plotted.
        corr : array_like
            Parallel correlation averaged in t, x, and y.
        corr_std : array_like
            Standard deviation in the correlation function as a function of 
            dl_par.
        """
        plot_style.white()

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.scatter(self.dl_par, corr, c=pal[0], 
                    label=r'$C(\Delta t = 0, \Delta x = 0, \Delta y = 0, \Delta z)$')
        plt.fill_between(self.dl_par, corr-corr_std, corr+corr_std, 
                         alpha=0.3)
        plt.plot(self.dl_par, fit.osc_gauss(self.dl_par, self.par_fit_params[it,0], 
                 self.par_fit_params[it,1], 0), c=pal[2] ,
                 label=r'$p_\parallel + (1-p_\parallel)\exp[- (\Delta z / l_{\parallel})^2] '
                        '\cos(k_{\parallel} \Delta z) $')
        plt.plot(self.dl_par, np.exp(-(self.dl_par/self.par_fit_params[it,0])**2),
                 'k--', label='Gaussian Envelope')
        plt.legend()
        plt.xlabel(r'$\Delta z$ (m)')
        plt.ylabel(r'$C(\Delta z)$')
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/parallel/corr_fns/par_fit_it_' + 
                    str(it) + '.pdf')
        plt.close(fig)

    def par_analysis_summary(self):
        """
        Summarize parallel correlation analysis by plotting parallel correlation 
        fitting parameters along with associated errors and writing to a .csv
        file.
        """
        plot_style.white()

        np.savetxt(self.out_dir + '/parallel/par_fit_params.csv', 
                   (self.par_fit_params), delimiter=',', fmt='%1.4f')

        np.savetxt(self.out_dir + '/parallel/par_fit_summary.csv', 
                   np.nanmean([self.par_fit_params[:,0],
                   self.par_fit_params_err[:,0],
                   self.par_fit_params[:,1],
                   self.par_fit_params_err[:,1]], axis=1)[np.newaxis,:], 
                   delimiter=',', fmt='%1.4f',
                   header='l_par, err(l_par), k_par, err(k_par)')

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.errorbar(range(self.nt_slices), np.abs(self.par_fit_params[:,0]), 
                     yerr=self.par_fit_params_err[:,0])
        plt.xlabel('Time Window')
        plt.ylabel(r'Parallel Correlation Length $l_{\parallel} (m)$')
        plt.ylim(ymin=0, ymax=2*np.mean(np.abs(self.par_fit_params[:,0])))
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/parallel/par_fit_length_vs_time_slice.pdf')
        plt.close(fig)

        plt.clf()
        fig, ax = plt.subplots(1, 1)
        plt.errorbar(range(self.nt_slices), np.abs(self.par_fit_params[:,1]), 
                     yerr=self.par_fit_params_err[:,1])
        plt.xlabel('Time Window')
        plt.ylabel(r'Parallel Correlation Wavenumber $k_{\parallel} (m^{-1})$')
        plt.ylim(ymin=0, ymax=2*np.mean(np.abs(self.par_fit_params[:,1])))
        plot_style.minor_grid(ax)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/parallel/par_fit_wavenumber_vs_time_slice.pdf')
        plt.close(fig)

    def write_field(self):
        """
        Outputs the field to NetCDF in real space.

        Notes
        -----

        * The radial and poloidal coordinates are centered at 0.
        * The radial coordinate is interpolated if neccessary to ensure a
          0.5cm resolution consistent with the BES.
        """
        logging.info("Starting write_field...")
        
        if 'write_field' not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/write_field')

        if self.write_field_interp_x:
            #interpolate radial coordinate to be approx 0.5cm
            interp_fac = int(np.ceil(self.x[1]/0.005))
            x_nc = np.linspace(min(self.x), max(self.x), interp_fac*self.nx)
            field_real_space_nc = np.empty([self.nt, len(x_nc), self.ny], 
                                           dtype=float)
            for it in range(self.nt):
                for iy in range(self.ny):
                        f = interp.interp1d(self.x, 
                                            self.field_real_space[it,:,iy])
                        field_real_space_nc[it,:,iy] = f(x_nc)
        else:
            x_nc = self.x
            field_real_space_nc = self.field_real_space

        if self.lab_frame:
            nc_file = netcdf.netcdf_file(self.out_dir + '/write_field/' + 
                                         self.in_field +'_lab_frame.cdf', 'w')
        elif not self.lab_frame:
            nc_file = netcdf.netcdf_file(self.out_dir + '/write_field/' + 
                                         self.in_field +'.cdf', 'w')
        nc_file.createDimension('x', len(x_nc))
        nc_file.createDimension('y', self.ny)
        nc_file.createDimension('t', self.nt)
        nc_file.createDimension('none', 1)
        nc_nref = nc_file.createVariable('nref','d', ('none',))
        nc_tref = nc_file.createVariable('tref','d', ('none',))
        nc_x = nc_file.createVariable('x','d',('x',))
        nc_y = nc_file.createVariable('y','d',('y',))
        nc_t = nc_file.createVariable('t','d',('t',))
        nc_field = nc_file.createVariable(self.in_field[:self.in_field.find('_')],
                                      'd',('t', 'x', 'y'))

        nc_field[:,:,:] = field_real_space_nc[:,:,:]
        nc_nref[:] = self.nref
        nc_tref[:] = self.tref
        nc_x[:] = x_nc[:] - x_nc[-1]/2
        nc_y[:] = self.y[:] - self.y[-1]/2 
        nc_t[:] = self.t[:] - self.t[0]
        nc_file.close()
        
        logging.info("Finished write_field...")

    def make_film(self):
        """
        Creates film from real space field time frames.

        Image cropping using Pillow may be necessary to ensure width and height
        of images are even.
        """
        logging.info("Starting make_film...")

        plot_style.white()

        if self.lab_frame:
            self.film_dir = 'film_lab_frame'
        elif not self.lab_frame:
            self.film_dir = 'film'

        if self.film_dir not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/'+self.film_dir)
        if 'film_frames' not in os.listdir(self.out_dir+'/'+self.film_dir):
            os.system("mkdir " + self.out_dir + '/'+self.film_dir+'/film_frames')
        os.system("rm " + self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                  self.in_field + "_spec_" + str(self.spec_idx) + "*.png")

        self.field_max = np.max(self.field_real_space)
        self.field_min = np.min(self.field_real_space)
        for it in range(self.nt):
            self.plot_real_space_field(it)

        im = Image.open(self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                        self.in_field + "_spec_" + str(self.spec_idx) + 
                        "_0000.png")
        if im.size[0] % 2 != 0 or im.size[1] % 2 != 0:
            self.crop_images() 

        logging.info('avconv command: ')
        logging.info("avconv -threads 2 -y -f image2 -r " + str(self.film_fps) + 
                  " -i '" + self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                  self.in_field + "_spec_" + str(self.spec_idx) + 
                  "_%04d.png' -q 1 " + self.out_dir + '/'+self.film_dir+'/' + 
                  self.in_field + "_spec_" + str(self.spec_idx) +".mp4")
        os.system("avconv -threads 2 -y -f image2 -r " + str(self.film_fps) + 
                  " -i '" + self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                  self.in_field + "_spec_" + str(self.spec_idx) + "_%04d.png' -q 1 " + 
                  self.out_dir + '/'+self.film_dir+'/' + self.in_field + "_spec_" + 
                  str(self.spec_idx) +".mp4")

        logging.info("Finished make_film.")

    def plot_real_space_field(self, it):
        """
        Plots real space field and saves as a file indexed by time index.

        Parameters
        ----------
        it : int
            Time index to plot and save.
        """
        logging.info('Saving frame %d of %d'%(it,self.nt))

        if self.film_lim == None:
            contours = np.around(np.linspace(self.field_min, self.field_max, 
                                             self.film_contours),7)
            cbar_ticks = np.around(np.linspace(self.field_min, self.field_max, 
                                               5),3)
        else:
            contours = np.around(np.linspace(self.film_lim[0], self.film_lim[1], 
                                             self.film_contours),7)
            cbar_ticks = np.around(np.linspace(self.film_lim[0], self.film_lim[1], 
                                               5),3)

        plt.clf()
        ax = plt.subplot(111)
        im = ax.contourf(self.x, self.y, np.transpose(self.field_real_space[it,:,:]),
                     levels=contours, cmap='coolwarm')
        ax.set_aspect('equal')
        plt.xlabel(r'$x (m)$')
        plt.ylabel(r'$y (m)$')
        plt.title(r'Time = %04d $\mu s$'%(int(np.round((self.t[it]-self.t[0])*1e6))))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax, label=r'$\delta n / n (-)$', ticks=cbar_ticks)
        plot_style.ticks_bottom_left(ax)
        plt.savefig(self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                    self.in_field + "_spec_" + str(self.spec_idx) + 
                    "_%04d.png"%it, dpi=110, bbox_inches='tight')

    def crop_images(self):
        """
        Ensures that PNG files have height and width that are even. 
        
        This owes to a quirk of avconv and libx264 that requires this. At the 
        moment there is no easy way to specifically set the pixel count using 
        Matplotlib and this is not desirable anyway since plots can be almost 
        any size and aspect ratio depending on the data. The other solution 
        found online is to use the `-vf` flag for avconv to control the output
        size but this does not seem to work. The most reliable solution 
        therefore is to use Pillow to load and crop images.
        """
        logging.info("Cropping film images...")

        for it in range(self.nt):
            im = Image.open(self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                            self.in_field + "_spec_" + str(self.spec_idx) + 
                            "_%04d.png"%it)
            im_crop = im.crop((0, 0, int(im.size[0]/2)*2, int(im.size[1]/2)*2))
            im_crop.save(self.out_dir + '/'+self.film_dir+'/film_frames/' + 
                         self.in_field + "_spec_" + str(self.spec_idx) + 
                         "_%04d.png"%it)

        logging.info("Finished cropping film images.")

