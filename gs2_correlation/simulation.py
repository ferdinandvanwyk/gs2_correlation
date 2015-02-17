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

class Simulation(object):
    """
    Class containing all simulation information.

    The class mainly reads from the simulation NetCDF file and operates on the
    field specified in the configuration file, such as performing correlations,
    FFTs, plotting, making films etc.

    Attributes
    ----------

    file_ext : str
        File extension for NetCDF output file. Default = '.cdf'
    cdf_file : str
        Path (relative or absolute) and/or name of input NetCDF file. If
        only a path is specified, the directory is searched for a file
        ending in '.cdf' and the name is appended to the path.
    in_field : str
        Name of the field to be read in from NetCDF file.
    analysis : str
        Type of analysis to be done. Options are 'all', 'perp', 'time', 
        'write_field'.
    out_dir : str
        Output directory for analysis: "analysis".
    interpolate_bool : bool
        Interpolate in time onto a regular grid. Default = True. Specify as
        interpolate in configuration file.
    zero_bes_scales_bool : bool
        Zero out scales which are larger than the BES. Default = False. Specify
        as zero_bes_scales in configuration file.
    zero_zf_scales_bool : bool
        Zero out the zonal flow (ky = 0) modes. Default = False. Specify as
        zero_zf_scales in configuration file.
    time_slice : int
        Size of time window for averaging
    perp_fit_length : int
        Number of points radially and poloidally to fit Gaussian over. Fitting
        over the whole domain usually does not produce a very good fit. Default
         = 20.
    perp_guess : array_like
        Initial guess for perpendicular correlation function fitting. Of the
        form [lx, ly, kx, ky] all in normalized rhoref units.
    time_guess : int
        Initial guess for the correlation time in normalized GS2 units.
    npeaks_fit : int
        Number of peaks to fit when calculating the correlation time.
    species_index : int
        Specied index to be read from NetCDF file. GS2 convention is to use
        0 for ion and 1 for electron in a two species simulation.
    theta_index : int or None
        Parallel index at which to do analysis. If no theta index in array
        set to None.
    amin : float
        Minor radius of device in *m*.
    vth : float
        Thermal velocity of the reference species in *m/s*
    rhoref : float
        Larmor radius of the reference species in *m*.
    rho_star : float
        The expansion parameter defined as rho_ref/amin.
    pitch_angle : float
        Pitch angle of the magnetic field lines in *rad*.
    seaborn_context : str
        Context for plot output: paper, notebook, talk, poster. See:
        http://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html
    film_fps : int, 40
        Frames per second of the film.
    film_contours : int, 30
        Number of contours to use when making films. More contours => bigger 
        files.
    field : array_like
        Field read in from the NetCDF file. Automatically converted to a complex
        array.
    field_real_space : array_like
        Field in real space coordinates (x,y)
    perp_corr : array_like
        Perpendicular correlation function calculated from the field array.
    perp_fit_params : array_like
        Parameters obtained from perp fitting procedure. Of size (nt_slices, 4) 
        since fitting finds lx, ly, kx, ky.
    time_corr : array_like
        Correlation function used to calculate the correlation function. It is
        of size (nt_slices, 2*time_slice-1, nx, 2*ny-1), owing to the 2D 
        correlation calculated in the t and y directions.
    corr_time : array_like
        Parameters obtained from time fitting procedure. Of size (nt_slices, nx).
    kx : array_like
        Values of the kx grid in the following order: 0,...,kx_max,-kx_max,...
        kx_min.
    ky : array_like
        Values of the ky grid.
    t : array_like
        Values of the time grid.
    dt : array_like
        Values of the time separations. Min and max values will depend on
        time_slice.
    nt_slices : int
        Number of time slices. nt_slices = nt/time_slice 
    x : array_like
        Values of the real space x (radial) grid.
    dx : array_like
        Values of the dx (radial separation) grid.
    fit_dx : array_like
        dx values for section that will be fitted
    fit_dx_mesh : array_like
        dx grid (in fitting region) as a 2D mesh.
    y : array_like
        Values of the real space y (poloidal) grid. This has been transformed
        from the toroidal plane to the poloidal plane by using the pitch-angle
        of the magnetic field lines.
    dy : array_like
        Values of the dy (poloidal separation) grid.
    fit_dy : array_like
        dy values for section that will be fitted
    fit_dy_mesh : array_like
        dy grid (in fitting region) as a 2D mesh.
    nkx : int
        Number of kx values.
    nky : int
        Number of ky values.
    nx : int
        Number of real space x points.
    ny : int
        Number of real space y points. This is ny = 2*(nky - 1).
    nt : int
        Number of time points.
    ncfile : object
        SciPy NetCDF object referencing all arrays in the NetCDF file. Arrays
        are not loaded into memory due to the large amount of memory needed, 
        but simply read from the NetCDF file. Since the field is manipulated, 
        it is copied into a NumPy array, however.
    field_max : float
        Maximum value of the real space field.
    field_min : float
        Minimum value of the real space field.
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
        * Calculates the real space field.

        Parameters
        ----------
        config_file : str
            Filename of configuration file and path if not in the same
            directory.

        Notes
        -----

        The configuration file should contain the following namelists:

        * 'analysis': information such as the analysis to be performed and
          where to find the NetCDF file.
        * 'normalization': normalization parameters for the simulation/experiment.
        * 'output': parameters which relate to the output produced by the code.

        The exact parameters read in are documented in the Attributes above.
        """

        self.config_file = config_file
        self.read_config()

        # Set plot options
        sns.set_context(self.seaborn_context)

        self.rho_star = self.rhoref/self.amin

        self.read_netcdf()

        if self.out_dir not in os.listdir():
            os.system("mkdir " + self.out_dir)
        self.nt = len(self.t)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.nx = self.nkx
        self.ny = 2*(self.nky - 1)

        self.nt_slices = int(self.nt/self.time_slice)
        self.t = self.t*self.amin/self.vth
        self.time_guess = self.time_guess*self.amin/self.vth
        self.x = np.linspace(0, 2*np.pi/self.kx[1], self.nx)*self.rhoref
        self.y = np.linspace(0, 2*np.pi/self.ky[1], self.ny)*self.rhoref \
                             *np.tan(self.pitch_angle)
        self.dx = np.linspace(-2*np.pi/self.kx[1], 2*np.pi/self.kx[1],
                             self.nkx)*self.rhoref
        self.dy = np.linspace(-2*np.pi/self.ky[1], 2*np.pi/self.ky[1],
                             self.ny - 1)*self.rhoref*np.tan(self.pitch_angle)
        self.fit_dx = self.dx[int(self.nkx/2) - self.perp_fit_length :
                      int(self.nkx/2) + self.perp_fit_length]
        self.fit_dy = self.dy[int((self.ny-1)/2) - self.perp_fit_length :
                      int((self.ny-1)/2) + self.perp_fit_length]
        self.fit_dx_mesh, self.fit_dy_mesh = np.meshgrid(self.fit_dx, self.fit_dy)
        self.fit_dx_mesh = np.transpose(self.fit_dx_mesh)
        self.fit_dy_mesh = np.transpose(self.fit_dy_mesh)

        if self.interpolate_bool:
            self.interpolate()

        if self.zero_bes_scales_bool:
            self.zero_bes_scales()

        if self.zero_zf_scales_bool:
            self.zero_zf_scales()

        self.field_to_complex()
        self.fourier_correction()
        self.field_to_real_space()

    def read_config(self):
        """
        Reads analysis and normalization parameters from self.config_file.

        The full list of possible configuration parameters is listed below.

        Parameters
        ----------
        file_ext : str, '.cdf'
            File extension for NetCDF output file.
        cdf_file : str, '*.cdf'
            Path (relative or absolute) and/or name of input NetCDF file. If
            only a path is specified, the directory is searched for a file
            ending in '.cdf' and the name is appended to the path.
        field : str
            Name of the field to be read in from NetCDF file.
        analysis : str, 'all'
            Type of analysis to be done. Options are 'all', 'perp', 'time',
            'write_field', 'film'.
        out_dir : str, 'analysis'
            Output directory for analysis.
        interpolate : bool, True
            Interpolate in time onto a regular grid.
        zero_bes_scales : bool, False
            Zero out scales which are larger than the BES.
        zero_zf_scales : bool, False
            Zero out the zonal flow (ky = 0) modes.
        time_slice : int, 50
            Size of time window for averaging
        perp_fit_length : int, 20
            Number of points radially and poloidally to fit Gaussian over. 
            Fitting over the whole domain usually does not produce a very good 
            fit.
        perp_guess : array_like, [1,1,1,1]
            Initial guess for perpendicular correlation function fitting. Of the
            form [lx, ly, kx, ky] all in normalized rhoref units.
        time_guess : int, 10
            Initial guess for the correlation time in normalized GS2 units.
        box_size : array_like, [0.1,0.1]
            When running correlation analysis in the middle of the full GS2
            domain, this sets the approximate [radial, poloidal] size of this 
            box in m. This variable is only used when using the 'middle' 
            command line parameter.
        time_range : array_like, [0,-1]
            Time range for which analysis is done. Default is entire range. -1
            for the final time step is interpreted as up to the final time step,
            inclusively.
        npeaks_fit : int, 5
            Number of peaks to fit when calculating the correlation time.
        species_index : int or None
            Specied index to be read from NetCDF file. GS2 convention is to use
            0 for ion and 1 for electron in a two species simulation. If reading
            phi or simulation with only one species, set to None.
        theta_index : int or None
            Parallel index at which to do analysis. If no theta index in array
            set to None.
        amin : float
            Minor radius of device in *m*.
        vth : float
            Thermal velocity of the reference species in *m/s*
        rhoref : float
            Larmor radius of the reference species in *m*.
        pitch_angle : float
            Pitch angle of the magnetic field lines in *rad*.
        r_maj : float, 0
            Major radius of the outboard midplane. Used when writing out 
            the field to the NetCDF file. This is **not** the *rmaj* value from
            GS2.
        seaborn_context : str, 'talk'
            Context for plot output: paper, notebook, talk, poster. See:
            http://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html
        film_fps : int, 40
            Frames per second of the film.
        film_contours : int, 30
            Number of contours to use when making films. More contours => bigger 
            files.
        film_lim : array_like, None
            This sets the min and max contour levels when making films. None 
            means that the contour min and max are automatically calculated.
        """
        logging.info('Started read_config...')

        config_parse = configparser.ConfigParser()
        config_parse.read(self.config_file)

        ##########################
        # Normalization Namelist #
        ##########################

        self.amin = float(config_parse['normalization']['a_minor'])
        self.vth = float(config_parse['normalization']['vth_ref'])
        self.rhoref = float(config_parse['normalization']['rho_ref'])
        self.pitch_angle = float(config_parse['normalization']['pitch_angle'])
        self.rmaj = float(config_parse.get('normalization', 'rmaj', fallback=0))
        self.nref = float(config_parse.get('normalization', 'nref', fallback=1))
        self.tref = float(config_parse.get('normalization', 'tref', fallback=1))

        #####################
        # Analysis Namelist #
        #####################

        self.file_ext = config_parse.get('analysis', 'file_ext',
                                         fallback='.cdf')
        # Automatically find .out.nc file if only directory specified
        self.in_file = str(config_parse['analysis']['cdf_file'])
        if self.in_file.find(self.file_ext) == -1:
            dir_files = os.listdir(self.in_file)
            found = False
            for s in dir_files:
                if s.find(self.file_ext) != -1:
                    self.in_file = self.in_file + s
                    found = True
                    break

            if not found:
                raise NameError('No file found ending in ' + self.file_ext)

        self.in_field = str(config_parse['analysis']['field'])

        self.analysis = config_parse.get('analysis', 'analysis',
                                         fallback='all')
        if self.analysis not in ['all', 'perp', 'time', 'write_field', 
                                 'film']:
            raise ValueError('Analysis must be one of (perp, time, '
                             'write_field, make_film)')

        self.out_dir = str(config_parse.get('analysis', 'out_dir',
                                            fallback='analysis'))

        self.interpolate_bool = config_parse.getboolean('analysis', 'interpolate',
                                             fallback=True)

        self.zero_bes_scales_bool = config_parse.getboolean('analysis',
                                   'zero_bes_scales', fallback=False)

        self.zero_zf_scales_bool = config_parse.getboolean('analysis',
                                   'zero_zf_scales', fallback=False)

        self.spec_idx = str(config_parse['analysis']['species_index'])
        if self.spec_idx == "None":
            self.spec_idx = None
        else:
            self.spec_idx = int(self.spec_idx)

        self.theta_idx = str(config_parse['analysis']['theta_index'])
        if self.theta_idx == "None":
            self.theta_idx = None
        else:
            self.theta_idx = int(self.theta_idx)

        self.time_slice = int(config_parse.get('analysis', 'time_slice',
                                               fallback=50))

        self.perp_fit_length = int(config_parse.get('analysis',
                                               'perp_fit_length', fallback=20))

        self.perp_guess = str(config_parse['analysis']['perp_guess'])
        self.perp_guess = self.perp_guess[1:-1].split(',')
        self.perp_guess = [float(s) for s in self.perp_guess]
        self.perp_guess = self.perp_guess * np.array([self.rhoref, self.rhoref,
                                             1/self.rhoref, 1/self.rhoref])
        self.perp_guess = list(self.perp_guess)

        self.npeaks_fit = int(config_parse.get('analysis',
                                               'npeaks_fit', fallback=5))
        self.time_guess = int(config_parse.get('analysis',
                                               'time_guess', fallback=10))

        self.box_size = str(config_parse.get('analysis',
                                               'box_size', fallback='[0.1,0.1]'))
        self.box_size = self.box_size[1:-1].split(',')
        self.box_size = [float(s) for s in self.box_size]

        self.time_range = str(config_parse.get('analysis',
                                               'time_range', fallback='[0,-1]'))
        self.time_range = self.time_range[1:-1].split(',')
        self.time_range = [float(s) for s in self.time_range]

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

        # Log the variables
        logging.info('The following values were read from ' + self.config_file)
        logging.info(vars(self))
        logging.info('Finished read_config.')

    def read_netcdf(self):
        """
        Read array from NetCDF file.

        Read array specified in configuration file as 'in_field'. Function
        uses information from the configuration object passed to it.
        """
        logging.info('Start reading from NetCDf file...')

        self.ncfile = netcdf.netcdf_file(self.in_file, 'r')

        # NetCDF order is [t, species, ky, kx, theta, r]
        # ncfile.variable returns netcdf object - convert to array
        if self.time_range[1] == -1:
            self.field = np.array(self.ncfile.variables[self.in_field]
                                            [self.time_range[0]:,
                                             self.spec_idx,:,:,self.theta_idx,:])
            self.t = np.array(self.ncfile.variables['t'][self.time_range[0]:])
        else:
            self.field = np.array(self.ncfile.variables[self.in_field]
                                            [self.time_range[0]:self.time_range[1],
                                             self.spec_idx,:,:,self.theta_idx,:])
            self.t = np.array(self.ncfile.variables['t'][self.time_range[0]:
                                                         self.time_range[1]])


        self.field = np.squeeze(self.field)
        self.field = np.swapaxes(self.field, 1, 2)

        self.kx = np.array(self.ncfile.variables['kx'][:])
        self.ky = np.array(self.ncfile.variables['ky'][:])

        logging.info('Finished reading from NetCDf file.')

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
        
        self.field[:,:,1:] = self.field[:,:,1:]/2

    def interpolate(self):
        """
        Interpolates in time onto a regular grid

        Depending on whether the user specified to interpolate, the time grid
        is interpolated into a regular grid. This is required in order to do
        FFTs in time. Interpolation is done by default if not specified.
        """
        logging.info('Started interpolating onto a regular grid...')

        t_reg = np.linspace(min(self.t), max(self.t), self.nt)
        for i in range(len(self.kx)):
            for j in range(len(self.ky)):
                for k in range(2):
                    f = interp.interp1d(self.t, self.field[:, i, j, k], axis=0)
                    self.field[:, i, j, k] = f(t_reg)
        self.t = t_reg

        logging.info('Finished interpolating onto a regular grid.')

    def zero_bes_scales(self):
        """
        Sets modes larger than the BES to zero.

        The BES is approximately 160x80mm(rad x pol), so we would set kx < 0.25
        and ky < 0.5 to zero, since k = 2 pi / L.
        """
        for ikx in range(len(self.kx)):
            for iky in range(len(self.ky)):
                # Roughly the size of BES (160x80mm)
                if abs(self.kx[ikx]) < 0.25 and self.ky[iky] < 0.5:
                    self.field[:,ikx,iky,:] = 0.0

    def zero_zf_scales(self):
        """
        Sets zonal flow (ky = 0) modes to zero.
        """
        self.field[:,:,0,:] = 0.0

    def field_to_complex(self):
        """
        Converts field to a complex array.

        Field is in the following format: field[t, kx, ky, ri] where ri
        represents a dimension of length 2.

        * ri = 0 - Real part of the field.
        * ri = 1 - Imaginary part of the field.
        """
        self.field = self.field[:,:,:,0] + 1j*self.field[:,:,:,1]

    def wk_2d(self):
        """
        Calculates perpendicular correlation function for each time step.

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

        The normalization required to be consistent with GS2 and `numpy` FFT
        packages is:

        C_norm = C * nkx * ny / 2

        Since GS2 normalizes going from real to spectral space, no additional
        factors are necessary when going from spectral to real. However, `numpy`
        FFT packages contain an implicit normalization in the inverse routines
        such that ifft(fft(x)) = x, so we multiply to removethese implicit
        factors.
        """

        logging.info("Performing 2D WK theorem on field...")

        # ny-1 below since to ensure odd number of y points and that zero is in
        # the middle of the y domain.
        self.perp_corr = np.empty([self.nt, self.nx, self.ny-1], dtype=float)
        for it in range(self.nt):
            sq = np.abs(self.field[it,:,:])**2
            self.perp_corr[it,:,:] = np.fft.irfft2(sq, s=[self.nx, self.ny-1])
            self.perp_corr[it,:,:] = np.fft.fftshift(self.perp_corr[it,:,:])
            self.perp_corr[it,:,:] = (self.perp_corr[it,:,:] /
                                      np.max(self.perp_corr[it,:,:]))

        logging.info("Finished 2D WK theorem.")

    def perp_fit(self, it):
        """
        Fits tilted Gaussian to perpendicular correlation function.

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being fitted.
        """
        corr_fn = self.perp_corr[it*self.time_slice:(it+1)*self.time_slice,
                                 self.nx/2 - self.perp_fit_length :
                                 self.nx/2 + self.perp_fit_length,
                                 (self.ny-1)/2 - self.perp_fit_length :
                                 (self.ny-1)/2 + self.perp_fit_length]

        # Average corr_fn over time
        avg_corr = np.mean(corr_fn, axis=0)

        popt, pcov = opt.curve_fit(fit.tilted_gauss, (self.fit_dx_mesh,
                                                      self.fit_dy_mesh),
                                   avg_corr.ravel(), p0=self.perp_guess)

        self.perp_fit_params[it, :] = popt

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
        corr_fn = self.perp_corr[:, int(self.nx/2) - self.perp_fit_length :
                                    int(self.nx/2) + self.perp_fit_length,
                                    int((self.ny-1)/2) - self.perp_fit_length :
                                    int((self.ny-1)/2) + self.perp_fit_length]

        avg_corr = np.mean(corr_fn, axis=0) # Average over time

        plt.clf()
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

    def perp_analysis_summary(self):
        """
        Prints out a summary of the perpendicular analysis.

        * Plots fitting parameters as a function of time window.
        * Averages them in time and calculates a standard deviation.
        * Writes summary to a text file.
        """
        logging.info("Writing perp_analysis summary...")

        plt.clf()
        plt.plot(self.perp_fit_params[:,0], label=r'$l_x (m)$')
        plt.plot(self.perp_fit_params[:,1], label=r'$l_y (m)$')
        plt.plot(self.perp_fit_params[:,2], label=r'$k_x (m^{-1})$')
        plt.plot(self.perp_fit_params[:,3], label=r'$k_y (m^{-1})$')
        plt.legend()
        plt.xlabel('Time Window')
        plt.yscale('log')
        plt.savefig(self.out_dir + '/perp/perp_fit_params_vs_time_slice.pdf')

        summary_file = open(self.out_dir + '/perp/perp_fit_summary.txt', 'w')
        summary_file.write('lx = ' + str(np.mean(self.perp_fit_params[:,0]))
                           + " m\n")
        summary_file.write('std(lx) = ' + str(np.std(self.perp_fit_params[:,0]))
                           + " m\n")
        summary_file.write('ly = ' + str(np.mean(self.perp_fit_params[:,1]))
                           + " m\n")
        summary_file.write('std(ly) = ' + str(np.std(self.perp_fit_params[:,1]))
                           + " m\n")
        summary_file.write('kx = ' + str(np.mean(self.perp_fit_params[:,2]))
                           + " m^-1\n")
        summary_file.write('std(kx) = ' + str(np.std(self.perp_fit_params[:,2]))
                           + " m^-1\n")
        summary_file.write('ky = ' + str(np.mean(self.perp_fit_params[:,3]))
                           + " m^-1\n")
        summary_file.write('std(ky) = ' + str(np.std(self.perp_fit_params[:,3]))
                           + " m^-1\n")
        summary_file.write('theta = ' + str(np.arctan(np.mean(self.perp_fit_params[:,2]/ \
                            self.perp_fit_params[:,3]))) + "\n")
        summary_file.write('std(theta) = ' + str(np.std(self.perp_fit_params[:,3]/\
                           self.perp_fit_params[:,3])) + "\n")
        summary_file.close()

        logging.info("Finished writing perp_analysis summary...")

    def time_analysis(self):
        """
        Performs a time correlation analysis on the field.

        Notes
        -----

        * Change from (kx, ky) to (x, y)
        * Split into time windows and perform correlation analysis on each 
          window separately.
        """
        logging.info("Starting time_analysis...")

        if 'time' not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/time')
        if 'corr_fns' not in os.listdir(self.out_dir+'/time'):
            os.system("mkdir " + self.out_dir + '/time/corr_fns')
        os.system('rm ' + self.out_dir + '/time/corr_fns/*')

        
        self.time_corr = np.empty([self.nt_slices, 2*self.time_slice-1, self.nx,
                                   2*self.ny-1], dtype=float)
        self.corr_time = np.empty([self.nt_slices, self.nx], dtype=float)

        for it in range(self.nt_slices):
            self.calculate_time_corr(it)
            self.time_corr_fit(it)

        self.time_analysis_summary()

        logging.info("Finished time_analysis...")

    def field_to_real_space(self):
        """
        Converts field from (kx, ky) to (x, y) and saves as new array attribute.

        Notes
        -----

        * Since python defines x = IFFT[FFT(x)] need to undo the implicit 
          normalization by multiplying by the size of the arrays.
        * GS2 fluctuations are O(rho_star) and must be multiplied by rho_star
          to get their true values.
        """

        self.field_real_space = np.empty([self.nt,self.nx,self.ny],dtype=float)
        for it in range(self.nt):
            self.field_real_space[it,:,:] = np.fft.irfft2(self.field[it,:,:])
            self.field_real_space[it,:,:] = np.roll(self.field_real_space[it,:,:],
                                                    int(self.nx/2), axis=0)

        self.field_real_space = self.field_real_space*self.nx*self.ny/2
        self.field_real_space = self.field_real_space*self.rho_star

    def calculate_time_corr(self, it):
        """
        Calculate the time correlation for a given time window at each x.

        Parameters
        ----------

        it : int
            This is the index of the time slice currently being calculated.
        """
        
        field_window = self.field_real_space[it*self.time_slice:(it+1)*
                                             self.time_slice,:,:]

        for ix in range(self.nx):
            self.time_corr[it,:,ix,:] = sig.fftconvolve(field_window[:,ix,:], 
                                                        field_window[::-1,ix,::-1])
            self.time_corr[it,:,ix,:] = (self.time_corr[it,:,ix,:] /  
                                        np.max(self.time_corr[it,:,ix,:]))

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
        self.dt = np.linspace(-max(t)+t[0], max(t)-t[0], 2*self.time_slice-1)

        peaks = np.zeros([self.nx, self.npeaks_fit], dtype=float)
        max_index = np.empty([self.nx, self.npeaks_fit], dtype=int);
        mid_idx = self.ny-1

        for ix in range(self.nx):
            for iy in range(mid_idx,mid_idx+self.npeaks_fit):
                max_index[ix, iy-mid_idx], peaks[ix, iy-mid_idx] = \
                    max(enumerate(self.time_corr[it,:,ix,iy]), 
                        key=operator.itemgetter(1))

            if (fit.strictly_increasing(max_index[ix,:]) == True or 
                fit.strictly_increasing(max_index[ix,::-1]) == True):
                if max_index[ix, self.npeaks_fit-1] > max_index[ix, 0]:
                    self.corr_time[it, ix], pcov = opt.curve_fit(
                                                       fit.decaying_exp, 
                                                       (self.dt[max_index[ix,:]]), 
                                                       peaks[ix,:].ravel(), 
                                                       p0=self.time_guess)
                    self.time_plot(it, ix, max_index, peaks, 'decaying')
                    logging.info("(" + str(it) + "," + str(ix) + ") was fitted "
                                 "with decaying exponential. tau = " 
                                 + str(self.corr_time[it,ix]) + " s\n")
                else:
                    self.corr_time[it, ix], pcov = opt.curve_fit(
                                                       fit.growing_exp, 
                                                       (self.dt[max_index[ix,:]]), 
                                                       peaks[ix,:].ravel(), 
                                                       p0=self.time_guess)
                    self.time_plot(it, ix, max_index, peaks, 'growing')
                    logging.info("(" + str(it) + "," + str(ix) + ") was fitted "
                                 "with growing exponential. tau = " 
                                 + str(self.corr_time[it,ix]) + " s\n")
            else:
                # If abs(max_index) is not monotonically increasing, this 
                # usually means that there is no flow and that the above method
                # cannot be used to calculate the correlation time. Try fitting
                # a decaying oscillating exponential to the central peak.
                self.time_corr[it,:,ix,mid_idx] = self.time_corr[it,:,ix,mid_idx]/ \
                                                  max(self.time_corr[it,:,ix,mid_idx])
                init_guess = (self.time_guess, 1.0)
                tau_and_omega, pcov = opt.curve_fit(
                                          fit.osc_exp, 
                                          (self.dt), 
                                          (self.time_corr[it,:,ix,mid_idx]).ravel(), 
                                          p0=init_guess)
                self.corr_time[it,ix] = tau_and_omega[0]
                self.time_plot(it, ix, max_index, peaks, 'oscillating', 
                               omega=tau_and_omega[1])
                logging.info("(" + str(it) + "," + str(ix) + ") was fitted "
                             "with an oscillating "
                             "Gaussian to the central peak. (tau, omega) = " 
                             + str(tau_and_omega) + "\n")

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
        sns.set_style('darkgrid', {'axes.axisbelow':True, 'legend.frameon': True})
            
        mid_idx = self.ny-1
        plt.clf()
        plt.plot(self.dt*1e6, self.time_corr[it,:,ix,mid_idx:mid_idx+self.npeaks_fit])
        plt.hold(True)
        plt.plot(self.dt[max_index[ix,:]]*1e6, peaks[ix,:], 'o', color='#7A1919')
        plt.hold(True)

        if plot_type == 'decaying':
            plt.plot(self.dt[self.time_slice-1:]*1e6, 
                     fit.decaying_exp(self.dt[self.time_slice-1:],self.corr_time[it,ix]), 
                     color='#3333AD', lw=2, 
                     label=r'$\exp[-|\Delta t_{peak} / \tau_c|]$')
            plt.legend()
        if plot_type == 'growing':
            plt.plot(self.dt[:self.time_slice-1]*1e6, 
                     fit.growing_exp(self.dt[:self.time_slice-1],self.corr_time[it,ix]), 
                     color='#3333AD', lw=2, 
                     label=r'$\exp[|\Delta t_{peak} / \tau_c|]$')
            plt.legend()
        if plot_type == 'oscillating':
            plt.plot(self.dt*1e6, fit.osc_exp(self.dt,self.corr_time[it,ix], 
                     kwargs['omega']), color='#3333AD', lw=2, 
                     label=r'$\exp[- (\Delta t_{peak} / \tau_c)^2] '
                            '\cos(\omega \Delta t) $')
            plt.legend()

        plt.xlabel(r'$\Delta t (\mu s)})$')
        plt.ylabel(r'$C_{\Delta y}(\Delta t)$')
        plt.savefig(self.out_dir + '/time/corr_fns/time_fit_it_' + str(it) + 
                    '_ix_' + str(ix) + '.pdf')

    def time_analysis_summary(self):
        """
        Prints out a summary of the time analysis.

        * Plots average correlation time as a function of radius.
        * Calculates standard deviation.
        * Writes summary to a text file.
        """
        logging.info("Writing time_analysis summary...")

        np.savetxt(self.out_dir + '/time/corr_time.csv', (self.corr_time),
                   delimiter=',', fmt='%.4e')

        self.corr_time = np.abs(self.corr_time)

        # Plot corr_time as a function of radius, average over time window
        plt.clf()
        plt.plot(self.x, np.mean(self.corr_time*1e6, axis=0))
        plt.ylim(ymin=0)
        plt.xlabel("Radius (m)")
        plt.ylabel(r'Correlations Time $\tau_c$ ($\mu$ s)')
        plt.savefig(self.out_dir + '/time/corr_time.pdf')

        summary_file = open(self.out_dir + '/time/time_fit_summary.txt', 'w')
        summary_file.write('tau_c = ' + str(np.mean(self.corr_time)*1e6)
                           + " mu s\n")
        summary_file.write('std(tau_c) = ' + str(np.std(np.mean(self.corr_time, axis=0))*1e6)
                           + " mu s\n")
        summary_file.close()

        logging.info("Finished writing time_analysis summary...")

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

        #interpolate radial coordinate to be approx 0.5cm
        interp_fac = int(np.ceil(self.x[2]/0.005))
        x_bes = np.linspace(min(self.x), max(self.x), interp_fac*self.nx)
        field_real_space_interp = np.empty([self.nt, len(x_bes), self.ny], 
                                           dtype=float)
        for it in range(self.nt):
            for iy in range(self.ny):
                f = interp.interp1d(self.x, self.field_real_space[it,:,iy])
                field_real_space_interp[it,:,iy] = f(x_bes)

        nc_file = netcdf.netcdf_file(self.out_dir + '/write_field/' + 
                                     self.in_field +'.cdf', 'w')
        nc_file.createDimension('x', len(x_bes))
        nc_file.createDimension('y', self.ny)
        nc_file.createDimension('t', self.nt)
        nc_file.createDimension('none', 1)
        nc_nref = nc_file.createVariable('nref','d', ('none',))
        nc_tref = nc_file.createVariable('tref','d', ('none',))
        nc_x = nc_file.createVariable('x','d',('x',))
        nc_y = nc_file.createVariable('y','d',('y',))
        nc_t = nc_file.createVariable('t','d',('t',))
        nc_ntot = nc_file.createVariable('ntot','d',('t', 'x', 'y',))
        nc_nref[:] = self.nref
        nc_tref[:] = self.tref
        nc_x[:] = x_bes[:] - x_bes[-1]/2
        nc_y[:] = self.y[:] - self.y[-1]/2 
        nc_t[:] = self.t[:] - self.t[0]
        nc_ntot[:,:,:] = field_real_space_interp[:,:,:]
        nc_file.close()
        
        logging.info("Finished write_field...")

    def make_film(self):
        """
        Creates film from real space field time frames.
        """
        logging.info("Starting make_film...")

        if 'film' not in os.listdir(self.out_dir):
            os.system("mkdir " + self.out_dir + '/film')
        if 'film_frames' not in os.listdir(self.out_dir+'/film'):
            os.system("mkdir " + self.out_dir + '/film/film_frames')
        os.system("rm " + self.out_dir + "/film/film_frames/" + self.in_field + 
                  "_spec_" + str(self.spec_idx) + "*.png")

        self.field_max = np.max(self.field_real_space)
        self.field_min = np.min(self.field_real_space)
        for it in range(self.nt):
            self.plot_real_space_field(it)

        os.system("avconv -threads 2 -y -f image2 -r " + str(self.film_fps) + 
                  " -i '" + self.out_dir + "/film/film_frames/" + self.in_field + 
                  "_spec_" + str(self.spec_idx) + "_%04d.png' " + 
                  self.out_dir +"/film/" + self.in_field + "_spec_" + 
                  str(self.spec_idx) +".mp4")

        logging.info("Finished make_film...")

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
        else:
            contours = np.around(np.linspace(self.film_lim[0], self.film_lim[1], 
                                             self.film_contours),7)

        plt.clf()
        plt.contourf(self.x, self.y, np.transpose(self.field_real_space[it,:,:]),
                     levels=contours, cmap='coolwarm')
        plt.xlabel(r'$x (m)$')
        plt.ylabel(r'$y (m)$')
        plt.title(r'Time = %f $\mu s$'%((self.t[it]-self.t[0])*1e6))
        plt.colorbar()
        plt.savefig(self.out_dir + "/film/film_frames/" + self.in_field + 
                    "_spec_" + str(self.spec_idx) + "_%04d.png"%it, dpi=110)
        
        
        























