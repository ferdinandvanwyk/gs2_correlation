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

# Third Party
import numpy as np
from scipy.io import netcdf
import scipy.interpolate as interp
import scipy.optimize as opt
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.autolayout': True})
import seaborn as sns

# Local
import gs2_correlation.fit as fit

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
        ending in '.out.nc' and the name is appended to the path.
    field : str
        Name of the field to be read in from NetCDF file.
    analysis : str
        Type of analysis to be done. Options are 'all', 'perp', 'time', 'zf',
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
    pitch_angle : float
        Pitch angle of the magnetic field lines in *rad*.
    seaborn_context : str
        Context for plot output: paper, notebook, talk, poster. See: 
        http://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html
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
        Number of kx values. Also the number of real space x points.
    nky : int
        Number of ky values.
    ny : int
        Number of real space y. This is ny = 2*(nky - 1). 
    nt : int
        Number of time points.
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
        sns.set_style('darkgrid', {'axes.axisbelow':False, 'legend.frameon': True})

        self.read_netcdf()

        if self.out_dir not in os.listdir():
            os.system("mkdir " + self.out_dir)
        self.nt = len(self.t)
        self.nkx = len(self.kx)
        self.nky = len(self.ky)
        self.ny = 2*(self.nky - 1)

        self.x = np.linspace(0, 2*np.pi/self.kx[1], self.nkx)*self.rhoref
        self.y = np.linspace(0, 2*np.pi/self.ky[1], self.ny - 1)*self.rhoref \
                             *np.tan(self.pitch_angle)
        self.dx = np.linspace(-2*np.pi/self.kx[1], 2*np.pi/self.kx[1], 
                             self.nkx)*self.rhoref
        self.dy = np.linspace(-2*np.pi/self.ky[1], 2*np.pi/self.ky[1], 
                             self.ny - 1)*self.rhoref*np.tan(self.pitch_angle)
        self.fit_dx = self.dx[self.nkx/2 - self.perp_fit_length : 
                      self.nkx/2 + self.perp_fit_length]
        self.fit_dy = self.dy[(self.ny-1)/2 - self.perp_fit_length : 
                      (self.ny-1)/2 + self.perp_fit_length]
        self.fit_dx_mesh, self.fit_dy_mesh = np.meshgrid(self.fit_dx, self.fit_dy)
        self.fit_dx_mesh = np.transpose(self.fit_dx_mesh)
        self.fit_dy_mesh = np.transpose(self.fit_dy_mesh)

        if self.interpolate_bool:
            self.interpolate()

        if self.zero_bes_scales_bool:
            self.zero_bes_scales()

        if self.zero_zf_scales_bool:
            self.zero_zf_scales()

        self.to_complex()

    def read_config(self):
        """
        Reads analysis and normalization parameters from self.config_file.

        The full list of possible configuration parameters is listed below.

        Parameters
        ----------
        file_ext : str, '.cdf'
            File extension for NetCDF output file.
        cdf_file : str
            Path (relative or absolute) and/or name of input NetCDF file. If
            only a path is specified, the directory is searched for a file
            ending in '.out.nc' and the name is appended to the path.
        in_field : str
            Name of the field to be read in from NetCDF file.
        analysis : str
            Type of analysis to be done. Options are 'all', 'perp', 'time', 'zf',
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
        pitch_angle : float
            Pitch angle of the magnetic field lines in *rad*.
        seaborn_context : str
            Context for plot output: paper, notebook, talk, poster. See: 
            http://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html
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

        self.analysis = str(config_parse['analysis']['analysis'])
        if self.analysis not in ['perp', 'time', 'zf', 'write_field']:
            raise ValueError('Analysis must be one of (perp, time, zf, '
                             'write_field)')

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
                                               'perp_fit_length', fallback=50))

        self.perp_guess = str(config_parse['analysis']['perp_guess'])
        self.perp_guess = self.perp_guess[1:-1].split(',')
        self.perp_guess = [float(s) for s in self.perp_guess]
        self.perp_guess = self.perp_guess * np.array([self.rhoref, self.rhoref, 
                                             1/self.rhoref, 1/self.rhoref])
        self.perp_guess = list(self.perp_guess)

        ###################
        # Output Namelist #
        ###################

        self.seaborn_context = str(config_parse.get('output', 'seaborn_context', 
                                              fallback='talk'))

        # Log the variables
        logging.info('The following values were read from ' + self.config_file)
        logging.info(vars(self))
        logging.info('Finished read_config.')

    def read_netcdf(self):
        """
        Read array from NetCDF file.

        Read array specified in configuration file as 'cdf_field'. Function 
        uses information from the configuration object passed to it. 
        """
        logging.info('Start reading from NetCDf file...')

        # mmap=False does not read directly from cdf file. Copies are created.
        # This prevents seg faults when cdf file is closed after function exits
        ncfile = netcdf.netcdf_file(self.in_file, 'r', mmap=False)

        # NetCDF order is [t, species, ky, kx, theta, r]
        self.field = ncfile.variables[self.in_field][:,self.spec_idx,:,:,self.theta_idx,:]
        self.field = np.squeeze(self.field) 
        self.field = np.swapaxes(self.field, 1, 2)

        self.kx = ncfile.variables['kx'][:]
        self.ky = ncfile.variables['ky'][:]
        self.t = ncfile.variables['t'][:]

        logging.info('Finished reading from NetCDf file.')

    def interpolate(self):
        """
        Interpolates in time onto a regular grid

        Depending on whether the user specified to interpolate, the time grid
        is interpolated into a regular grid. This is required in order to do 
        FFTs in time. Interpolation is done by default if not specified.
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

    def to_complex(self):
        """
        Converts field to a complex array.

        Field is in the following format: field[t, kx, ky, ri] where ri 
        represents a dimension of length 2. 

        * ri = 0 - Real part of the field. 
        * ri = 1 - Imaginary part of the field. 
        """
        self.field = self.field[:,:,:,0] + 1j*self.field[:,:,:,1] 
    
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

        self.wk_2d() 

        nt_slices = int(self.nt/self.time_slice)
        self.perp_fit_params = np.empty([nt_slices, 4], dtype=float)

        for it in range(nt_slices):
            self.perp_fit(it)
            self.perp_guess = self.perp_fit_params[it,:]

        np.savetxt(self.out_dir + '/perp_fit_params.csv', (self.perp_fit_params), 
                   delimiter=',', fmt='%1.3f')

        self.perp_plots()
        self.perp_analysis_summary()

        logging.info('Finished perpendicular correlation analysis.')
        
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
        self.perp_corr = np.empty([self.nt, self.nkx, self.ny-1], dtype=float) 
        for it in range(self.nt):
            sq = np.abs(self.field[it,:,:])**2  
            self.perp_corr[it,:,:] = np.fft.irfft2(sq, s=[self.nkx, self.ny-1])
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
                                 self.nkx/2 - self.perp_fit_length : 
                                 self.nkx/2 + self.perp_fit_length, 
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
        
        #Time averaged correlation
        plt.clf()
        corr_fn = self.perp_corr[:, self.nkx/2 - self.perp_fit_length : 
                                    self.nkx/2 + self.perp_fit_length, 
                                    (self.ny-1)/2 - self.perp_fit_length : 
                                    (self.ny-1)/2 + self.perp_fit_length] 
        avg_corr = np.mean(corr_fn, axis=0) # Average over time
        plt.contourf(self.fit_dx, self.fit_dy, np.transpose(avg_corr), 11,
                     levels=np.linspace(-1, 1, 11), cmap='jet')
        plt.colorbar(ticks=np.linspace(-1, 1, 11))
        plt.xlabel(r'$\Delta x (m)$')
        plt.ylabel(r'$\Delta y (m)$')
        plt.savefig(self.out_dir + '/time_avg_correlation.pdf')

        # Tilted Gaussian using time-averaged fitting parameters 
        data_fitted = fit.tilted_gauss((self.fit_dx_mesh, self.fit_dy_mesh), 
                                        *np.mean(self.perp_fit_params, axis=0))
        plt.clf()
        plt.contourf(self.fit_dx, self.fit_dy, 
                     np.transpose(data_fitted.reshape(len(self.fit_dx),
                                                      len(self.fit_dy))), 
                                  11, levels=np.linspace(-1, 1, 11), cmap='jet')
        plt.title('$C_{fit}(\Delta x, \Delta y)$')
        plt.colorbar(ticks=np.linspace(-1, 1, 11))
        plt.xlabel(r'$\Delta x (m)$')
        plt.ylabel(r'$\Delta y (m)$')
        plt.savefig(self.out_dir + '/perp_corr_fit.pdf')

        # Avg correlation and fitted function overlayed
        plt.clf()
        plt.contourf(self.fit_dx, self.fit_dy, np.transpose(avg_corr), 10, 
                     levels=np.linspace(-1, 1, 11), cmap='jet')
        plt.colorbar(ticks=np.linspace(-1, 1, 11))
        plt.contour(self.fit_dx, self.fit_dy, 
                     np.transpose(data_fitted.reshape(len(self.fit_dx),
                                                      len(self.fit_dy))), 
                                  11, levels=np.linspace(-1, 1, 11), colors='k')
        plt.xlabel(r'$\Delta x (m)$')
        plt.ylabel(r'$\Delta y (m)$')
        plt.savefig(self.out_dir + '/perp_fit_comparison.pdf')

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
        plt.plot(self.perp_fit_params[:,2], label=r'$k_x (m^{-1}$')
        plt.plot(self.perp_fit_params[:,3], label=r'$k_y (m^{-1})$')
        plt.legend()
        plt.xlabel('Time Window')
        plt.yscale('log')
        plt.savefig(self.out_dir + '/perp_fit_params_vs_time_slice.pdf')

        summary_file = open(self.out_dir + '/perp_fit_summary.txt', 'w')
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

        * 
        """
        logging.info("Starting time_analysis...")

        self.field_to_real_space()




        logging.info("Finished time_analysis...")

    def field_to_real_space(self):
        """
        Converts field from kx, ky to x and y and saves as new attribute.
        """

        self.field_real_space = np.empty([nt,nx,ny],dtype=float)
        for it in range(nt):
            field_real_space[it,:,:] = np.fft.irfft2(real_to_complex_2d(
                                            field[it,:,:,:]), axes=[0,1])
            field_real_space[it,:,:] = np.roll(field_real_space[it,:,:], 
                                                 nx/2, axis=0)


















