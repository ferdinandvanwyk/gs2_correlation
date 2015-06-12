Simulation Class
===================

This class contains data and methods relating to the simulation.

Instance Variables of Simulation Class
--------------------------------------

.. py:class:: Simulation(config_file)

   Attributes
   ----------
   file_ext : str
       File extension for NetCDF output file. Default = '.cdf'
   run_folder : str, '../..'
       Path to run folder.
   cdf_file : str, None
       Path (relative or absolute) and name of input NetCDF file. If
       None, the directory is searched for a file ending in '.cdf' and the 
       name is appended to the run_folder path.
   g_file : str, None
       Path to the '.g' file. If None, run_folder will be searched and the
       first returned file will be used.
   in_file : str, None
       Path to '.in' file.
   geometry : array_like
       Array containing entire '.g' file.
   input_file : dict
       Dictionary containing all namelist variables from the '.inp' file
       produced by GS2.
   in_field : str
       Name of the field to be read in from NetCDF file.
   analysis : str
       Type of analysis to be done. Options are 'all', 'perp', 'par', 'time', 
       'write_field'.
   out_dir : str, 'analysis'
       Output directory for analysis.
   time_interpolate_bool : bool, True
       Interpolate in time onto a regular grid. Specify as interpolate in 
       configuration file.
   time_interp_fac : int, 1
       Sets the time interpolation multiple.
   zero_bes_scales_bool : bool, False
       Zero out scales which are larger than the BES. Specify as 
       zero_bes_scales in configuration file.
   zero_zf_scales_bool : bool, False
       Zero out the zonal flow (ky = 0) modes. Specify as zero_zf_scales in 
       configuration file.
   lab_frame : bool, False
       Transform from rotating to lab frame.
   domain : str, 'full'
       Specifies whether to analyze the full real space domain, or only the 
       middle part of size *box_size*.
   time_slice : int
       Size of time window for averaging
   perp_fit_length : int
       Number of points radially and poloidally to fit Gaussian over. Fitting
       over the whole domain usually does not produce a very good fit. Default
        = 20.
   perp_guess_x : array_like
       Initial guess for radial correlation length in normalized rho_ref units.
   perp_guess_y : array_like
       Initial guess for poloidal correlation length in normalized rho_ref 
       units.
   ky_free : bool, False
      Determines whether ky is free during the poloidal fitting procedure.
   time_guess : int
       Initial guess for the correlation time in normalized GS2 units.
   box_size : array_like, [0.2,0.2]
       When running correlation analysis in the middle of the full GS2
       domain, this sets the approximate [radial, poloidal] size of this 
       box in m. This variable is only used when domain = 'middle'
   time_range : array_like, [0,-1]
       Time range for which analysis is done. Default is entire range. -1
       for the final time step is interpreted as up to the final time step,
       inclusively.
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
   bref : float
       Reference magnetic field at the centre of the LCFS.
   rho_ref : float
       Larmor radius of the reference species in *m*.
   rho_star : float
       The expansion parameter defined as rho_ref/amin.
   pitch_angle : float
       Pitch angle of the magnetic field lines in *rad*.
   rmaj : float, 0
       Major radius of the outboard midplane. Used when writing out 
       the field to the NetCDF file. This is **not** the *rmaj* value from
       GS2.
   nref : float, 1
       Density of the reference species in m^-3.
   tref : float, 1
       Temperature of the reference species in eV.
   omega : float, 0
       Angular frequency of the plasma at the radial location of the flux tube.
   dpsi_da : float, 0
       Relationship between psi_n and rho_miller (a_n = diameter/diameter LCFS)
   drho_dpsi: float, 1
       Gradient of flux surface label with respect to psi.
   gradpar : array_like
       Value of the parallel gradient as a function of theta.
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
   field_real_space_norm_x : array_like
       Field in real space coordinates (x,y) normalized in the x dimension.
   field_real_space_norm_y : array_like
       Field in real space coordinates (x,y) normalized in the y dimension.
   perp_corr_x : array_like
       Radial correlation function calculated from field_real_space_norm_x.
   perp_corr_y : array_like
       Poloidal correlation function calculated from field_real_space_norm_y.
   perp_fit_len_x : array_like
       Radial correlation length obtained from perp fitting procedure. Of size 
       (nt_slices).
   perp_fit_len_err_x : array_like
       Error in the radial correlation length obtained from perp fitting 
       procedure, calculated from the covariance matrix. Of size (nt_slices).
   perp_fit_len_y : array_like
       Poloidal correlation length obtained from perp fitting procedure. Of size 
       (nt_slices).
   perp_fit_len_err_y : array_like
       Error in the poloidal correlation length obtained from perp fitting 
       procedure, calculated from the covariance matrix. Of size (nt_slices).
   time_corr : array_like
       Correlation function used to calculate the correlation function. It is
       of size (nt_slices, time_slice, nx, ny), owing to the 2D 
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
   write_field_interp_x : bool, True
       Determines whether the radial coordinate is interpolated to match
       the BES resolution when writing to a NetCDF file.
   r : array_like
       Radial coordinate *x*, centered at the major radius *rmaj*.
   z : array_like
       Poloidal coordinate *z* centered at 0.
   fluc_level : float
       Mean fluctuation level for all space and time.
   fluc_level_std : float
       Standard deviation of the mean fluctuation level.
   R : array_like
       Radial location of centre of the flux tube as read in from geometry
       file.
   Z : array_like
       Poloidal location of centre of the flux tube as read in from geometry
       file.
   phi_tor : array_like
       Toroidal angle of location of centre of the flux tube as read in from 
       geometry file. It is equal to -alpha4 in geometry file.
   dR_drho : array_like
       Derivative of R with respect to the radial coordinate rho.
   dZ_drho : array_like
       Derivative of Z with respect to the radial coordinate rho.
   dalpha_drho : array_like
       Derivative of alpha with respect to the radial coordinate rho.
   bpol : array_like
       Poloidal magnetic field as a function of theta as printed out in the 
       geometry file.
   btor : float
       Toroidal magnetic field at the radial location of the flux tube.
       btor = [r_geo/R(theta=0)]*bref
   bmag : float
       Magnitude of the magnetic field at the location of the flux tube.
   r_geo : float
       Radial location of the reference magnetic field.
   par_corr : array_like
      Parallel correlation function C(t,x,y,dtheta)
   l_par : array_like
       Regular real space parallel grid. 
   dl_par : array_like
       Values of parallel separation in real space.
   par_guess : array_like, [1,0.1]
      Initial guess for the parallel fitting in SI units in the form 
      [l_par, k_par].
   par_fit_params : array_like
      Array which stores parallel correlation fitting parameters. It is a 
      function of time slice and contains both l and k which define an
      oscillating Gaussian.
   par_fit_params_err : array_like
      Stores errors associated with fitting the parallel correlation function.
      These are calculated by taking the square root of the diagonal terms in
      the covariance matrix and give the error in each fitting parameter.

Configuration Variables 
-----------------------

.. py:method:: read_config()

   The following parameters can be set in the configuration file '<name>.ini'.

   Parameters
   ----------
   file_ext : str, '.cdf'
       File extension for NetCDF output file.
   run_folder : str, '../..'
       Path to run folder.
   cdf_file : str, None
       Path (relative or absolute) and name of input NetCDF file. If
       None, the directory is searched for a file ending in '.cdf' and the 
       name is appended to the run_folder path.
   g_file : str, None
       Path to the '.g' file. If None, run_folder will be searched and the
       first returned file will be used.
   in_file : str, None
       Path to '.in' file.
   field : str
       Name of the field to be read in from NetCDF file.
   analysis : str, 'all'
       Type of analysis to be done. Options are 'all', 'perp', 'par, 
       'time', 'write_field', 'film'.
   out_dir : str, 'analysis'
       Output directory for analysis.
   time_interpolate : bool, True
       Interpolate in time onto a regular grid.
   time_interp_fac : int, 1
       Sets the time interpolation multiple.
   zero_bes_scales : bool, False
       Zero out scales which are larger than the BES.
   zero_zf_scales : bool, False
       Zero out the zonal flow (ky = 0) modes.
   lab_frame : bool, False
       Transform from rotating to lab frame.
   domain : str, 'full'
       Specifies whether to analyze the full real space domain, or only the 
       middle part of size *box_size*.
   time_slice : int, 50
       Size of time window for averaging
   perp_guess : array_like, [1,1]
       Initial guess for the radial and poloidal correlation lengths. Of the
       form [lx, ly] in normalized rho_ref units.
   ky_free : bool, False
      Determines whether ky is free during the poloidal fitting procedure.
   time_guess : int, 10
       Initial guess for the correlation time in normalized GS2 units.
   box_size : array_like, [0.2,0.2]
       When running correlation analysis in the middle of the full GS2
       domain, this sets the approximate [radial, poloidal] size of this 
       box in m. This variable is only used when domain = 'middle'
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
   geom_file : str
       Location of the geometry file. By default searches the run folder
       for a '.g' file and loads the first one found.
   par_guess : array_like, [1,0.1]
      Initial guess for the parallel fitting in SI units in the form 
      [l_par, k_par].
   amin : float
       Minor radius of device in *m*.
   vth : float
       Thermal velocity of the reference species in *m/s*
   bref : float
       Reference magnetic field at the centre of the LCFS.
   rho_ref : float
       Larmor radius of the reference species in *m*.
   nref : float, 1
       Density of the reference species in m^-3.
   tref : float, 1
       Temperature of the reference species in eV.
   omega : float, 0
       Angular frequency of the plasma at the radial location of the flux tube.
   dpsi_da : float, 0
       Relationship between psi_n and rho_miller (a_n = diameter/diameter LCFS)
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
   write_field_interp_x : bool, True
       Determines whether the radial coordinate is interpolated to match
       the BES resolution when writing to a NetCDF file.

Method Documentation
--------------------

.. autoclass:: gs2_correlation.simulation.Simulation
   :members:
   :special-members: __init__
