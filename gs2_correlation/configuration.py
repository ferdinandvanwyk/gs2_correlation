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
.. module:: Configuration
   :platform: Unix, OSX
   :synopsis: Class describing configuration parameters.
                                                                                
.. moduleauthor:: Ferdinand van Wyk <ferdinandvwyk@gmail.com>                   
                                                                                
"""

# Standard
import os
import sys
import configparser
import logging

# Third Party

# Local

class Configuration(object):

    def __init__(self, config_file):
        """Initialize Configuration instance with config file filename.

        Parameters
        ----------
        config_file : str
            Filename of configuration file and path if not in the same 
            directory.

        Notes
        -----

        The configuration file should contain the following namelists: 

        * 'analysis' which gives information such as the analysis to be 
          performed and where to find the NetCDF file.
        * 'normalization' which gives the normalization parameters for the 
          simulation/experiment.

        The exact parameters read in are documented in the read_config method. 
        """

        self.config_file = config_file

    def read_config(self):
        """Reads analysis and normalization parameters from config file.

        The full list of possible configuration parameters is listed below, 
        along with their default values. The parameter names follow the 
        convention of <config namelist>.<parameter>.

        Parameters
        ----------
        analysis.file_ext : str
            File extension for NetCDF output file. Default = '.out.nc' 
        analysis.cdf_file : str
            Path (relative or absolute) and/or name of input NetCDF file. If
            only a path is specified, the directory is searched for a file
            ending in '.out.nc' and the name is appended to the path.
        analysis.field : str
            Name of the field to be read in from NetCDF file.
        analysis.analysis : str
            Type of analysis to be done. Options are 'full', 'perp', 'time', 
            'bes', 'zf'.
        analysis.out_dir : str
            Output directory for analysis. Default = './analysis'.
        analysis.interpolate : bool
            Interpolate in time onto a regular grid.
        analysis.zero_bes_scales : bool
            Zero out scales which are larger than the BES.
        analysis.species_index : int or None
            Specied index to be read from NetCDF file. GS2 convention is to use
            0 for ion and 1 for electron in a two species simulation.
        analysis.theta_index : int or None
            Parallel index at which to do analysis.
        normalization.amin : float
            Minor radius of device in *m*.
        normalization.vth : float
            Thermal velocity of the reference species in *m/s*
        normalization.rhoref : float
            Larmor radius of the reference species in *m*.
        normalization.pitch_angle : float
            Pitch angle of the magnetic field lines in *rad*.
        """

        logging.info('Started read_config...')

        config_parse = configparser.ConfigParser()
        config_parse.read(self.config_file)

        # Analysis information
        self.file_ext = config_parse.get('analysis', 'file_ext', 
                                         fallback='.out.nc')
        # Automatically find .out.nc file if only directory specified
        self.in_file = str(config_parse['analysis']['cdf_file'])
        if self.in_file.find(self.file_ext) == -1:
            dir_files = os.listdir(self.in_file)
            for s in dir_files:
                if s.find('self.file_ext') != -1:
                    self.in_file = self.in_file + s
                    break
                else:
                    raise NameError('No file found ending in ' + self.file_ext)
        self.in_field = str(config_parse['analysis']['field'])
        self.analysis = str(config_parse['analysis']['analysis'])
        self.out_dir = str(config_parse.get('analysis', 'out_dir', 
                                            fallback='analysis'))
        self.interpolate = bool(config_parse['analysis']['interpolate'])
        self.zero_bes_scales = str(config_parse['analysis']['zero_bes_scales'])
        if self.zero_bes_scales == "True":
            self.zero_bes_scales = True
        else:
            self.zero_bes_scales = False
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

        # Normalization parameters
        self.amin = float(config_parse['normalization']['a_minor'])
        self.vth = float(config_parse['normalization']['vth_ref'])
        self.rhoref = float(config_parse['normalization']['rho_ref'])
        self.pitch_angle = float(config_parse['normalization']['pitch_angle'])
                
        # Log the variables
        logging.info('The following values were read from ' + self.config_file)
        logging.info(vars(self))
        logging.info('Finished read_config.')













