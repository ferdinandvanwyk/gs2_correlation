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

# Standard imports
import os
import sys
import operator #enumerate list
import time
import gc #garbage collector
import configparser

# Third Party imports

# Local imports

class Configuration(object):

    def __init__(self, config_file):
        self.config_file = config_file

    def read_config(self):
        config_parse = configparser.ConfigParser()
        config_parse.read(self.config_file)

        # Analysis information
        self.in_field = str(config_parse['analysis']['field'])
        self.analysis = str(config_parse['analysis']['analysis'])
        self.in_file = str(config_parse['analysis']['cdf_file'])
        # Automatically find .out.nc file if only directory specified
        if self.in_file.find(".out.nc") == -1:
            in_dir_files = os.listdir(self.in_file)
            for s in in_dir_files:
                if s.find('.out.nc') != -1:
                    self.in_file = self.in_file + s
                    break

        self.out_dir = str(config_parse['analysis']['out_dir'])
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
            self.theta_idx = int(self.spec_idx)

        # Normalization parameters
        self.amin = float(config_parse['normalization']['a_minor']) # m
        self.vth = float(config_parse['normalization']['vth_ref']) # m/s
        self.rhoref = float(config_parse['normalization']['rho_ref']) # m
        self.pitch_angle = float(config_parse['normalization']['pitch_angle']) # in radians
                














