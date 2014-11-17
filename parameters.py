############################
# gs2_correlation_analysis #
#    Ferdinand van Wyk     #
############################

###############################################################################
# This file is part of gs2_correlation_analysis.
#
# gs2_correlation_analysis is free software: you can redistribute it and/or 
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gs2_correlation_analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gs2_correlation_analysis.  
# If not, see <http://www.gnu.org/licenses/>.
###############################################################################


class Parameters(object):
    """ This class  handles reading and manipulating configuration parameters 
    and arrays which are read in from NetCDF files based on the configuration 
    parameters.
    """

    def __init__(self, config_file):
        self.config_file = config_file

    def read_config(self):
        
        

