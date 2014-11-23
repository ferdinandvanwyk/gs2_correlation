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

# Standard
import os
import sys
import logging
import time

# Third Party

# Local
import configuration
import simulation

#############
# Main Code #
#############

# Set up logging framework
logging.basicConfig(filename='main.log', level=logging.INFO)
logging.info('')
logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
logging.info('')

# Create and Configuration object
conf = configuration.Configuration('config.ini')

#Create Simulation object
run = simulation.Simulation(conf)

if conf.analysis == 'all':
    run.perp_analysis(conf)
    run.time_analysis(conf)
    run.zf_analysis(conf)
    run.field_write(conf)
elif conf.analysis == 'perp':
    run.perp_analysis(conf)
elif conf.analysis == 'time':
    run.time_analysis(conf)
elif conf.analysis == 'zf':
    run.zf_analysis(conf)
elif conf.analysis == 'write_field':
    run.write_field(conf)















