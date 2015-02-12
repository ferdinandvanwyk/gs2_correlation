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
import argparse

# Third Party

# Local
import full_box_analysis
import middle_box_analysis

#############
# Main Code #
#############

# Get command line argument specifying configuration file
parser = argparse.ArgumentParser(description='Perform correlation and other '
                                 'analyses')
parser.add_argument('config_file', metavar='config_file', type=str, 
                    help='Location of the configuration file')
parser.add_argument('size', metavar='size', type=str, 
        help='Size of box to analyze: full/middle')
args = parser.parse_args()

# Set up logging framework
logging.basicConfig(filename='main.log', level=logging.INFO)
logging.info('')
logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
logging.info('')

#Create Simulation object

if args.size == 'full':
    run = full_box_analysis.FullBox(args.config_file)
elif args.size == 'middle':
    run = middle_box_analysis.MiddleBox(args.config_file)

if run.analysis == 'perp':
    run.perp_analysis()
elif run.analysis == 'time':
    run.time_analysis()
elif run.analysis == 'write_field':
    run.write_field()
elif run.analysis == 'film':
    run.make_film()














