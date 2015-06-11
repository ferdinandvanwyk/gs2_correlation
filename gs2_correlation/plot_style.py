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
.. module:: plot_styles
   :platform: Unix, OSX
   :synopsis: Functions setting plot aesthetics.

.. moduleauthor:: Ferdinand van Wyk <ferdinandvwyk@gmail.com>

"""

# Third Party
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams.update({'figure.autolayout': True})
mpl.rcParams['axes.unicode_minus']=False

def white():
    """
    Set plot aesthetics for 1D plots with white backgrounds.

    The plots are set up to have
    * White backgrounds
    * Grey minor and major gridlines
    * x and y ticks on bottom and left axes 
    * Thin black outer border
    * Minor grid lines halfway between major ticks
    """
    sns.set_style('whitegrid', {'axes.edgecolor':'0.1', 
                                'legend.frameon': True,
                                'xtick.color': '.15',
                                'xtick.major.size': 5,
                                'xtick.minor.size': 0.0,
                                'xtick.direction': 'out',
                                'ytick.color': '.15',
                                'ytick.major.size': 5,
                                'ytick.minor.size': 0.0,
                                'ytick.direction': 'out',
                                'axes.axisbelow': True,
                                'axes.linewidth': 0.4,
                                'font.family' : 'sans-serif',                       
                                'font.sans-serif' : ['Helvetica', 'Arial',          
                                                     'Verdana', 'sans-serif'] 
                                })

def minor_grid(ax):
    """
    Sets up a minor grid halfway between major ticks.

    This is mainly for 1D whitegrid plots since the gridlines are grey.
    """
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator( (plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 ))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator( (plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 ))
    ax.grid(True, 'major', color='0.92', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.92', linestyle='-', linewidth=0.7)

def dark():
    """
    Set plot aesthetics for 2D contour plots.

    The plots are set up to have
    * 'Dark' backgrounds (not usually visible)
    * White major gridlines in front of the plot
    * x and y ticks on bottom and left axes 
    * Thin black outer border
    """
    sns.set_style('darkgrid', {'axes.edgecolor':'0.1', 
                               'legend.frameon': True,
                               'xtick.color': '.15',
                               'xtick.major.size': 5,
                               'xtick.minor.size': 0.0,
                               'xtick.direction': 'out',
                               'ytick.color': '.15',
                               'ytick.major.size': 5,
                               'ytick.minor.size': 0.0,
                               'ytick.direction': 'out',
                               'axes.axisbelow': False,
                               'axes.linewidth': 0.4,
                               'font.family' : 'sans-serif',                       
                               'font.sans-serif' : ['Helvetica', 'Arial',          
                                                     'Verdana', 'sans-serif'] 
                               })

def ticks_bottom_left(ax):
    """
    Sets the ticks to be bottom left only.
    """
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')





