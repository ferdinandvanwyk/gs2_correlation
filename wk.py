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

# This file contains functions which relate to use of the Wiener Khinchin
# Theorem, including functions that create complex fields out of arrays read
# in from the NetCDF file (1D and 2D) and the actual implementation of the WK
# theorem in 1D and 2D.

# Function which converts from GS2 field to complex field which can be passed 
# to fft routines
def r_to_c_1d(field):
    n1 = field.shape[0]
    cplx_field = np.empty([n1],dtype=complex)
    cplx_field = field[:,0] + 1j*field[:,1]
    return cplx_field

def r_to_c_2d(field):
    [nk1, nk2] = [ field.shape[0], field.shape[1]]
    n1 = nk1; n2 = (nk2-1)*2 #assuming second index is half complex
    cplx_field = np.empty([nk1,nk2],dtype=complex)
    cplx_field = field[:,:,0] + 1j*field[:,:,1]
    return cplx_field

def wk_thm_1d(field_1, field_2):
    field = np.conjugate(field_1)*field_2
    corr = np.fft.ifft(field)
    # multiply by number of pts to normalize correctly. See README.
    return corr.real*field.shape[0]

# Function which applies WK theorem to a real 2D field field(x,y,ri) where y is
# assumed to be half complex and 'ri' indicates the real/imaginary axis 
# (0 real, 1 imag). The output is the correlation function C(dx, dy).
def wk_thm_2d(c_field):
    # The WK thm states that the autocorrelation function is the 
    # FFT of the power spectrum. The power spectrum is defined as abs(A)**2 
    # where A is a COMPLEX array. In this case f.
    c_field = np.abs(c_field**2)
    # option 's' below truncates by ny by 1 such that an odd number of y pts 
    # are output => need to have corr fn at (0,0)
    corr = np.fft.irfft2(c_field,axes=[0,1], s=[c_field.shape[0], 
                         2*(c_field.shape[1]-1)-1])
    # multiply by number of pts to normalize correctly. See README.
    return corr*c_field.shape[0]*c_field.shape[1]/2

