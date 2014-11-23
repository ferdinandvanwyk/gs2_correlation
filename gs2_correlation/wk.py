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

