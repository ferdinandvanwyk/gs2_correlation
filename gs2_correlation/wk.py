def wk_thm_1d(field_1, field_2):
    field = np.conjugate(field_1)*field_2
    corr = np.fft.ifft(field)
    # multiply by number of pts to normalize correctly. See README.
    return corr.real*field.shape[0]
