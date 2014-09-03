gs2_correlation
==================

Several scripts to calculate correlation function, fit correlation function, write out field to poloidal plane.

correlation.py
--------------

This program calculates the space and time correlation function using the Wiener-Khinchin theorem and writes the output to a NetCDf file. Notes:

- x and y directions are already in Fourier space and periodic => simply use a 2D real FFT on squared field to find correlation function as a function of dx and dy
- Time correlation
  - Time series is real and needs to be put into fourier space before WK theorem can be used.
  - According to Bendat & Piersol (Sec 11.4): To avoid edge effects due to a non-periodic finite time series, need to pad the time series with zeros. If time series is of length N, pad with N zeros => after FT 0...N-1 will contain the required correlation function and N...2N+1 will contain second part of _circular correlation function_.
  - Second part can be discarded.

film.py
-------

Contains functions for making films.

fit.py
------

Contains functions for fitting perp and time correlation functions.
