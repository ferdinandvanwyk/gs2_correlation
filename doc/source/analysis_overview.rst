Analysis Overview
=================

This page describes the different analysis types that can be performed using
`gs2_correlation`, as well as additional useful features.

Perpendicular Correlation
-------------------------

In GS2, fields are written out as a function of kx and ky. This allows the use
of the Wiener-Khinchin theorem in 2D to calculate the correlation function:

.. math:: C(\Delta x, \Delta y) = IFFT2[|f(k_x, k_y)|^2]

where C is the correlation function, *f* is the field, and IFFT2 is the 2D 
inverse Fourier transform. Briefly, the perpendicular correlation analysis
performs the following:

* Calculates the 2D correlation function using the WK theorem.
* Splits the correlation function into time windows (of length *time_slice*, 
  as specified in the configuration file).
* Time averages those windows and fits them with a tilted Gaussian to find the
  correlation parameters lx, ly, kx, ky, theta.
* Writes correlation parameters into a csv file, with one row per time window.
* Generates and saves various plots of the true and fitted correlation functions.

Time Correlation
----------------

Calculating the correlation time consists of two main parts:

* Calculating the time correlation function.
* Fitting the correlation function with an appropriate function to get the
  correlation time.

Time Correlation Function
^^^^^^^^^^^^^^^^^^^^^^^^^

The field is firstly converted to real space and saved as a new variable called
*field_real_space*. This leaves us with a field *f(t, x, y)*. In order to have 
some statistics about how the correlation time is changing over the course of
the simulation, we split the time domain into time slices, of size *time_slice*
defined in the configuration file. The correlation time may also depend on the
size of this time slice, so some tests should be done to ensure that this is 
understood.

For each time slice we want to calculate the correlation function C(dt, x, dy), 
leaving us with a function C(it, dt, x, dy), where *it* denotes the time slice
index. This is done by using the SciPy function, ``scipy.signal.fftconvolve``.
Noting that a convolution and a correlation calculation is related by a 
reversal of the indices of the second function.

Create a Film
-------------

Zonal Flows
-----------


