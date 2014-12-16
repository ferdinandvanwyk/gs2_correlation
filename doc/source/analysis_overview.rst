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
inverse Fourier transform.

Time Correlation
----------------

Create a Film
-------------

Zonal Flows
-----------


