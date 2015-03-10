gs2_correlation
==================

[![Build Status](https://travis-ci.org/ferdinandvwyk/gs2_correlation.svg?branch=master)](https://travis-ci.org/ferdinandvwyk/gs2_correlation)
[![Coverage Status](https://coveralls.io/repos/ferdinandvwyk/gs2_correlation/badge.svg)](https://coveralls.io/r/ferdinandvwyk/gs2_correlation)
[![Documentation Status](https://readthedocs.org/projects/gs2-correlation/badge/?version=latest)](https://readthedocs.org/projects/gs2-correlation/?badge=latest)

gs2_correlation is python package to perform a full correlation analysis of GS2 
fluctuations. This includes spatial (both radial and poloidal), and temporal 
correlation analysis.

Documentation
-------------

The full documentation is hosted on [ReadtheDocs](https://www.readthedocs.org):
[gs2-correlation.rtfd.org](http://gs2-correlation.rtfd.org)

correlation.py
--------------

This program calculates the space and time correlation function using the 
Wiener-Khinchin theorem and writes the output to a NetCDf file. Notes:

- x and y directions are already in Fourier space and periodic => simply use a 
2D real FFT on squared field to find correlation function as a function of dx 
and dy
- Time Correlation
  - Calculation of the time correlation follows Y.C. Ghim's procedure 
    (PRL 2013).
  - For each radial grid point, calculate the function C(dt,dy) using the 
    Python correlate routines.
  - Take a few dy values around dy = 0 and fit the peaks of the function 
    C_dy(dt) with a decaying exponential.
  - If there is no flow, the dy = 0 C(dt) function will be fitted with a 
    Gaussian function.
  - This method will break down when the flow is too fast. This usually 
    manifests itself as a significantly shorter time correlation.

