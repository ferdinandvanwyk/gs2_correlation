gs2_correlation
==================

Python package to perform a full correlation analysis of GS2 fluctuations. This 
includes spatial (both radial and poloidal), and temporal correlation analysis.

Documentation
-------------

The documentation is hosted on [ReadtheDocs](www.readthedocs.org)

Getting Started
---------------

* Install the requirements by typing
```
pip install -r requirements.txt
```

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

film.py
-------

Contains functions for making films using ffmpeg.

fit.py
------

Contains functions for fitting perp and time correlation functions as well as 
plotting time correlation functions with with fitted functions.
