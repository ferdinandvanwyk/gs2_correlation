To Do
=====

* Do profiling and invesitigate possible perfromance enhancement using cython.
* Consider use of JSON object to save results of correlation analyses. At the moment
  writing a number of files containing results because some are 2D and some are just
  single floats. JSON would be perfect for this and reduce cluttered output dirs.
* Fix radial and perpendicular box sizes.
* Remove reading from '.g' file, all information now contained in new NetCDF
  file.
