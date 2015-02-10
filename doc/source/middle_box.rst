Middle Box Class
================

This class contains data and methods relating to correlation analysis of a box  
in the middle of the full GS2 domain. This is a separate class is since the     
calculation of the perpendicular correlation function can not take advantage of 
the Wiener-Khinchin theorem. Calculation of the perpendicular correlation       
function is done starting from real space using standard `SciPy` functions.

.. autoclass:: gs2_correlation.middle_box_analysis.MiddleBox
   :members:
   :special-members: __init__
