Full Box Class
==============

This class contains data and methods relating to correlation analysis of the    
full GS2 domain. It inherets all methods of the Simulation class and adds a     
function which calls the Wiener-Khinchin theorem when calculating the           
perpendicular correlation function. The fitting functions are contained in the  
simulation class and are independent of how the perpendicular correlation       
function is calculated.

.. autoclass:: gs2_correlation.full_box_analysis.FullBox
   :members:
   :special-members: __init__
