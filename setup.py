#!/usr/bin/env python

from setuptools import setup

# Utilty function to read the README file. 
# Used for the long_description.  It's nice, because now 1) we have a top level 
# README file and 2) it's easier to type in the README file than to put a raw   
# string in below ...                                                           
def read(fname):                                                                
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='gs2_correlation',
      version='0.3.0',
      description='Calculates temporal and spatial correlation parameters from 
      GS2 fluctuation data.',
      author='Ferdinand van Wyk',
      author_email='ferdinandvwyk@gmail.com',
      url='https://github.com/ferdinandvwyk/gs2_correlation',
      packages=['gs2_correlation', 'test'],
      license='GPL',
      long_description=read('README.md'),
      classifiers=[                                                               
          "Development Status :: 3 - Alpha",                                      
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Programming Language :: Python",
      ],
     )
