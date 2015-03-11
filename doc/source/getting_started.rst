Getting Started
===============

Python Version Support
----------------------

At the moment, `gs2_correlation` requires at least Python 3.4. No support for 
previous versions is planned due to backwards incompatibility, however getting
it to run under Python 2 shouldn't be a problem due to the use of very standard
and otherwise well supported libraries.

Install Dependencies
--------------------

The dependencies are listed in the requirements.txt file and are installed by
running:

.. code:: bash

    $ pip install -r requirements.txt

Since this project is structured as a PIP package, it also needs to be installed
using the following command (in the package root directory):

.. code:: bash

    $ pip install -e .


Running Tests
-------------

`gs2_correlation` uses the pytest framework for unit and functional tests. To 
run the tests, run the following in the package root directory:

.. code:: bash

    $ py.test

Documentation
-------------

The documentation is completely built on Sphinx with `numpydoc` docstring 
convention and is hosted on `Read the Docs`_: gs2-correlation.rtfd.org_. Using 
RTD/GitHub webhooks, the documentation is rebuilt upon every commit that makes
changes to the documentation files The current build status is shown by the 
``docs`` badge at the top of the main page. To make the docs, run:

.. code:: bash

    $ cd doc
    $ make html

where ``html`` can be replaced with other acceptable formats, such as latex,
latexpdf, text, etc. In order to view the Latex document, it first has to be 
built:

.. code:: bash

   $ cd build/latex
   $ make

Continuous Integration and Testing
----------------------------------

`gs2_correlation` utilizes the Travis_ continuous integration (CI) framework
to build and run the tests upon every push to the GitHub repository. The current 
build status is shown by the ``build`` badge at the top of the main page and 
build history can be found by clicking on the badge or the following link: 

https://travis-ci.org/ferdinandvwyk/gs2_correlation

As well as this, source code test coverage is measured and reported using 
Coveralls_. The Coveralls page for this project can found by clicking on the 
``coverage`` badge or by the following link:

https://coveralls.io/r/ferdinandvwyk/gs2_correlation

.. _Read the Docs: https://readthedocs.org/ 
.. _gs2-correlation.rtfd.org : http://gs2-correlation.rtfd.org
.. _Travis: https://travis-ci.org/
.. _Coveralls: https://coveralls.io/
