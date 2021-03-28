.. |lrengineGitLink| raw:: html

   <a href="https://github.com/thcasey3/lrengine" target="_blank"> lrengine </a>


===================
Installing lrengine
===================

Required Packages
=================
The following packages are required in order to use all features of lrengine:

.. list-table::
   :widths: 40 60

   * - **Package**
     - **Version**
   * - Pandas
     - 1.2.3+
   * - NumPy
     - 1.20.0+
   * - dateutils
     - 0.6.12+
   * - seaborn
     - 0.11.1+


To install the required packages with pip use the command:

.. code-block:: bash

   $ python -m pip install pandas numpy dateutils seaborn


.. _installing:

Installing with pip
===================

.. code-block:: bash

   $ python -m pip install lrengine


Confirm Successful Installation
===============================
To confirm that your installation use the following command:

.. code-block:: bash

    $ pip show lrengine

Confirm the output is similar to:

.. code-block:: bash

    Name: lrengine
    Version: 0.0.1
    Summary: lrengine
    Home-page: http://github.com/thcasey3/lrengine
    Author: Thomas Casey
    Author-email: None
    License: MIT
    Location: /path_to/lrengine
    Requires: pandas, numpy, etc.

