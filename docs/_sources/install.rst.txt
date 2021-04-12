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
   * - pandas
     - 1.0.4+
   * - numpy
     - 1.13.3+
   * - python-dateutil
     - 2.6.1+
   * - seaborn
     - 0.10+

To install the required packages with pip use the command:

.. code-block:: bash

   $ pip install pandas numpy python-dateutil seaborn


.. _installing:

Installing with pip
===================

.. code-block:: bash

   $ pip install lrengine


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
    Home-page: https://github.com/thcasey3/lrengine
    Author: Thomas Casey
    Author-email: None
    License: MIT
    Location: /path_to/lrengine
    Requires: pandas, numpy, etc.

