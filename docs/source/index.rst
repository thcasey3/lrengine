.. lsframe documentation master file, created by
   sphinx-quickstart on Sun Oct 31 12:00:00 2021.

.. .. image:: logo.png
..     :target: https://github.com/thcasey3/lsframe

.. |pypi| image:: https://img.shields.io/pypi/v/lsframe.svg?style=flat-square
    :target: https://pypi.org/project/lsframe/

.. |GitHub| image:: https://img.shields.io/badge/GitHub-development-blue?style=flat-square
    :target: https://github.com/thcasey3/lsframe/tree/development

.. |IssuesTracker| image:: https://img.shields.io/badge/GitHub-Issues-yellow?style=flat-square
    :target: https://github.com/thcasey3/lsframe/issues



|pypi| |GitHub| |IssuesTracker|

**lsframe** is an open-source Python package for mapping directories and correlating specific language patterns in file and folder names with each other and with outputs of user-supplied functions that act on the files or folders.

Features
========
* Create DataFrames that are maps of directories
* Organize DataFrames using dates and language in file and folder names
* Find possible dates in arbitrarily formatted file and folder names 
* Construct DataFrames from a function that processes files and folders
* Visualize correlations between dates or language and function outputs
* Use machine learning, time series analysis, and anomaly detection


Guide
=====

.. toctree::
   :maxdepth: 1

   introduction
   user_guide
   examples
   simple class
   ls operations
   sklearn interface
   statsmodels interface
   adtk interface
   Using pipeline
   version_history


Documentation Build Date: |date|
