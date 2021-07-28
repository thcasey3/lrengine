.. lsframe documentation master file, created by
   sphinx-quickstart on Fri Jan 10 17:26:18 2020.

.. .. image:: logo.png
..     :target: https://github.com/thcasey3/lsframe

.. |pypi| image:: https://img.shields.io/pypi/v/lsframe.svg?style=flat-square
    :target: https://pypi.org/project/lsframe/

.. |GitHub| image:: https://img.shields.io/badge/GitHub-development-blue?style=flat-square
    :target: https://github.com/thcasey3/lsframe/tree/development

.. |IssuesTracker| image:: https://img.shields.io/badge/GitHub-Issues-red?style=flat-square
    :target: https://github.com/thcasey3/lsframe/issues



|pypi| |GitHub| |IssuesTracker|

**lsframe** is an open-source Python package for mapping directories and correlating specific language patterns in file and folder names with each other and with outputs of user-supplied functions.

Features
========
* Create DataFrames that are maps of directories
* Organize DataFrames using dates/language in file/folder names
* Find possible dates in arbitrarily formatted file/folder names 
* Construct DataFrames from a function that processes files/folders of a parent directory
* Visualize correlations between dates/language in file/folder names and function outputs


Guide
=====

.. toctree::
   :maxdepth: 1

   install
   introduction
   create
   define_patterns
   reduce_names
   find_dates
   reduce_dates
   narrow_dates
   drive
   map_directory
   map_to_frame
   sea
   save
   classes
   auto_examples/index
   version_history


Documentation Build Date: |date|
