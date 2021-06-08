.. lrengine documentation master file, created by
   sphinx-quickstart on Fri Jan 10 17:26:18 2020.

.. .. image:: logo.png
..     :target: https://github.com/thcasey3/lrengine

.. |pypi| image:: https://img.shields.io/pypi/v/lrengine.svg?style=flat-square
    :target: https://pypi.org/project/lrengine/

.. |GitHub| image:: https://img.shields.io/badge/GitHub-development-blue?style=flat-square
    :target: https://github.com/thcasey3/lrengine/tree/development

.. |IssuesTracker| image:: https://img.shields.io/badge/GitHub-Issues-red?style=flat-square
    :target: https://github.com/thcasey3/lrengine/issues


===================
Welcome to lrengine
===================
|pypi| |GitHub| |IssuesTracker|

**lrengine** is an open-source Python package for correlating dates and specific language patterns within file and folder names with each other and with outputs of user-supplied functions. 

Features
========
* Organize DataFrames using dates and specific language in file and folder names
* Find specific format date strings, or look for possible date strings, in file or folder names 
* Add classifiers using a custom function that processes contents of a parent directory
* Visualize correlations between function outputs, dates, and language in file and folder names


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
