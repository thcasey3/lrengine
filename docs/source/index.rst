.. lrengine documentation master file, created by
   sphinx-quickstart on Fri Jan 10 17:26:18 2020.

.. .. image:: logo.png
..     :target: https://github.com/thcasey3/lrengine


.. |GitHub| raw:: html

   <a href="https://github.com/thcasey3/lrengine/" target="_blank"> GitHub</a>

.. |IssuesTracker| raw:: html

   <a href="https://github.com/thcasey3/lrengine/issues" target="_blank"> Report Issues</a>


===================
Welcome to lrengine
===================

lrengine is an open-source Python package for correlating dates and specific language patterns within file and folder names either with each other or with outputs of a user supplied function. 
    
The source code for the project is published here: |GitHub|

Please report all issues using the: |IssuesTracker|

.. list-table::
   :widths: 60 40

   * - Current Release
     - |release|
   * - Documentation Build Date
     - |date|
   * - Author(s)
     - |author|


Features
========
* Organize DataFrames using dates and specific language in file and folder names
* Find specific format date strings, or look for possible date strings, in file or folder names 
* Construct DataFrames from a function that processes files or sub-directories of a parent directory
* Visualize correlations between function outputs, and dates or specific language in file and folder names

.. toctree::
   :maxdepth: 2

   install
   introduction
   create start object
   drive() method
   map_directory() method
   find_dates() method
   reduce_dates() method
   save() method
   sea() method
   auto_examples/index
   Planned features

