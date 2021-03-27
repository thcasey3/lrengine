.. install:

=================
Quick-Start Guide
=================

Importing the Package
=====================

.. code-block:: python

   import lrengine as lr


Defining Directory
===================
.. code-block:: python

   # Create 'start' object using a parent directory, look for dates in file or folder names
   path = 'path/to/parent_directory'
   lrobject = lr.start(path, date_format="YYYYMMDD")
   # Optional, create dictionary that is a map of the parent directory
   lrobject.map_directory()


Running Functions
=================
.. code-block:: python

   # Perform element wise operation of a function on files or sub-directories
   lrobject = lr.start(path, function=function_handle, function_args=dictionary)
