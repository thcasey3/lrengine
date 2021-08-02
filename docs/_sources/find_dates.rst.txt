=============
Finding Dates
=============

You may also classify by dates found in the file or folder names and the days elapsed since the found date.

Imagine the folders in a directory contain dates in their names,

.. figure:: _static/images/dir_dates.png
    :width: 350
    :align: center

    Example parent directory


In this case add a **date_format** to search for. If the date format is known and consistent, you may set the format to ensure the correct date is the only date chosen from the folder name.

.. code-block:: python

    import lsframe as ls

    path = '/path/to/directory/'
    lsobject = ls.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='YYYYMMDD')


If the format is not known, or there is more than one format, you can search for any of a list of formats. For this behavior, supply a list of formats or look for all possible dates by setting **date_format='any'**. By using **'any'** all logical dates are listed, along with a date_delta list and a date_format list. For example,

.. code-block:: python

    path = '/path/to/directory/'
    lsobject = ls.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='any')


To just look for any of three formats,

.. code-block:: python

    path = '/path/to/directory/'
    lsobject = ls.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format=['YYYYMMDD', 'YYMMDD', 'MMDDYYYY'])


Find Dates Method
=================

You can also modify an existing object to include dates or change to a different format using the **find_dates()** method. For example, change the **date_format** then update the dates information,

.. code-block:: python

    lsobject.date_format = 'YYYY-MM-DD'
    lsobject.find_dates()

You may use lists here as well,

.. code-block:: python

    lsobject.date_format = ['YYYY-MM-DD', 'YY-MM-DD']
    lsobject.find_dates()


Allowed Formats
===============

The currently supported date_format options are,

* All possible combinations of four or two digit years, two or one digit months, and two or one digit days; with or without '-', '_', '/', ':', or ';' included. For example, any of these can be found: **'YYYYMMDD'**, **'YYMD'**, **'DD-MM-YY'**, **'DMYY'**, **'DD:MM:YYYY'**, **'D/M/YYYY'**, etc.
* **'any'** will try all of the formats described above and give a list of all that are found if more than one