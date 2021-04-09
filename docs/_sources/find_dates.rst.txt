=============
Finding Dates
=============

You may also classify by dates found in the file or folder names and the days elapsed since the found date. This example looks for dates of the format 'YYYYMMDD' and adds 'date' and 'date_delta' columns,

.. code-block:: python

    import lrengine as lr

    path = '/path/to/directory/'
    lrobject = lr.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='YYYYMMDD')


You can search for any of a list of formats if you supply a list, or even look for all possible dates by setting **date_format='any'**. This finds all logical dates and gives them as a list, along with a date_delta list and a date_format list. For example,

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='any')


Or just look for any of three formats,

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format=['YYYYMMDD', 'YYMMDD', 'MMDDYYYY'])


You can also modify an existing object to include dates or change to a different format using the **find_dates()** method. For example, change the **date_format** then update the dates information,

.. code-block:: python

    lrobject.date_format = 'YYYY-MM-DD'
    lrobject.find_dates()

You may use lists here as well,

.. code-block:: python

    lrobject.date_format = ['YYYY-MM-DD', 'YY-MM-DD']
    lrobject.find_dates()


The currently supported date_format options are,

* All possible combinations of four or two digit years, two or one digit months, and two or one digit days; with or without '-', '_', '/', ':', or ';' included. For example, any of these can be found: **'YYYYMMDD'**, **'YYMD'**, **'DD-MM-YY'**, **'DMYY'**, **'DD:MM:YYYY'**, **'D/M/YYYY'**, etc.
* **'any'** will try all of the formats described above and give a list of all that are found if more than one