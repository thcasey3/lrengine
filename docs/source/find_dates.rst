=============
Finding Dates
=============

You may also classify by dates found in the file or folder names and the days elapsed since the found date. This example looks for dates of the format "YYYYMMDD" and adds "date" and "date_delta" columns,

.. code-block:: python

    path = "/path/to/directory/"
    lrobject = lr.start(path, patterns={"sample": "\d\d"}, skip="sample7", date_format="YYYYMMDD")


You can search for all possible dates by setting **date_format="any"**. This finds all logical dates and gives them as a list, along with a date_delta list and a date_format list. For example,

.. code-block:: python

    path = "/path/to/directory/"
    lrobject = lr.start(path, patterns={"sample": "\d\d"}, skip="sample7", date_format="any")


The currently supported date_format options are,

* "any" (try all of the formats below)
* "YYYYMMDD"
* "YYYYDDMM"
* "MMDDYYYY"
* "DDMMYYYY"
* "YYMMDD"
* "YYDDMM"
* "MMDDYY"
* "DDMMYY"
* "YYYY-MM-DD" ("-" also implies "_", "/", ":", and ";")
* "YYYY-DD-MM"
* "MM-DD-YYYY"
* "DD-MM-YYYY"
* "YY-MM-DD"
* "YY-DD-MM"
* "MM-DD-YY"
* "DD-MM-YY"