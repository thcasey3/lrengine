================
Narrow the Dates
================

You may also reduce the frame to a specific date or range of dates using the **on_date()** or **in_range()** methods. For example, to keep only the elements of a frame having the date Aug. 2nd 1985, 

.. code-block:: python

    lrobject.on_date(keep="1985-08-02")


To do the exact opposite and remove specific dates, use the keyword **remove=** instead,

.. code-block:: python

    lrobject.on_date(remove="1985-08-02")


Use Lists
=========

You may also give a list. For example, keep any element with either the date above or Feb. 8th 1988,

.. code-block:: python

    lrobject.on_date(keep=["1985-08-02", "1988-02-08"])

Or to do the opposite,

.. code-block:: python

    lrobject.on_date(remove=["1985-08-02", "1988-02-08"])


Use Ranges
==========

To specify a range or ranges rather than specific dates, use **in_range()** instead. Ranges must be two element lists. For example, to keep all elements having dates between Aug. 1st and Sept. 1st 1985, 

.. code-block:: python

    lrobject.in_range(keep=["1985-08-01", "1985-09-01"])


As with **on_date()**, you may also give a list of ranges (list of lists). For example, keep only elements either in the range above or in the month of Dec. 1985,

.. code-block:: python

    lrobject.in_range(keep=[["1985-08-01", "1985-09-01"], ["1985-12-01", "1985-12-31"]])

Again, you may also do the exact opposite with the keyword arg **remove=**, 

.. code-block:: python

    lrobject.in_range(remove=["1985-08-01", "1985-09-01"])


Or,

.. code-block:: python

    lrobject.in_range(remove=[["1985-08-01", "1985-09-01"], ["1985-12-01", "1985-12-31"]])


Remove Zeros
============

There is also an option to remove or keep any elements with 0 for date using the keyword arg **strip_zeros=**. Default is **True** for **on_date()** and **in_range()**.  For example,

.. code-block:: python

    lrobject.on_date(keep="1985-08-02")                    #removes where date=0
    lrobject.on_date(keep="1985-08-02", strip_zeros=False) #keeps where date=0

And,

.. code-block:: python

    lrobject.in_range(keep=["1985-08-01", "1985-09-01"])   #removes where date=0
    lrobject.in_range(keep=["1985-08-01", "1985-09-01"], 
                      strip_zeros=False)                   #keeps where date=0