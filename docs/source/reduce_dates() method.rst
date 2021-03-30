===================================
Reduce the date lists to one format
===================================

The **find_dates()** method with the option **date_format="any"** may return lists for each name if more than one date are found. For example, the date 2000-01-01 will be found to be a date as is and also as 00-01-01. To reduce the **"dates"** to a single date format, use the **reduce_dates()** method with the desired format as an argument, 

.. figure:: _static/images/listed_dates.png
    :width: 600
    :align: center

    **.head()** with found dates added

.. code-block:: python

    lrobject.reduce_dates(format="YYYYMMDD")

.. figure:: _static/images/reduced_dates.png
    :width: 600
    :align: center

    **.head()** after reduction