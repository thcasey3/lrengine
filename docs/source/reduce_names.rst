========================================
Reduce the frame using specific language
========================================

To filter the names by a given pattern, simply call the **.reduce_names()** method and give the patterns to use to decide which names to drop. This example drops from the frame any element having "blank_sample" in the name, 

.. code-block:: python

    lrobject.reduce_dates(skip="blank_sample")


You may also do the reverse and keep only elements with a given pattern in the name using the **keep=** keyword arg. For example, keep only elements with "blank_sample" in the name, 

.. code-block:: python

    lrobject.reduce_dates(keep="blank_sample")