=========================
Save the DateFrame to csv
=========================

Use the **save()** method to save the **.frame** as a .csv using pandas.to_csv(). Specify the path and filename using **filename=** and set **header=True** or **header=False**. Default is **header=True**, and if a filename is not specified the **.frame** will be saved to the parent directory with a name that is the date followed by "_DataFrame.csv".

.. code-block:: python

    lrobject.save(filename="/path/to/my_file.csv", header=True)