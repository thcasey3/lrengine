=====================
Save the Frame to csv
=====================

Use the **save()** method to save the **.frame** as a .csv using **pandas.to_csv()**. Specify the path and filename using **filename=** and set **header=True** or **header=False**, default is **header=True**. If a filename is not specified the **.frame** will be saved to the parent directory with a name that is the date followed by **"_DataFrame.csv"**.

.. code-block:: python

    lsobject.save(filename='/path/to/my_file.csv', header=True)

If this **.csv** file is not modified and in a subsequent call to **ls.start(path)** the path is to this **.csv** file, the same frame that was created and saved will be re-created in the new **start** object. However, the other attributes of the **start** object that was saved will be missing, and will need to be defined manually.