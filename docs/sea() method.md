===================
seaborn scatterplot
===================

With the **sea()** method a scatterplot is made using the lrobject.frame. Use the same argument names used in **seaborn.scatterplot()** as dictionary keys with their corresponding values being any allowed values according to the **seaborn.scatterplot()** docs,

.. figure:: _static/images/sea_df.png
    :width: 700
    :alt: lrengine concept
    :align: center

    **lrobject.frame**

.. code-block:: python

    lrobject.sea(options={"x": "output1",
                          "y": "output2",
                          "hue": "date_delta",
                          "s": 100})

.. figure:: _static/images/sea_scatter.png
    :width: 500
    :alt: lrengine concept
    :align: center

    Scatterplot from seaborn