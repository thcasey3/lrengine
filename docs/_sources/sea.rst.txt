===============
seaborn relplot
===============

With the **sea()** method a **relplot** is made using the **lrobject.frame**. Use the same argument names used in **seaborn.relplot()** as dictionary keys with their corresponding values being any allowed values according to the **seaborn.relplot()** docs, assign this dictionary to the keyword arg **seaborn_args=**,

.. figure:: _static/images/sea_df.png
    :width: 700
    :alt: lrengine concept
    :align: center

    **lrobject.frame**

.. code-block:: python

    lrobject.sea(seaborn_args={'x': 'output1',
                               'y': 'output2',
                               'hue': 'date_delta',
                               's': 100})

.. figure:: _static/images/sea_scatter.png
    :width: 500
    :alt: lrengine concept
    :align: center

    Scatterplot from seaborn.relplot()