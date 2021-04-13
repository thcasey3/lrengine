=================
Defining Patterns
=================

You may define patterns to classify by. 

Using str or list
=================

If a single pattern or list of patterns is given, the columns will be named according to the patterns and a bool will be supplied indicating the pattern was or was not found. This example adds the column 'sample1' and puts **True** where found, **False** where not found,

.. code-block:: python

    import lrengine as lr

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns='sample1') # single pattern
    lrobject = lr.start(path, patterns=['sample1', 'sample2', 'sample3']) # OR list of patterns


Using dict
==========

You may also use regular expressions to do more sophisticated pattern searches and classify by specific language rather than bool. To do this, use a dictionary rather than a list. The keys of the dict are the column names, the values are the expression to search for and classify by. This example creates a column called 'sample' and classifies by any one digit number found just to the right of the word 'sample',

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns={'sample': 'sample\d'})


If this patterning is not found, **False** is entered. 

To mix these behaviors, add 'bool' for the dict value and the column for the key is added with True or False added if the pattern is or is not found, respectively. This example makes a column for 'sample' and classifies by any one digit number as above, but also adds a column called 'blank_run' and classifies as **True** or **False**.

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns={'sample': 'sample\d', 'blank_run': bool})


Modify Existing Frame
=====================

You can also modify an existing object to classify by different patterns using the **find_patterns()** method. For example, if patterns were not originally defined, add **patterns** then update the frame,

.. code-block:: python

    lrobject.patterns = {'sample': 'sample\d'}
    lrobject.find_patterns()
