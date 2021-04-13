======================
Call a Custom Function
======================

You can even use a custom function that operates on each element of the parent directory to add the outputs as classifiers. Do this my adding the names of the classifier columns, defining the function call, and adding any needed arguments in the form of a dictionary. For example, if the function is:

.. code-block:: python

    def function_handle(directory, args_dict):

        use_directory = directory
        output1 = random.randint(0, args_dict['par1'])
        output2 = random.randint(args_dict['par1'], args_dict['par2'])

        return [output1, output2]

Create the object,

.. code-block:: python

    import lrengine as lr

    lrobject = lr.start(path,
                        patterns={'sample': '\d\d'}, 
                        skip='sample7', 
                        date_format='any'
                        classifiers=['output1', 'output2'],
                        function=function_handle,
                        function_args={'par1': 1,
                                       'par2': 2}
                        )

Call the **drive()** method

.. code-block:: python

    lrobject.drive()

and two new columns would be added called "output1" and "output2" with the values corresponding to the function outputs. Make sure to have the function accept a path and a single dictionary that contains any additional parameters needed. Also make sure the function returns the outputs in a list that is equal in length to the given list of classifiers. Use the above example function as a template.


Handling Errors
===============

If the custom function errors, for example if it tries to operate on a file/folder in the path that is not compatible, it will return the string "null" for the classifier. This can be useful for avoiding tedious reorganizing of directories. Simply run **lrobject.drive()**, collect a frame full of successful runs or "null", then use something similar to,

.. code-block:: python

    lrobject.frame = lrobject.frame[lrobject.frame['output1'] != 'null']

to reduce the frame to only compatible files/folders.
