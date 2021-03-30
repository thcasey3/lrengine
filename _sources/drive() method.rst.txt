============================
Call a user defined function
============================

Use the **drive()** method to apply the user-defined function to each file or sub-directory of the parent directory. First construct the function,

.. code-block:: python

    def function_handle(directory, args_dict):

        use_directory = directory
        output1 = random.randint(0, args_dict["par1"])
        output2 = random.randint(args_dict["par1"], args_dict["par2"])

        return [output1, output2]

Add some additional arguments to the start object,

.. code-block:: python

    lrobject = lr.start(path,
                        patterns=["sample1", "sample2", "sample3"],
                        classifiers=["output1", "output2"],
                        function=function_handle,
                        function_args={"par1": 1,
                                       "par2": 2}
                        )

Use the **drive()** method to apply the function to each file or sub-directory of the parent directory,

.. code-block:: python

    lrobject.drive()

The **start** **object** now contains a **.frame** that is a Pandas DataFrame of classifiers pulled from the file or sub-directory names using **patterns=** and also those returned from the user-defined function,

.. figure:: _static/images/df_head.png
    :width: 500
    :align: center

    Head of **.frame** created by lrengine


