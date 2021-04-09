=====================
Create a start object
=====================

Create an object that contains a DataFrame with at minimum one column that is the names of the files or folders in the supplied directory,

.. code-block:: python
    
    import lrengine as lr

    path = '/path/to/directory/'
    lrobject = lr.start(path)


You may define patterns to classify by. If a single pattern or list of patterns is given, the columns will be named according to the patterns and a bool will be supplied indicating the pattern was or was not found. This example adds the column 'sample1' and puts **True** where found, **False** where not found,

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns='sample1') # single pattern
    lrobject = lr.start(path, patterns=['sample1', 'sample2', 'sample3']) # OR list of patterns

You may also use regular expressions to do more sophisticated pattern searches and classify by specific language rather than bool. To do this, use a dictionary rather than a list. The keys of the dict are the column names, the values are the expression to search for and classify by. This example creates a column called 'sample' and classifies by any one digit number found just to the right of the word 'sample',

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns={'sample': 'sample\d'})


If this patterning is not found, **False** is entered. 

To mix these behaviors, add **bool** for the dict value and the column for the key is added with **True** or **False** added if the pattern is or is not found, respectively. This example makes a column for 'sample' and classifies by any one digit number as above, but also adds a column called 'blank_run' and classifies as **True** or **False**.

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns={'sample': 'sample\d', 'blank_run': bool})


You may skip directories according to specific language. This example classifies by sample number but skips any file or folder with 'sample7' in the name,

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, patterns={'sample': 'sample\d'}, skip='sample7')


You may also classify by dates found in the file or folder names and the days elapsed since the found date. This example looks for dates of the format 'YYYYMMDD' and adds 'date' and 'date_delta' columns,

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='YYYYMMDD'
                        )


You can search for all possible dates by setting **date_format='any'**. This finds all logical dates and gives them as a list, along with a date_delta list and a date_format list. For example,

.. code-block:: python

    path = '/path/to/directory/'
    lrobject = lr.start(path, 
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='any'
                        )


You can even use a custom function that operates on each element of the parent directory to add the outputs as classifiers. Do this my adding the names of the classifier columns, defining the function call, and adding any needed arguments in the form of a dictionary. For example, if the function is:

.. code-block:: python

    def function_handle(directory, args_dict):

        use_directory = directory
        output1 = random.randint(0, args_dict['par1'])
        output2 = random.randint(args_dict['par1'], args_dict['par2'])

        return [output1, output2]

the call would look like,

.. code-block:: python

    lrobject = lr.start(path,
                        patterns={'sample': 'sample\d'}, 
                        skip='sample7', 
                        date_format='any'
                        classifiers=['output1', 'output2'],
                        function=function_handle,
                        function_args={'par1': 1,
                                       'par2': 2}
                        )

and two new columns would be added called 'output1' and 'output2' with the values corresponding to the function outputs. Make sure to have the function accept a path and a single dictionary that contains any additional parameters needed. Also make sure the function returns the outputs in a list that is equal in length to the given list of classifiers. Use the above example function as a template.
