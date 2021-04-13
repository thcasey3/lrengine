===============================
Create Frame from directory_map
===============================

With the **map_directory()** method you get a comprehensive dictionary where the keys are the directories and the values are lists of files within the directory. This can be useful for defining the ["name"] column of the frame. Use the method **map_to_frame()** to create a **frame** in the  start object from the contents of the **directory_map**. For a given **directory_map** that has several layers of directories and files throughout, you may specify the **depth** to consider, whether to use **"files"**, **"folders"**, or **"any"**, and whether to replace the current **frame**. For example, consider a depth of two relative to the parent directory, accept files or folders, and replace the **frame** in the start object, 

.. code-block:: python

    lrobject.map_to_frame(depth=2, kind='any')

Or, consider a depth of three relative to the parent directory, accept only folders, and choose not replace the **frame** in the start object by setting **to_frame=False**, and return a new frame,

.. code-block:: python

    frame = lrobject.map_to_frame(depth=3, kind='folders', to_frame=False)

You may also consider multiple depths. For example, consider depths of one or four relative to the parent directory, accept only files, and replace the **frame** in the start object,

.. code-block:: python

    lrobject.map_to_frame(depth=[1, 4], kind='files')

