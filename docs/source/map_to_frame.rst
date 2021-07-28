===============================
Create Frame from directory_map
===============================

With the **map_directory()** method you get a comprehensive dictionary where the keys are the directories and the values are lists of files within the directory. This can be useful for defining the ["name"] column of the frame. Use the method **map_to_frame()** to create a **frame** in the  start object from the contents of the **directory_map**. For a given **directory_map** that has several layers of directories and files throughout, you may specify the **depth** to consider, whether to use **"files"**, **"folders"**, or **"any"**, and whether to replace the current **frame**. For example, consider a depth of two relative to the parent directory, accept files or folders, and replace the **frame** in the start object, 

.. code-block:: python

    lsobject.map_to_frame(depth=2, kind='any')

Or, consider a depth of three relative to the parent directory, accept only folders, and choose not replace the **frame** in the start object by setting **to_frame=False**, and return a new frame,

.. code-block:: python

    frame = lsobject.map_to_frame(depth=3, kind='folders', to_frame=False)


Multiple Depths
===============

You may also consider multiple depths. For example, consider depths of one or four relative to the parent directory, accept only files, and replace the **frame** in the start object,

.. code-block:: python

    lsobject.map_to_frame(depth=[1, 4], kind='files')


Zero Depth
==========

If you set **depth=0**, this will return only files from the parent directory regardless of the **kind** that is set. To return only the folders that are in the parent directory, set **depth=1** and **kind='folders'**. To return files and folders that are in the parent directory use a list for depth, **depth=[0, 1]**, and set **kind='any'**.


Maximum Depth
=============

The default depth setting is **depth='max'**. This will result in a dictionary that accounts for all folders and files under the parent directory.

