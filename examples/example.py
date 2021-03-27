# %% [markdown]
"""
lrengine example
================

This example demonstrates how lrengine makes a pandas DataFrame using a parent directory with sub-directories, some user-defined patterns, and a user-defined function that operates on the sub-directories. This can be applied with any combination of sub-files or sub-directories with a user-supplied function that operates on them to returns numerical measures. For example,

.. code-block:: python
    
    lr.start(path, ... measures=["measure1", "measure2", etc.], function=func, ...)
    
    def func():
        ...code...
        return [measure1, measure2, etc.]

"""
#%%

# %% [markdown]
# Import lrengine,
import random
# %%

def function_handle(directory, args_dict):

    use_directory = directory
    output1 = random.randint(0, args_dict["par1"])
    output2 = random.randint(args_dict["par1"], args_dict["par2"])

    return [output1, output2]




import lrengine as lr

path = "/Users/thomascasey/lrengine/data/Data/"
lrobject = lr.start(path,
                    patterns=["sample1", "sample2", "sample3"],
                    skip=["DS_Store"],
                    measures=["output1", "output2"],
                    function=function_handle,
                    function_args={"par1": 5,
                                   "par2": 10},
                    date_format="YYYYMMDD")

lrobject.drive()

lrobject.map_directory()

lrobject.sea(options={"x": "output1",
                      "y": "output2",
                      "hue": "date_delta",
                      "s": 100})

print(lrobject.frame.head())
