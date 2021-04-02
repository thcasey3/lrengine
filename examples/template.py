# sphinx_gallery_thumbnail_path = '_static/logo-white.png'
# %% [markdown]
"""
lrengine Template
=================

This Template serves as an example layout for a script using lrengine.

"""
# %%

# %% [markdown]
# Import lrengine and any other packages that may be needed for your function,
import lrengine as lr

# import numpy as np
# etc.
# %%

# %% [markdown]
# Set the path to a parent directory,
path = "../path/to/parent/directory"
# %%

# %% [markdown]
# Define a function that accepts the parent directory and any additional arguments in the form of a dictionary and
# returns a list of outputs. The list must have a single value for each member of the list and have **len()**
# equal to the list of **'classifiers'** given during the creation of the **start** object below.
def function_handle(directory, args_dict):

    # insert code that acts on each file or sub-directory of directory and makes output(s)
    # output1 =
    # output2 =

    return [output1, output2]


# %%

# %% [markdown]
# Create the **start** object,
lrobject = lr.start(
    path,
    patterns=[],  # enter any patterns in the file or folder names to use as classifiers
    skip=[],  # enter any sub-strings in the names of file folders that should be skipped
    classifiers=[
        "output1",
        "output2",
    ],  # these are the column names corresponding to the function outputs
    function=function_handle,  # complete the function above
    function_args={},  # enter any arguments required by function_handle
    date_format="YYYYMMDD",  # give the format of any date strings in the file or folder names
)
# %%

# %% [markdown]
# Pass the **start** object to the user defined function to add classifiers to the **.frame**,
# then print the **.head()** of **.frame**,
lrobject.drive()
print(lrobject.frame.head())
# %%

# %% [markdown]
# Add a dictionary to the **start** object that is a map of the parent directory,
lrobject.map_directory()
# %%

# %% [markdown]
# Create a seaborn.relplot correlating the two outputs. Replace **'None'** for hue with date_delta if it exists in
# your **.frame**, or maybe with a third output if you have more than two. The **'seaborn_args'** dictionary should have
# keys that are the arguments that would be given to **seaborn.relplot** and any allowed values according to **seaborn.relplot**
# documentation,
lrobject.sea(seaborn_args={"x": "output1", "y": "output2", "hue": None, "s": 100})
# %%
