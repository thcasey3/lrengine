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
path = "../path to parent directory"
# %%

# %% [markdown]
# Define a function that accepts the parent directory and any additional arguments in the form of a dictionary and
# returns a list of outputs. The list must have a single value for each member of the list and have a **'len()'**
# equal to the list of **'classifiers'** given during the instantiation of the object below.
def function_handle(directory, args_dict):

    # insert code that acts on each file or sub-directory of directory and makes two floats or ints
    # output1 =
    # output2 =

    return [output1, output2]


# %%

# %% [markdown]
# Instantiate the **'lrobject'**,
lrobject = lr.start(
    path,
    patterns=[],  # enter any patterns in the file or folder names to use as classifiers, e.g. ["sample1", "sample2"]
    skip=[],  # enter any sub-strings in the names of file folders that should be skipped, e.g. ["blank_run"]
    classifiers=[
        "output1",
        "output2",
    ],  # these are the column names corresponding to the function outputs
    function=function_handle,  # complete the function above
    function_args={},  # enter any arguments required by function_handle, e.g. {"skiprows": 0}
    date_format="YYYYMMDD",  # (optional) give the format of any date strings in the file or folder names
)
# %%

# %% [markdown]
# Pass the **'lrobject'** to the user defined function to add **'frame'** to the **'lrobject'**,
# then print the **'head()'** of **'frame'**,
lrobject.drive()
print(lrobject.frame.head())
# %%

# %% [markdown]
# Add to **'lrobject'** a dictionary that is a map of the parent directory,
lrobject.map_directory()
# %%

# %% [markdown]
# Create a seaborn.scatterplot correlating the two outputs. Replace **'None'** for hue with date_delta if it exists in
# your **'frame'**, or maybe with a third output if you have more than two. The **'seaborn_args'** dictionary should have
# keys that are the arguments that would be given to **seaborn.scatterplot** and any allowed values according to seaborn
# docs,
lrobject.sea(seaborn_args={"x": "output1", "y": "output2", "hue": None, "s": 100})
# %%
