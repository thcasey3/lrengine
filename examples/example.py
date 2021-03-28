# %% [markdown]
"""
lrengine example
================

This example demonstrates creating and manipulating a DataFrame from a directory. The data loaded from 'example.csv' is
an example of a frame that could be created with lrengine. Since we are using pre-defined data and not an actual
directory, this example also demonstrates how to load and use an existing DataFrame that has been saved in csv format.

"""
# %%

# %% [markdown]
# Import lrengine and any other packages that may be needed,
import lrengine as lr

# %%

# %% [markdown]
# Next, set the path to a parent directory. For this example we will load "example.csv" and take the list of
# sub-directories from the column called "names".
path = "./data/example.csv"
# %%

# %% [markdown]
# Define a function that accepts a parent directory and any additional arguments in the form of a dictionary,
# and returns a list of outputs. For this example, the function will simply take from the loaded data but in a real
# application of lrengine the function should operate on the individual files or sub-directories of the parent
# directory and return "measures" to be correlated with certain language or dates found in the names.


def function_handle(directory, args_dict):

    data = pd.read_csv(directory)
    output1 = data[args_dict["par1"]]
    output2 = data[args_dict["par2"]]

    return [output1, output2]


# %%

# %% [markdown]
# Create the lrobject with the path to the directory and,
# 1. patterns = a list of patterns in file or sub-directory names to be used as classifiers.
# 2. skip = a list of patterns in the file or sub-directory names to be used to decide which to skip.
# 3. measures = a list of measures that will be classifiers. This corresponds to the function outputs.
# 4. function = handle to a function that returns a list of outputs corresponding to the defined measures.
# 5. function_args = dictionary of arguments for the function.
# 6. date_format = format of date pattern that may be in the file or sub-directory names

lrobject = lr.start(
    path,
    patterns=["sample1", "sample2", "sample3", "sample4", "sample5"],
    skip=["DS_Store"],
    measures=["output1", "output2"],
    function=function_handle,
    function_args={"par1": "measure1", "par2": "measure2"},
    date_format="YYYYMMDD",
)

# %%

lrobject.drive()

lrobject.map_directory()

lrobject.sea(options={"x": "output1", "y": "output2", "hue": "date_delta", "s": 100})

print(lrobject.frame.head())
