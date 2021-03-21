# %% [markdown]
"""
lrengine example with pre-defined tools function and sample data
======================

This example demonstrates how lrengine makes a pandas DataFrame using a parent directory with sub-directories, some user-defined patterns, and a user-defined function that operates on the sub-directories. This can be applied with any combination of sub-files or sub-directories with a user-supplied function that operates on them to returns numerical measures. For example,

.. code-block:: python
    
    lr.start(path, ... measures=["measure1", "measure2", etc.], function=func, ...)
    
    def func():
        ...code...
        return [measure1, measure2, etc.]

"""
# %% [markdown]
# Import lrengine,
import lrengine as lr

# %%

# %% [markdown]
# Set the path to the example data folder called 'nmr_odnp_data',
#
path = "/data/nmr_odnp_data/"
# %%

# %% [markdown]
# For this example the dnplab package will be used. The dnpHydration module takes a dictionary of constants/options shown below in the variable 'hydration' and will also be given some additional inputs supplied by the tools function from han_lab called 'calc_odnp',
hydration = {
    "T10": 1.5,
    "T100": 2.0,
    "spin_C": 150.0,
    "field": 348.5,
    "smax_model": "tethered",
    "t1_interp_method": "second_order",
}
# %%

# %% [markdown]
# In the call to lrengine.start the user gives: 1) the path to the parent directory, 2) a list of patterns to set as classifier columns in the pandas DataFrame, 3) a list of patterns to use as criteria for skipping a file or sub-directory, 4) a list of names of measurables corresponding to the function outputs, 5) the function that operates on the files or sub-directories, 6) any function arguments. For demonstration purposes this example pulls data from a .csv instead of performing the full calculations with dnplab. This example would skip any sub-directorys with the skip items in the sub-directory names, classify by the items in patterns, and add columns for the measures given,
data = lr.start(
    path,
    patterns=[
        "samp1",
        "samp2",
        "samp3",
        "samp4",
        "samp5",
        "samp6",
        "00sec",
        "15sec",
        "30sec",
    ],
    skip=["DS_Store", "blank"],
    measures=["ksigma", "tcorr"],
    function=lr.tools.han_lab.calc_odnp,
    function_args=hydration,
)
# %%

# %% [markdown]
# Show the head() of the pandas DataFrame. Notice the patterns that were searched for are given as columns, with a 1 or 0 indicting the presence or absence of the pattern in the file or sub-directory name. Also notice the columns called 'ksigma' and 'tcorr' that are the outputs of the supplied function. The columns for date and date_delta are given if a 8 number string such as '19850802' is found in the file or sub-directory name. This example would be interpreted as Aug. 2nd 1985 and put in as 1985-08-02, with date_delta being the number of days elapsed since Aug 2nd 1985.
print(data.frame.head())
# %%
