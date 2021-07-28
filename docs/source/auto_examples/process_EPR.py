# %% [markdown]
"""
EPR data processing
===================

An example user-defined function for processing EPR data with the DNPLab package.

"""
# %%

# %% [markdown]
# For the function below the call would look something like,
"""
lrobject = lr.start(
    parent_directory,
    skip=[".DSC", ".YGF", ".par"],  # otherwise duplicates
    classifiers=["max_loc", "frequency"],
    function=process_EPR.proc_epr,
    function_args={},
)

lrobject.drive()
"""
# parent_directory contains Bruker EPR data. Add patterns, skip, date searching, etc.
# according to the lsframe docs. The function_args are empty in this case. Since DTA
# and spc files come with companion DSC, YGF, or par files and DNPLab uses any of these,
# skip these files to avoid duplicates.
# %%

# %% [markdown]
# Import DNPLab and any other packages that may be needed for your function,
import dnplab as dnp
import numpy as np

# %%

# %% [markdown]
# The function accepts a path to an EPR spectrum file and returns the field value where the spectrum is maximum and the frequency. The function returns zeros where errors are encountered.
def proc_epr(path, args):

    try:
        data = dnp.dnpImport.load(path)
        if len(data.dims) == 1 and "frequency" in data.attrs.keys():
            return [
                np.argmax(data.values, axis=0) / len(data.values),
                data.attrs["frequency"],
            ]
        else:
            return [0, 0]
    except:
        return [0, 0]


# %%
