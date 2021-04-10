# %% [markdown]
"""
EPR data processing
===================

An example user-defined function for processing EPR data with the DNPLab package.

"""
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
