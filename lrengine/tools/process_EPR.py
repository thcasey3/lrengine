import sys

sys.path.append("/Users/thomascasey/dnplab")
import dnplab as dnp
import numpy as np


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
