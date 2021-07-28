"""
engine module, for calling functions and checking outputs
"""

import os
import numpy as np


class cylinders:
    """
    class for interacting with a user-supplied functions

    Args:
        lsdata (start): start object

    Returns:
        updated start object
    """

    def __init__(self, lsdata):

        if (
            not isinstance(lsdata.function_args, dict)
            and lsdata.function_args is not None
        ):
            raise TypeError("function_args must be a dictionary")

        if (lsdata.function is not None) and (lsdata.classifiers is not None):
            classifiers = self.run_function(lsdata)
            classifiers = self.check_func_output(lsdata.classifiers, classifiers)

            for indx1, meas_col in enumerate(lsdata.classifiers):
                lsdata.frame[meas_col] = ""
                for indx2, meas_val in enumerate(classifiers):
                    lsdata.frame.loc[lsdata.frame.index[indx2], meas_col] = meas_val[
                        indx1
                    ]
        else:
            print(
                "DataFrame will only classify by Names, dates if any exist in the Names, and patterns if any are given"
            )

    def run_function(self, lsdata):

        classifiers = []
        for name in lsdata.frame.name:
            try:
                classifiers.append(
                    lsdata.function(
                        os.path.join(lsdata.directory, name), lsdata.function_args
                    )
                )
            except:
                classifiers.append(["null" for _ in range(len(lsdata.classifiers))])

        return classifiers

    @staticmethod
    def check_func_output(lsdata_measures, classifiers):

        if not isinstance(classifiers, list):
            if len(classifiers) == 1:
                classifiers = list(classifiers)
            else:
                raise TypeError("the classifiers are not in a list")

        elif isinstance(classifiers, list) and (
            isinstance(any(classifiers), list)
            and not isinstance(all(classifiers), list)
        ):
            raise TypeError(
                "classifiers is a mixture of lists and non-lists, this is not allowed"
            )

        for indx, _ in enumerate(classifiers):
            if len(classifiers[indx]) != len(lsdata_measures):
                raise ValueError(
                    "len of classifiers returned != len of classifiers given, this is not allowed"
                )

        return classifiers
