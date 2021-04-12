"""
engine module, for calling functions and checking outputs
"""

import os
import numpy as np


class cylinders:
    """
    class for interacting with a user-supplied functions

    Args:
        lrdata (start): start object

    Returns:
        updated start object
    """

    def __init__(self, lrdata):

        if not isinstance(lrdata.function_args, dict):
            raise TypeError("function_args must be a dictionary")

        if (lrdata.function is not None) and (lrdata.classifiers is not None):
            classifiers = self.run_function(lrdata)
            classifiers = self.check_func_output(lrdata.classifiers, classifiers)

            for indx1, meas_col in enumerate(lrdata.classifiers):
                lrdata.frame[meas_col] = ""
                for indx2, meas_val in enumerate(classifiers):
                    lrdata.frame.loc[lrdata.frame.index[indx2], meas_col] = meas_val[
                        indx1
                    ]
        else:
            print(
                "DataFrame will only classify by Names, dates if any exist in the Names, and patterns if any are given"
            )

    def run_function(self, lrdata):

        classifiers = []
        for name in lrdata.frame.name:
            try:
                classifiers.append(
                    lrdata.function(
                        os.path.join(lrdata.directory, name), lrdata.function_args
                    )
                )
            except:
                classifiers.append(["null" for _ in range(len(lrdata.classifiers))])

        return classifiers

    @staticmethod
    def check_func_output(lrdata_measures, classifiers):

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
            if len(classifiers[indx]) != len(lrdata_measures):
                raise ValueError(
                    "len of classifiers returned != len of classifiers given, this is not allowed"
                )

        return classifiers
