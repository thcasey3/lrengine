"""
class for calling function and checking outputs
"""

import os
import numpy as np
from . import tools


class cylinders:
    """
    cylinders class

    Attributes:
        lrdata (start object): Data object from start class
    """

    def __init__(self, lrdata):

        if not isinstance(lrdata.function_args, dict):
            raise TypeError("function_args must be a dictionary")

        if (lrdata.function is not None) and (lrdata.classifiers is not None):
            classifiers = self.run_function(lrdata)
            classifiers = self.check_func_output(lrdata.classifiers, classifiers)

            for indx1, meas_col in enumerate(lrdata.classifiers):
                lrdata.frame[meas_col] = np.zeros(len(lrdata.frame))
                for indx2, meas_val in enumerate(classifiers):
                    lrdata.frame.loc[indx2, meas_col] = meas_val[indx1]
        else:
            print(
                "DataFrame will only classify by Names, dates if any exist in the Names, and patterns if any are given"
            )

    def run_function(self, lrdata):

        classifiers = list(np.zeros(len(lrdata.frame.names)))
        for indx, name in enumerate(lrdata.frame.names):
            classifiers[indx] = lrdata.function(
                os.path.join(lrdata.directory, name), lrdata.function_args
            )

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
