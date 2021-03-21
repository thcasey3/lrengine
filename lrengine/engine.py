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
        start object (start object): Data object from start class
    """

    def __init__(self, lrdata):

        if (lrdata["function"] is not None) and (lrdata["measures"] is not None):
            measures = self.run_function(lrdata)
            measures = self.check_func_output(lrdata["measures"], measures)

            for indx1, meas_col in enumerate(lrdata["measures"]):
                lrdata["frame"][meas_col] = np.zeros(len(lrdata["frame"]))
                for indx2, meas_val in enumerate(measures):
                    lrdata["frame"].loc[indx2, meas_col] = meas_val[indx1]
        else:
            print(
                "DataFrame will only classify by Names, dates if any exist in the Names, and patterns if any are given"
            )

    def run_function(self, lrdata):

        measures = list(np.zeros(len(lrdata["frame"]["Names"])))
        for indx, name in enumerate(lrdata["frame"]["Names"]):
            measures[indx] = lrdata["function"](
                os.path.join(lrdata["directory"], lrdata["frame"]["Names"][indx]),
                lrdata["function_args"],
            )

        return measures

    @staticmethod
    def check_func_output(lrdata_measures, measures):

        if not isinstance(measures, list):
            if len(measures) == 1:
                measures = list(measures)
            else:
                raise TypeError("the measures are not in a list")
        elif isinstance(measures, list) and (
            isinstance(any(measures), list) and not isinstance(all(measures), list)
        ):
            raise TypeError(
                "measures is a mixture of lists and non-lists, this is not allowed"
            )
        for indx, _ in enumerate(measures):
            if len(measures[indx]) != len(lrdata_measures):
                raise TypeError(
                    "len of measures returned != len of measures given, this is not allowed"
                )

        return measures
