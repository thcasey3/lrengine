"""
class for the user defined functions
"""

# import dnplab as dnp
import numpy as np


class tools:
    def __init__(self, input_dict):

        self.col_name = "ksig"
        self.measure = self.calc_odnp(input_dict)

    def calc_odnp(self, input_dict):

        print(" INSIDE calc_odnp FUNCTION")

        return np.linspace(95.5, 95.5, len(input_dict["frame"]))
