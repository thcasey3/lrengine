import unittest
import numpy as np
import os
import lrengine as lr
from numpy.testing import assert_array_equal


class startTester(unittest.TestCase):
    def setUp(self):
        self.path = "./data/"
        self.subd = os.listdir(self.path)
        self.meas = ["meas1", "meas2"]
        self.patt = ["patt1", "patt2"]
        self.skip = ["skip1", "skip2"]
        self.func = None
        self.farg = None

    def test_start_object(self):

        start_result = lr.start(
            self.path, self.patt, self.skip, self.measures, self.func, self.farg
        )
        assertEqual(self.path, start_result["directory"])
        assertEqual(self.subd, start_result["Names"])
        assertEqual(self.meas, start_result["measures"])
        assertEqual(self.patt, start_result["patterns"])
        assertEqual(self.skip, start_result["skip"])
        assertEqual(self.func, start_result["function"])
        assertEqual(self.farg, start_result["function_args"])


if __name__ == "__main__":
    pass
