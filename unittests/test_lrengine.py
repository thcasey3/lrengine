import unittest
import numpy as np
import os
from datetime import date
from lrengine import start, intake, engine, tools
from numpy.testing import assert_array_equal


class startTester(unittest.TestCase):
    def setUp(self):
        self.path = "./data/"
        self.subd = os.listdir(self.path)
        self.csv_file = os.path.join(self.path, "example_short.csv/")
        self.meas = ["meas1", "meas2"]
        self.patt = ["patt1", "patt2"]
        self.skip = ["skip1", "skip2"]
        self.func = tools.utest.utest
        self.farg = {"key": "value"}
        self.date_format = "YYYYMMDD"

        self.start_result = start(
            self.path,
            patterns=self.patt,
            skip=self.skip,
            measures=self.meas,
            function=self.func,
            function_args=self.farg,
            date_format=self.date_format,
        )

    def test_a_start_object(self):

        self.assertEqual(self.path, self.start_result.directory)
        self.assertEqual(self.subd, os.listdir(self.start_result.directory))
        self.assertEqual(self.meas, self.start_result.measures)
        self.assertEqual(self.patt, self.start_result.patterns)
        self.assertEqual(self.skip, self.start_result.skip)
        self.assertEqual(self.func, self.start_result.function)
        self.assertEqual(self.farg, self.start_result.function_args)
        self.assertEqual(self.date_format, self.start_result.date_format)

        with self.assertRaises(TypeError):
            result = start(
                self.path,
                measures=("one", "two"),
                function=self.func,
                function_args=self.farg,
            )

        with self.assertRaises(TypeError):
            result = start(
                self.path,
                patterns=("one", "two"),
                function=self.func,
                function_args=self.farg,
            )

        with self.assertRaises(TypeError):
            result = start(
                self.path,
                skip=("one", "two"),
                function=self.func,
                function_args=self.farg,
            )

        with self.assertRaises(TypeError):
            result = start(directory=["one", "two"])

        with self.assertRaises(ValueError):
            result = start(self.path, date_format="YearMonthDay")

        with self.assertRaises(TypeError):
            self.start_result.save(filename=100)

        if "utest.csv" in os.listdir(self.path):
            os.remove(self.path + "utest.csv")

        self.start_result.save(filename="utest")
        self.assertTrue("utest.csv" in os.listdir(self.path))

        os.remove(self.path + "utest.csv")

        self.start_result.save(filename="utest.csv")
        self.assertTrue("utest.csv" in os.listdir(self.path))
        self.assertFalse("utest.csv.csv" in os.listdir(self.path))

        os.remove(self.path + "utest.csv")
        if "utest.csv.csv" in os.listdir(self.path):
            os.remove(self.path + "utest.csv.csv")

    def test_b_drive_method(self):

        self.start_result.drive()
        self.assertEqual(len(self.start_result.frame.columns), 7)
        self.assertEqual(self.start_result.frame.columns[0], "names")
        self.assertEqual(self.start_result.frame.columns[1], "patt1")
        self.assertEqual(self.start_result.frame.columns[2], "patt2")
        self.assertEqual(self.start_result.frame.columns[3], "date")
        self.assertEqual(self.start_result.frame.columns[4], "date_delta")
        find_loc = self.start_result.frame[
            self.start_result.frame["names"] == "example.csv"
        ].index[0]
        self.assertEqual(self.start_result.frame.loc[find_loc, "date_delta"], 0)
        find_loc = self.start_result.frame[
            self.start_result.frame["names"] == "19850802_example_short.csv"
        ].index[0]
        self.assertEqual(
            self.start_result.frame.loc[find_loc, "date"], date(1985, 8, 2)
        )
        self.assertNotEqual(self.start_result.frame.loc[find_loc, "date_delta"], 0)
        self.assertEqual(self.start_result.frame.columns[5], "meas1")
        self.assertEqual(self.start_result.frame.columns[6], "meas2")

        result = start(
            self.path, measures=["justone"], function=self.func, function_args=self.farg
        )
        with self.assertRaises(ValueError):
            result.drive()

        result = start(
            self.path, measures="justone", function=self.func, function_args=self.farg
        )
        with self.assertRaises(ValueError):
            result.drive()

        result = start(self.path, function=self.func, function_args=["arg1", "arg2"])
        with self.assertRaises(TypeError):
            result.drive()

    def test_c_map_directory_method(self):

        self.start_result.map_directory()
        self.assertTrue(
            self.start_result.directory_map[self.start_result.directory].__contains__,
            "example.csv",
        )

        with self.assertRaises(ValueError):
            self.start_result.map_directory(only_hidden=True)

    def test_d_sea_method(self):

        with self.assertRaises(KeyError):
            self.start_result.sea(seaborn_args={})

        with self.assertRaises(ValueError):
            self.start_result.sea(
                kind="scatterplot", seaborn_args={"x": 0, "y": 0, "hue": 1}
            )


if __name__ == "__main__":
    pass
