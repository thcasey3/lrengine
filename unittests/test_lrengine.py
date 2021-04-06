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
            classifiers=self.meas,
            function=self.func,
            function_args=self.farg,
            date_format=self.date_format,
        )

    def test_a_start_object(self):

        self.assertEqual(self.path, self.start_result.directory)
        self.assertEqual(self.subd, os.listdir(self.start_result.directory))
        self.assertEqual(self.meas, self.start_result.classifiers)
        self.assertEqual(self.patt, self.start_result.patterns)
        self.assertEqual(self.skip, self.start_result.skip)
        self.assertEqual(self.func, self.start_result.function)
        self.assertEqual(self.farg, self.start_result.function_args)
        self.assertEqual(self.date_format, self.start_result.date_format)

        with self.assertRaises(TypeError):
            result = start(
                self.path,
                classifiers=("one", "two"),
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

    def test_b_drive_method(self):

        self.start_result.drive()
        self.assertEqual(len(self.start_result.frame.columns), 7)
        self.assertEqual(self.start_result.frame.columns[0], "names")
        self.assertEqual(self.start_result.frame.columns[3], "patt1")
        self.assertEqual(self.start_result.frame.columns[4], "patt2")
        self.assertEqual(self.start_result.frame.columns[1], "date")
        self.assertEqual(self.start_result.frame.columns[2], "date_delta")
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
            self.path,
            classifiers=["justone"],
            function=self.func,
            function_args=self.farg,
        )
        with self.assertRaises(ValueError):
            result.drive()

        result = start(
            self.path,
            classifiers="justone",
            function=self.func,
            function_args=self.farg,
        )
        with self.assertRaises(ValueError):
            result.drive()

        result = start(self.path, function=self.func, function_args=["arg1", "arg2"])
        with self.assertRaises(TypeError):
            result.drive()

    def test_c_patterns_methods(self):

        result = start(self.path, patterns={"example": "_test", "short": bool})
        find_loc = result.frame[result.frame["example"] == "_test"].index[0]
        self.assertFalse(result.frame.loc[find_loc, "short"])
        result.reduce_names(skip=["short"])
        find_loc = result.frame[result.frame["names"] == "example.csv"].index[0]
        self.assertFalse(result.frame.loc[find_loc, "example"])
        find_loc = result.frame[
            result.frame["names"] == "850802_example_test.csv"
        ].index[0]
        self.assertEqual(result.frame.loc[find_loc, "example"], "_test")

    def test_d_skip_methods(self):

        result = start(self.path, skip="short")
        self.assertEqual(len(os.listdir(self.path)) - 2, len(result.frame))

        with self.assertRaises(TypeError):
            result = start(self.path, skip=1)

    def test_e_dates_methods(self):

        self.start_result.date_format = "any"
        self.start_result.find_dates()
        find_loc = self.start_result.frame[
            self.start_result.frame["names"] == "19850802_example_short.csv"
        ].index[0]
        self.assertTrue(isinstance(self.start_result.frame.loc[find_loc, "date"], list))
        self.start_result.reduce_dates(keep="YYYYMMDD")
        self.assertTrue(isinstance(self.start_result.frame.loc[find_loc, "date"], date))

        self.start_result.find_dates()
        self.start_result.reduce_dates(remove=["YYYYMMDD"])
        self.assertTrue(isinstance(self.start_result.frame.loc[find_loc, "date"], list))
        self.assertTrue(
            [
                x
                for x in self.start_result.frame.loc[find_loc, "date"]
                if isinstance(x, date)
            ]
        )

        self.start_result.find_dates()
        self.start_result.reduce_dates(remove=["YYMMDD", "YYDDMM"])
        self.assertTrue(isinstance(self.start_result.frame.loc[find_loc, "date"], list))
        find_loc = self.start_result.frame[
            self.start_result.frame["names"] == "850802_example_test.csv"
        ].index[0]
        self.assertTrue(self.start_result.frame.loc[find_loc, "date_format"] == 0)

    def test_f_map_directory_method(self):

        self.start_result.map_directory()
        self.assertTrue(
            self.start_result.directory_map[self.start_result.directory].__contains__,
            "example.csv",
        )

        with self.assertRaises(ValueError):
            self.start_result.map_directory(only_hidden=True)

    def test_g_sea_method(self):

        with self.assertRaises(KeyError):
            self.start_result.sea(seaborn_args={})

        with self.assertRaises(ValueError):
            self.start_result.sea(
                kind="scatterplot", seaborn_args={"x": 0, "y": 0, "hue": 1}
            )

    def test_h_save_method(self):

        with self.assertRaises(TypeError):
            self.start_result.save(filename=100)

        if "utest.csv" in os.listdir(self.path):
            os.remove(self.path + "utest.csv")

        self.start_result.save(filename=os.path.join(self.path, "utest"))
        self.assertTrue("utest.csv" in os.listdir(self.path))

        os.remove(self.path + "utest.csv")

        self.start_result.save(filename=os.path.join(self.path, "utest.csv"))
        self.assertTrue("utest.csv" in os.listdir(self.path))
        self.assertFalse("utest.csv.csv" in os.listdir(self.path))
        os.remove(self.path + "utest.csv")

        if "utest.csv.csv" in os.listdir(self.path):
            os.remove(self.path + "utest.csv.csv")

        if str(date.today()) + "_DataFrame.csv" in os.listdir(self.path):
            os.remove(os.path.join(self.path, str(date.today()) + "_DataFrame.csv"))

        self.start_result.save()
        self.assertTrue(str(date.today()) + "_DataFrame.csv" in os.listdir(self.path))
        os.remove(os.path.join(self.path, str(date.today()) + "_DataFrame.csv"))


if __name__ == "__main__":
    pass
