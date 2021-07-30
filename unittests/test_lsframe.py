import unittest
import numpy as np
import os
import pandas as pd
from datetime import date
from lsframe import start, intake, engine, tools
from numpy.testing import assert_array_equal


class startTester(unittest.TestCase):
    def setUp(self):
        self.path = os.path.normpath("./data/")
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
        self.assertEqual(self.start_result.frame.columns[0], "name")
        self.assertEqual(self.start_result.frame.columns[3], "patt1")
        self.assertEqual(self.start_result.frame.columns[4], "patt2")
        self.assertEqual(self.start_result.frame.columns[1], "date")
        self.assertEqual(self.start_result.frame.columns[2], "date_delta")
        find_loc = self.start_result.frame[
            self.start_result.frame["name"] == "example.csv"
        ].index[0]
        self.assertEqual(self.start_result.frame.loc[find_loc, "date_delta"], 0)
        find_loc = self.start_result.frame[
            self.start_result.frame["name"] == "19850802_example_short.csv"
        ].index[0]
        self.assertEqual(
            self.start_result.frame.loc[find_loc, "date"], date(1985, 8, 2)
        )
        self.assertNotEqual(self.start_result.frame.loc[find_loc, "date_delta"], 0)
        self.assertEqual(self.start_result.frame.columns[5], "meas1")
        self.assertEqual(self.start_result.frame.columns[6], "meas2")

        result = start(self.path)
        result.drive(
            classifiers=["new1", "new2"], function=self.func, function_args=self.farg
        )
        self.assertEqual(result.frame.columns[1], "new1")
        self.assertEqual(result.frame.columns[2], "new2")

        self.start_result.function = tools.utest2.proc_epr
        self.start_result.drive()
        self.assertTrue([x == "null" for x in self.start_result.frame["meas1"]])
        self.assertTrue([x == "null" for x in self.start_result.frame["meas2"]])

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
        result.reduce_names(remove=["short"])
        find_loc = result.frame[result.frame["name"] == "example.csv"].index[0]
        self.assertFalse(result.frame.loc[find_loc, "example"])
        find_loc = result.frame[
            result.frame["name"] == "850802_example_test.csv"
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
            self.start_result.frame["name"] == "19850802_example_short.csv"
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

        find_loc = self.start_result.frame[
            self.start_result.frame["name"] == "850802_example_test.csv"
        ].index[0]
        self.start_result.find_dates()
        self.assertTrue(
            date(1985, 8, 2) in self.start_result.frame.loc[find_loc, "date"]
        )
        self.start_result.reduce_dates(remove=["YYMMDD", "YYDDMM"])
        self.assertTrue(isinstance(self.start_result.frame.loc[find_loc, "date"], list))
        self.assertTrue(
            self.start_result.frame.loc[find_loc, "date_format"]
            == ["YYMDD", "MDDYY", "YYDMM", "DMMYY", "MDYY", "DMYY"]
        )
        self.assertFalse(
            date(1985, 8, 2) in self.start_result.frame.loc[find_loc, "date"]
        )

        self.start_result.frame = pd.DataFrame(
            [
                "19950509",
                "19950905",
                "05091995",
                "09051995",
                "950509",
                "950905",
                "050995",
                "090595",
                "1995509",
                "1995095",
                "5091995",
                "0951995",
                "95509",
                "95095",
                "50995",
                "09595",
                "1995059",
                "1995905",
                "0591995",
                "9051995",
                "95059",
                "95905",
                "05995",
                "90595",
                "199559",
                "199595",
                "591995",
                "951995",
                "9559",
                "9595",
                "5995",
                "9595",
                "1995-05-09",
                "1995-09-05",
                "05-09-1995",
                "09-05-1995",
                "95-05-09",
                "95-09-05",
                "05;09:95",
                "09-05-95",
                "1995-5-09",
                "1995-09-5",
                "5_09-1995",
                "09-5-1995",
                "95-5-09",
                "95-09-5",
                "5-09-95",
                "09-5-95",
                "1995-05-9",
                "1995-9-05",
                "05-9-1995",
                "9-05-1995",
                "95-05-9",
                "95-9-05",
                "05-9-95",
                "9-05-95",
                "1995;5;9",
                "1995:9:5",
                "5/9/1995",
                "9_5_1995",
                "95-5-9",
                "95-9-5",
                "5-9-95",
                "9-5-95",
            ],
            columns=["name"],
        )

        date_strngs = [
            "YYYYMMDD",
            "YYYYDDMM",
            "MMDDYYYY",
            "DDMMYYYY",
            "YYMMDD",
            "YYDDMM",
            "MMDDYY",
            "DDMMYY",
            "YYYYMDD",
            "YYYYDDM",
            "MDDYYYY",
            "DDMYYYY",
            "YYMDD",
            "YYDDM",
            "MDDYY",
            "DDMYY",
            "YYYYMMD",
            "YYYYDMM",
            "MMDYYYY",
            "DMMYYYY",
            "YYMMD",
            "YYDMM",
            "MMDYY",
            "DMMYY",
            "YYYYMD",
            "YYYYDM",
            "MDYYYY",
            "DMYYYY",
            "YYMD",
            "YYDM",
            "MDYY",
            "DMYY",
            "YYYY-MM-DD",
            "YYYY-DD-MM",
            "MM-DD-YYYY",
            "DD-MM-YYYY",
            "YY;MM;DD",
            "YY-DD-MM",
            "MM-DD-YY",
            "DD-MM-YY",
            "YYYY-M-DD",
            "YYYY/DD/M",
            "M-DD-YYYY",
            "DD-M-YYYY",
            "YY-M-DD",
            "YY-DD-M",
            "M-DD-YY",
            "DD-M-YY",
            "YYYY:MM:D",
            "YYYY-D-MM",
            "MM-D-YYYY",
            "D-MM-YYYY",
            "YY-MM-D",
            "YY-D-MM",
            "MM-D-YY",
            "D-MM-YY",
            "YYYY-M-D",
            "YYYY-D-M",
            "M-D-YYYY",
            "D-M-YYYY",
            "YY-M-D",
            "YY_D_M",
            "M-D-YY",
            "D-M-YY",
        ]

        for indx, dat in enumerate(self.start_result.frame["name"]):
            self.start_result.date_format = date_strngs[indx]
            self.start_result.find_dates()
            self.assertTrue(isinstance(self.start_result.frame.loc[indx, "date"], date))
            self.assertEqual(
                self.start_result.frame.loc[indx, "date"], date(1995, 5, 9)
            )

        self.start_result.date_format = "any"
        self.start_result.find_dates()
        self.assertEqual(len(self.start_result.frame.loc[0, "date_format"]), 9)
        self.start_result.reduce_dates()
        self.assertEqual(len(self.start_result.frame.loc[0, "date_format"]), 6)

        self.start_result.frame = pd.DataFrame(
            ["19900509", "19950509", "20000509", "20050509", "20100509"],
            columns=["name"],
        )
        self.start_result.date_format = "YYYYMMDD"
        self.start_result.find_dates()
        self.assertEqual(len(self.start_result.frame), 5)
        self.start_result.in_range(
            keep=[["1989-05-09", "1999-05-09"], ["2001-05-09", "2011-05-09"]]
        )
        self.assertEqual(len(self.start_result.frame), 4)
        self.start_result.on_date(remove="1990-05-09")
        self.assertEqual(len(self.start_result.frame), 3)

        self.start_result.frame = pd.DataFrame(
            [
                "19900509",
                "19950509",
                "20000509",
                "20050509",
                "20100509",
                "test1",
                "test2",
                "test3",
            ],
            columns=["name"],
        )
        self.start_result.date_format = "YYYYMMDD"
        self.start_result.find_dates()
        self.assertEqual(len(self.start_result.frame), 8)
        self.start_result.in_range(
            remove=[["1989-05-09", "1994-05-09"], ["2001-05-09", "2006-05-09"]],
            strip_zeros=False,
        )
        self.assertEqual(len(self.start_result.frame), 6)
        self.start_result.on_date(keep="2010-05-09", strip_zeros=False)
        self.assertEqual(len(self.start_result.frame), 4)
        self.start_result.reduce_dates(strip_zeros=True)
        self.assertEqual(len(self.start_result.frame), 1)

    def test_f_map_directory_methods(self):

        self.start_result.map_directory()
        self.assertTrue(
            self.start_result.directory_map[self.start_result.directory].__contains__,
            "example.csv",
        )

        lsobject = start()
        lsobject.directory_map = {
            "test_red": ["r", "re", "rd", "ed"],
            "test_blue": ["b", "bl", "blu"],
            "test_green": ["g", "gr"],
            os.path.join("test_red", "maroon"): ["r", "re", "rd", "ed"],
            os.path.join("test_blue", "navy", "sky"): ["b", "bl", "blu"],
            os.path.join("test_green", "winter", "neon", "forest"): ["g", "gr"],
        }
        self.assertEqual(len(lsobject.frame), 0)
        lsobject.map_to_frame(depth=2, kind="folders", to_frame=True)
        self.assertEqual(len(lsobject.frame), 1)
        new = lsobject.map_to_frame(depth=1, kind="files", to_frame=False)
        self.assertEqual(len(new), 9)
        new = lsobject.map_to_frame(depth=3, kind="files", to_frame=False)
        self.assertEqual(len(new), 3)
        lsobject.map_to_frame(depth=4, kind="any", to_frame=True)
        self.assertEqual(len(lsobject.frame), 2)

        lsobject.map_directory(self.path, skip=["short", ".DS_Store"])
        self.assertEqual(len(lsobject.directory_map.keys()), 1)
        self.assertEqual(
            len(lsobject.directory_map[list(lsobject.directory_map.keys())[0]]), 2
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
            os.remove(os.path.join(self.path, "utest.csv"))

        self.start_result.save(filename=os.path.join(self.path, "utest"))
        self.assertTrue("utest.csv" in os.listdir(self.path))

        os.remove(os.path.join(self.path, "utest.csv"))

        self.start_result.save(filename=os.path.join(self.path, "utest.csv"))
        self.assertTrue("utest.csv" in os.listdir(self.path))
        self.assertFalse("utest.csv.csv" in os.listdir(self.path))
        os.remove(os.path.join(self.path, "utest.csv"))

        if "utest.csv.csv" in os.listdir(self.path):
            os.remove(os.path.join(self.path, "utest.csv.csv"))

        if str(date.today()) + "_DataFrame.csv" in os.listdir(self.path):
            os.remove(os.path.join(self.path, str(date.today()) + "_DataFrame.csv"))

        self.start_result.save()
        self.assertTrue(str(date.today()) + "_DataFrame.csv" in os.listdir(self.path))
        os.remove(os.path.join(self.path, str(date.today()) + "_DataFrame.csv"))


if __name__ == "__main__":
    pass
