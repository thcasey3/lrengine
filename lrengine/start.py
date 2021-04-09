"""
start module, checks and packages the inputs and sends them to intake
"""

import os
import pandas as pd
import numpy as np
from datetime import date
from . import intake, engine, tools


class start:
    """
    start class for creating the object used by lrengine

    Args:
        directory (str): The path to the parent directory, or .csv that can be made into a Pandas DataFrame
        patterns (list or dict): List of patterns or dict of custom pattern pairs to recognize in file or folder names
        skip (list): List of patterns used to decide which elements to skip
        date_format (str): format of date string to search for
        classifiers (list): User-defined classifier(s)
        function (function): User-supplied function that returns classifier value(s)
        function_args (dict): Dictionary of arguments for user-defined function

    Returns:
        start object
    """

    def __init__(
        self,
        directory=None,
        patterns=None,
        skip=None,
        date_format=None,
        classifiers=None,
        function=None,
        function_args=None,
    ):

        self.directory = directory
        self.patterns = patterns
        self.skip = skip
        self.date_format = date_format
        self.classifiers = classifiers
        self.function = function
        self.function_args = function_args
        self.frame = pd.DataFrame({})

        if os.path.isfile(directory) and ".csv" in directory:
            self.check_file(self.directory)
            self.sub_directories = []
        else:
            self.sub_directories = os.listdir(self.directory)
            self.check_directory(self.directory)
            self.check_lists(self.patterns, "patterns")
            self.check_lists(self.skip, "skip")
            self.check_lists(self.classifiers, "classifiers")
            self.check_function(self.function)

        if date_format:
            self.check_date_format(date_format)

        self._checks_passed()

        if self.date_format:
            intake.date_injectors(self)
        if self.patterns:
            intake.pattern_injectors(self)
        if self.skip:
            intake.patterns_filter(self, remove=self.skip)

    def _checks_passed(self):

        if os.path.isfile(self.directory) and ".csv" in self.directory:
            self.frame = pd.read_csv(self.directory)
            lrdata = {
                "directory": self.directory,
                "frame": self.frame,
            }
        else:
            df = {"names": self.sub_directories}

            self.frame = pd.DataFrame(df)

            lrdata = {
                "directory": self.directory,
                "patterns": self.patterns,
                "skip": self.skip,
                "date_format": self.date_format,
                "classifiers": self.classifiers,
                "function": self.function,
                "function_args": self.function_args,
                "frame": self.frame,
            }

        return lrdata

    def drive(self):

        if not self.function:
            raise TypeError(
                "this object was created from a csv, the .run() method is not allowed"
            )
        else:
            return engine.cylinders(self)

    def map_directory(
        self,
        skip=[],
        skip_empty=True,
        skip_hidden=True,
        only_hidden=False,
        walk_topdown=True,
    ):

        if os.path.isdir(self.directory):
            self.directory_map = {}
            for root, dirs, files in os.walk(self.directory, topdown=walk_topdown):
                if root == self.directory:
                    self.directory_map[root] = files
                else:
                    self.directory_map[root.replace(self.directory, "")] = files

            skip_list = []
            if only_hidden:
                for ky in self.directory_map.keys():
                    if not ky == "":
                        skip_list.append(ky)
                if len(skip_list) == len(self.directory_map.keys()):
                    raise ValueError("No hidden directories were found")
            else:
                if skip_hidden:
                    try:
                        self.directory_map.pop("")
                    except KeyError:
                        pass

                for ky in self.directory_map.keys():
                    if skip_empty:
                        if len(self.directory_map[ky]) == 0:
                            skip_list.append(ky)
                    if skip:
                        if any(map(ky.__contains__, skip)):
                            skip_list.append(ky)

            skip_list = set(skip_list)

            for sl in skip_list:
                self.directory_map.pop(sl)

            return self.directory_map

        else:
            raise TypeError("This is not a path to a directory")

    def sea(self, kind="relplot", seaborn_args={}):

        if not all(map(seaborn_args.keys().__contains__, ["x", "y"])):
            raise KeyError(
                "you must specify at least 'x', 'y', and 'hue' in your options dictionary. see seaborn documentation"
            )
        else:
            return tools.sea_born.sea(
                df=self.frame, kind=kind, seaborn_args=seaborn_args
            )

    def save(self, filename=None, header=True):

        if filename:
            if not isinstance(filename, str):
                raise TypeError(
                    "Filename must be a string, make sure to add .csv to the end"
                )
            if ".csv" not in filename:
                filename = filename + ".csv"
        else:
            filename = os.path.join(
                self.directory, str(date.today()) + "_DataFrame.csv"
            )

        self.frame.to_csv(filename, header=header)

    def find_dates(self):

        intake.date_injectors(self)
        if "date" in self.frame.keys():
            return self.frame[["date", "date_delta"]]
        else:
            return self.frame

    def reduce_dates(self, remove=None, keep=None):

        intake.dates_filter(self, remove=remove, keep=keep)
        if "date_format" in self.frame.columns:
            return self.frame[["date", "date_format", "date_delta"]]
        elif "date" in self.frame.columns:
            return self.frame[["date", "date_delta"]]
        else:
            return self.frame

    def find_patterns(self):

        intake.pattern_injectors(self)
        return self.frame["names"]

    def reduce_names(self, remove=None, keep=None):

        intake.patterns_filter(self, remove=remove, keep=keep)
        return self.frame["names"]

    @staticmethod
    def check_directory(directory):

        if not isinstance(directory, str):
            raise TypeError("path to directory must be a string")

        if not os.path.exists(directory):
            raise ValueError("this directory does not exist")

    @staticmethod
    def check_file(directory):

        frame = pd.read_csv(directory)

        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "Importing the csv does not yield a pandas DataFrame, check the csv file format"
            )

    @staticmethod
    def check_lists(items, kind):

        if not type(items) in [list, str, dict] and items is not None:
            raise TypeError(kind + " must be a list of strings, or None")

        if isinstance(items, list):
            for indx, item in enumerate(items):
                if not isinstance(item, str):
                    raise TypeError(
                        "all items in the "
                        + kind
                        + " list must be strings, item ["
                        + str(indx)
                        + "] is not class 'str'"
                    )

        if isinstance(items, dict):
            for item in items.keys():
                if not isinstance(items[item], str) and items[item] is not bool:
                    raise TypeError(
                        "all values in the "
                        + kind
                        + " dict must be strings, item "
                        + item
                        + " is not class 'str'"
                    )

    @staticmethod
    def check_function(function):

        if not callable(function) and function is not None:
            raise TypeError("this function is not callable")

    @staticmethod
    def check_date_format(date_format):

        allowed_list = [
            "any",
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
            "YY-MM-DD",
            "YY-DD-MM",
            "MM-DD-YY",
            "DD-MM-YY",
            "YYYY-M-DD",
            "YYYY-DD-M",
            "M-DD-YYYY",
            "DD-M-YYYY",
            "YY-M-DD",
            "YY-DD-M",
            "M-DD-YY",
            "DD-M-YY",
            "YYYY-MM-D",
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
            "YY-D-M",
            "M-D-YY",
            "D-M-YY",
        ]
        if (
            isinstance(date_format, list)
            and not all(map(allowed_list.__contains__, date_format))
        ) or (isinstance(date_format, str) and date_format not in allowed_list):
            raise ValueError(
                "The date format "
                + date_format
                + " is not one of the allowed options, see documentation"
            )
