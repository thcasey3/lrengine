"""
start class, checks and packages the inputs and sends them to intake
"""

import os
import pandas as pd
import numpy as np
from datetime import date
from . import intake, engine, tools


class start:
    """
    start class

    Attributes:
        directory (str): The path to the parent directory, or .csv that can be made into a Pandas DataFrame
        patterns (list): List of patterns to recognize in file or folder names
        skip (list): List of patterns used to decide which elements to skip
        date_format (str): format of date string to search for
        classifiers (list): User-defined classifier(s)
        function (function): User-defined function that returns classifier value(s)
        function_args (dict): Dictionary of arguments for user-defined function
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

    def _checks_passed(self):

        if os.path.isfile(self.directory) and ".csv" in self.directory:
            self.frame = pd.read_csv(self.directory)
            lrdata = {
                "directory": self.directory,
                "frame": self.frame,
            }
        else:
            if self.skip is not None:
                if isinstance(self.sub_directories, list):
                    sub_dir = []
                    for _, subdir in enumerate(self.sub_directories):
                        if not any(map(subdir.__contains__, self.skip)):
                            sub_dir.append(subdir)
                    if not self.sub_directories:
                        raise TypeError(
                            "Your skip patterns removed all of your sub-directories!"
                        )
                    else:
                        self.sub_directories = sub_dir

                elif isinstance(self.sub_directories, str):
                    if any(map(self.sub_directories.__contains__, self.skip)):
                        raise TypeError(
                            "Your skip patterns removed your only directory!"
                        )
                else:
                    raise TypeError("No directories to use")

            df = {"names": self.sub_directories}
            if self.patterns:
                for patts in self.patterns:
                    df[patts] = np.zeros(len(self.sub_directories))

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

    def reduce_dates(self, format=None):

        if format and "date_format" in self.frame.columns:
            new_formats = list(np.zeros(len(self.frame)))
            new_dates = list(np.zeros(len(self.frame)))
            new_deltas = list(np.zeros(len(self.frame)))
            for indx, names in enumerate(self.frame["date_format"]):
                if isinstance(names, list):
                    for indx2, forms in enumerate(names):
                        if forms == format:
                            new_formats[indx] = self.frame.loc[indx, "date_format"][
                                indx2
                            ]
                            new_dates[indx] = self.frame.loc[indx, "date"][indx2]
                            new_deltas[indx] = self.frame.loc[indx, "date_delta"][indx2]

                elif isinstance(names, str):
                    if names == format:
                        new_formats[indx] = self.frame.loc[indx, "date_format"]
                        new_dates[indx] = self.frame.loc[indx, "date"]
                        new_deltas[indx] = self.frame.loc[indx, "date_delta"]

            self.frame["date_format"] = new_formats
            self.frame["date"] = new_dates
            self.frame["date_delta"] = new_deltas
            # self.frame = self.frame.assign(date_format=new_formats)
            # self.frame = self.frame.assign(date=new_dates)
            # self.frame = self.frame.assign(date_delta=new_deltas)

            return self.frame[["date", "date_format", "date_delta"]]

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

        if (
            not isinstance(items, list)
            and not isinstance(items, str)
            and items is not None
        ):
            raise TypeError(kind + " must be a list of strings, or None")

        elif isinstance(items, list):
            for indx, item in enumerate(items):
                if not isinstance(item, str):
                    raise TypeError(
                        "all items in the "
                        + kind
                        + " list must be strings, item ["
                        + str(indx)
                        + "] is not class 'str'"
                    )

    @staticmethod
    def check_function(function):

        if not callable(function) and function is not None:
            raise TypeError("this function is not callable")

    @staticmethod
    def check_date_format(date_format):

        if not date_format in [
            "any",
            "YYYYMMDD",
            "YYYYDDMM",
            "MMDDYYYY",
            "DDMMYYYY",
            "YYMMDD",
            "YYDDMM",
            "MMDDYY",
            "DDMMYY",
            "YYYY-MM-DD",
            "YYYY-DD-MM",
            "MM-DD-YYYY",
            "DD-MM-YYYY",
            "YY-MM-DD",
            "YY-DD-MM",
            "MM-DD-YY",
            "DD-MM-YY",
            "YYYY_MM_DD",
            "YYYY_DD_MM",
            "MM_DD_YYYY",
            "DD_MM_YYYY",
            "YY_MM_DD",
            "YY_DD_MM",
            "MM_DD_YY",
            "DD_MM_YY",
            "YYYY/MM/DD",
            "YYYY/DD/MM",
            "MM/DD/YYYY",
            "DD/MM/YYYY",
            "YY/MM/DD",
            "YY/DD/MM",
            "MM/DD/YY",
            "DD/MM/YY",
        ]:
            raise ValueError("This date format is not allowed, see documentation")
