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
        patterns (str, list, or dict): pattern(s) to recognize in file or folder name (see docs)
        skip (str or list): pattern(s) used to decide which elements to skip
        date_format (str or list): format(s) of date strings to search for
        classifiers (list): User-defined classifier(s)
        function (function): User-supplied function that returns classifier value(s) as list
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

        if directory is None:
            self.frame = pd.DataFrame({})
            self._empty_object()
        else:
            self.directory = os.path.normpath(directory)
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

    def _empty_object(self):
        lsdata = {"frame": self.frame}
        return lsdata

    def _checks_passed(self):

        if os.path.isfile(self.directory) and ".csv" in self.directory:
            self.frame = pd.read_csv(self.directory)
            lsdata = {
                "directory": self.directory,
                "frame": self.frame,
            }
        else:
            df = {"name": self.sub_directories}

            self.frame = pd.DataFrame(df)

            lsdata = {
                "directory": self.directory,
                "patterns": self.patterns,
                "skip": self.skip,
                "date_format": self.date_format,
                "classifiers": self.classifiers,
                "function": self.function,
                "function_args": self.function_args,
                "frame": self.frame,
            }

        return lsdata

    def drive(self, classifiers=None, function=None, function_args=None):

        if classifiers is not None:
            self.classifiers = classifiers

        if function is not None:
            self.function = function

        if function_args is not None:
            self.function_args = function_args

        if not self.function:
            raise TypeError(
                "this object does not have a function, the .drive() method is not allowed"
            )
        else:
            return engine.cylinders(self)

    def map_directory(
        self,
        directory=None,
        skip=[],
        skip_empty=False,
        skip_hidden=True,
        only_hidden=False,
        walk_topdown=True,
    ):
        if not hasattr(self, "directory") and directory is not None:
            self.directory = os.path.normpath(directory)
        elif hasattr(self, "directory") and directory is not None:
            raise TypeError(
                "Your object contains a directory, and you've given another directory. Please choose one by setting the attribute 'directory' to the directory you wish to map, and do not inlcude the directory keyword arg"
            )
        elif not hasattr(self, "directory") and directory is None:
            raise ValueError("No directory given to map!")

        if isinstance(skip, str):
            skip = [skip]

        if os.path.isdir(self.directory):
            self.directory_map = {}
            for root, dirs, files in os.walk(self.directory, topdown=walk_topdown):
                root = os.path.normpath(root)
                if root == self.directory:
                    self.directory_map[root] = files
                else:
                    self.directory_map[root.replace(self.directory, "")] = files

            skip_list = []
            if only_hidden:
                for x in self.directory_map.keys():
                    name = os.path.basename(os.path.abspath(x))
                    if not name.startswith("."):
                        skip_list.append(x)
                if len(skip_list) == len(self.directory_map.keys()):
                    raise ValueError("No hidden directories were found")
            else:
                for x in self.directory_map.keys():
                    if skip_hidden:
                        name = os.path.basename(os.path.abspath(x))
                        if name.startswith("."):
                            skip_list.append(x)
                    if skip_empty:
                        if (
                            len(self.directory_map[x]) == 0
                            and len(
                                [
                                    dirs
                                    for _, dirs, _ in os.walk(
                                        os.path.normpath(
                                            self.directory
                                            + x.replace(self.directory, "")
                                        )
                                    )
                                ][0]
                            )
                            == 0
                        ):
                            skip_list.append(x)
                    if skip:
                        if any(map(x.__contains__, skip)):
                            skip_list.append(x)
                        else:
                            temp = [
                                y
                                for y in self.directory_map[x]
                                if not any(map(y.__contains__, skip))
                            ]
                            self.directory_map[x] = temp

            skip_list = set(skip_list)

            for sl in skip_list:
                self.directory_map.pop(sl)

            self.max_depth = max([t.count(os.sep) for t in self.directory_map.keys()])

            return self.directory_map

        else:
            raise TypeError("This is not a path to a directory")

    def map_to_frame(self, depth="max", kind="any", to_frame=True):

        if hasattr(self, "directory_map"):
            if isinstance(depth, int):
                depth = [depth]
            if (
                depth != "max"
                and not isinstance(depth, list)
                and not any([isinstance(x, int) for x in depth])
            ):
                raise TypeError("depth must be single int or list of int")
            new_names = []
            if depth == "max" and kind == "folders":
                new_names = [os.path.normpath(x) for x in self.directory_map.keys()]
            elif depth == [0]:
                new_names = self.directory_map[list(self.directory_map.keys())[0]]
            elif isinstance(depth, list) and kind == "folders":
                new_names = [
                    os.path.normpath(x)
                    for x in self.directory_map.keys()
                    if x.count(os.sep) in depth
                ]
            elif kind == "files" or kind == "any":
                temp = []
                for x in self.directory_map.keys():
                    if kind == "any":
                        temp.append(os.path.normpath(x))
                    for y in self.directory_map[x]:
                        temp.append(os.path.normpath(os.path.join(x, y)))
                if isinstance(depth, list):
                    for t in temp:
                        if 0 in depth:
                            new_names = [
                                x
                                for x in self.directory_map[
                                    list(self.directory_map.keys())[0]
                                ]
                            ]
                            depth.pop(0)
                        if t.count(os.sep) in depth:
                            new_names.append(t)
                elif depth == "max":
                    new_names = temp

            if new_names:
                if hasattr(self, "directory") and depth != [0]:
                    new_names = [
                        x.replace(self.directory, "").strip(os.sep) for x in new_names
                    ]

                frame = pd.DataFrame(new_names, columns=["name"])

                if to_frame:
                    self.frame = frame

                return frame
            else:
                raise ValueError(
                    "The given options would return an empty directory_map"
                )
        else:
            raise TypeError(
                "You must map_directory() first in order to use the map_to_frame() method"
            )

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

    def on_date(self, keep=None, remove=None, strip_zeros=True):

        intake.dates_filter(
            self, keep=keep, remove=remove, strip_zeros=strip_zeros, which="ondate"
        )

        if len(self.frame) != 0:
            return self.frame["name"]
        else:
            raise TypeError("No items in frame for given date")

    def in_range(self, keep=None, remove=None, strip_zeros=True):

        intake.dates_filter(
            self, keep=keep, remove=remove, strip_zeros=strip_zeros, which="inrange"
        )

        if len(self.frame) != 0:
            return self.frame[["name", "date"]]
        else:
            raise TypeError("No items in frame for given date range(s)")

    def reduce_dates(self, remove=None, keep=None, only_unique=True, strip_zeros=False):

        intake.dates_filter(
            self,
            remove=remove,
            keep=keep,
            only_unique=only_unique,
            strip_zeros=strip_zeros,
            which="reduce",
        )
        if "date_format" in self.frame.columns:
            return self.frame[["date", "date_format", "date_delta"]]
        elif "date" in self.frame.columns:
            return self.frame[["date", "date_delta"]]
        else:
            return self.frame

    def find_patterns(self):

        intake.pattern_injectors(self)
        return self.frame["name"]

    def reduce_names(self, remove=None, keep=None):

        intake.patterns_filter(self, remove=remove, keep=keep)
        return self.frame["name"]

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
