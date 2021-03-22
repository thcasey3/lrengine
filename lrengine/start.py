"""
start class, checks and packages the inputs and sends them to intake
"""

import os
import pandas as pd
import numpy as np
from . import intake, tools


class start:
    """
    start class

    Attributes:
        directory (str): The path to the parent directory
        patterns (list): List of patterns to recognize in file or folder names
        skip (list): List of patterns used to decide which elements to skip
        measures (list): User-defined classifier(s)
        function (function): User-defined function that returns classifier values(s)
        function_args (dict): Dictionary of arguments for user-defined function
    """

    def __init__(
        self,
        directory=[],
        patterns=[],
        skip=None,
        measures=None,
        function=None,
        function_args=None,
    ):

        self.directory = directory
        self.patterns = patterns
        self.skip = skip
        self.measures = measures
        self.function = function
        self.function_args = function_args
        self.frame = pd.DataFrame({})

        if isinstance(directory, list):
            self.check_file(self.directory)
            self.sub_directories = []
        else:
            self.sub_directories = os.listdir(self.directory)

            self.check_directory(self.directory, self.sub_directories)
            self.check_patterns(self.patterns)
            self.check_skip(self.skip)
            self.check_measures(self.measures)
            self.check_function(self.function)

        self.checks_passed()

    def checks_passed(self):

        if isinstance(self.directory, list) and self.directory[1] == "csv":
            self.frame = pd.read_csv(self.directory[0])
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

            df = {"Names": self.sub_directories}
            if self.patterns:
                for patts in self.patterns:
                    df[patts] = np.zeros(len(self.sub_directories))

            self.frame = pd.DataFrame(df)

        lrdata = {
            "directory": self.directory,
            "patterns": self.patterns,
            "skip": self.skip,
            "measures": self.measures,
            "function": self.function,
            "function_args": self.function_args,
            "frame": self.frame,
        }

        intake.injectors(lrdata)

    def sea(self, kind="replot", options={}):

        tools.sea_born.sea(self.frame, kind, options)

    def skl(self, df, kind="RandomForestClassifier", options={}):

        tools.sk_learn.learn(df, kind, options)

    def tensor(self, df, kind="Sequential", options={}):

        tools.tensor_flow.flow(df, kind, options)

    def sql(self, df, url=""):

        tools.sq_lite.sql(df, url)

    @staticmethod
    def check_directory(directory, sub_directories):

        if not isinstance(directory, str):
            raise TypeError("path to directory must be a string")

        if not os.path.exists(directory):
            raise ValueError("this directory does not exist")

        if "nmr_odnp_data/odnp_data.csv" in directory:
            pass
        else:
            if not os.path.isdir(directory):
                raise TypeError("this is a file, not a directory")

            if not (len(sub_directories) > 1) or not isinstance(sub_directories, list):
                raise TypeError(
                    "the directory must contain at least two files or folders"
                )

    @staticmethod
    def check_file(dir_list):

        if (
            not isinstance(dir_list[0], str)
            or not isinstance(dir_list[1], str)
            or not os.path.isfile(dir_list[0])
        ):
            raise TypeError(
                "you must give a list of strings, [0]='path to file', [1]='file ext'"
            )

        if not dir_list[1] == "csv":
            raise TypeError("file type not supported, only .csv files at this time")

    @staticmethod
    def check_patterns(patterns):

        if (
            not isinstance(patterns, list)
            and not isinstance(patterns, str)
            and patterns is not None
        ):
            raise TypeError("patterns must be a list of strings, or None")

        if isinstance(patterns, list):
            for indx, items in enumerate(patterns):
                if not isinstance(items, str):
                    raise TypeError(
                        "all items in the patterns list must be strings, item ["
                        + str(indx)
                        + "] is not class 'str'"
                    )

    @staticmethod
    def check_skip(skip):

        if (
            not isinstance(skip, list)
            and not isinstance(skip, str)
            and (skip is not None)
        ):
            raise TypeError("skip must be a list or None")

        if isinstance(skip, list):
            if isinstance(skip, list):
                for indx, items in enumerate(skip):
                    if not isinstance(items, str):
                        raise TypeError(
                            "all items in the skip list must be strings, item ["
                            + str(indx)
                            + "] is not class 'str'"
                        )

    @staticmethod
    def check_measures(measures):

        if (
            not isinstance(measures, list)
            and not isinstance(measures, str)
            and (measures is not None)
        ):
            raise TypeError("measures must be a list or None")

        if isinstance(measures, list):
            if isinstance(measures, list):
                for indx, items in enumerate(measures):
                    if not isinstance(items, str):
                        raise TypeError(
                            "all items in the measures list must be strings, item ["
                            + str(indx)
                            + "] is not class 'str'"
                        )

    @staticmethod
    def check_function(function):

        if not callable(function) and function is not None:
            raise TypeError("this function is not callable")
