"""
start class, checks and packages the inputs and sends them to intake
"""

import os
import pandas as pd
import numpy as np
from . import intake, engine, tools


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
        date_format=None,
        measures=None,
        function=None,
        function_args=None,
    ):

        self.directory = directory
        self.patterns = patterns
        self.skip = skip
        self.date_format = date_format
        self.measures = measures
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
            self.check_lists(self.measures, "measures")
            self.check_function(self.function)

        if date_format is not None:
            self.check_date_format(date_format)

        self.checks_passed()

    def checks_passed(self):

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
                "measures": self.measures,
                "function": self.function,
                "function_args": self.function_args,
                "frame": self.frame,
            }

        intake.injectors(lrdata)

    def run(self):

        if not self.function and self.function is not None:
            raise TypeError(
                "this object was created from a csv, the .run() method is not allowed"
            )
        else:
            return engine.cylinders(self)

    def sea(self, options={}):

        if not all(map(options.keys().__contains__, ["x", "y"])):
            raise KeyError(
                "you must specify at least 'x', 'y', and 'hue' in your options dictionary. see seaborn documentation"
            )
        else:
            return tools.sea_born.sea(df=self.frame, type="relplot", options=options)

    def skl(self, kind="", options={}):

        tools.sk_learn.learn(self.frame, kind, options)

    def tensor(self, kind="", options={}):

        tools.tensor_flow.flow(self.frame, kind, options)

    def spark(self, kind="", options={}):

        tools.sea_born.sea(self.frame, kind, options)

    def sql(self, url="", options={}):

        tools.sq_lite.sql(self.frame, url, options)

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

        if not date_format in ["YYYYMMDD", "MMDDYY", "MMDDYYYY", "DDMMYY", "DDMMYYYY"]:
            raise ValueError(
                "Allowed values for date_format are None, 'YYYYMMDD', 'MMDDYY', 'MMDDYYYY', 'DDMMYY', or 'DDMMYYYY'"
            )
