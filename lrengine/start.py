"""
lrdata class
"""

import os
import pandas as pd
import numpy as np
from . import intake


class start:
    """
    lrdata Class

    Attributes:
        directory (str): The path to the parent directory
        patterns (list): List of patterns to recognize in file or folder names
        skip (list): List of patterns used to decide which elements to skip
        func (function): User-defined function that returns classifier(s)

    """

    def __init__(self, directory=[], patterns=[], skip=None, function=None):

        self.directory = directory
        self.patterns = patterns
        self.skip = skip
        self.function = function
        self.frame = pd.DataFrame({})

        self.check_directory(self.directory)
        self.check_patterns(self.patterns)
        self.check_skip(self.skip)
        self.check_function(self.function)

        self.checks_passed()

    def checks_passed(self):

        directs = os.listdir(self.directory)
        df = {"Names": directs}
        for indx in self.patterns:
            df[indx] = np.zeros(len(directs))

        self.frame = pd.DataFrame(df)

        lrdata = {
            "directory": self.directory,
            "patterns": self.patterns,
            "skip": self.skip,
            "function": self.function,
            "frame": self.frame,
        }

        intake.injectors(lrdata)

        return lrdata

    def check_directory(self, directory):

        if not isinstance(directory, str):
            raise TypeError("path to directory must be a string")

        if not os.path.exists(directory):
            raise ValueError("this directory does not exist")

        if not os.path.isdir(directory):
            raise TypeError("this is a file, not a directory")

        if not (len(os.listdir(directory)) > 1) or not isinstance(
            os.listdir(directory), list
        ):
            raise TypeError("the directory must contain at least two files or folders")

    def check_patterns(self, patterns):

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

    def check_skip(self, skip):

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

    def check_function(self, function):

        if not isinstance(function, str):
            raise TypeError("the name of the script must be a string")

        if not os.path.exists(function):
            raise ValueError("this file does not exist")

        if not os.path.isfile(function):
            raise TypeError("this is not a file")
