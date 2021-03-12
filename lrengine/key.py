"""
lrdata class
"""

import os
import types

class lrdata:
    """
    lrdata Class

    Attributes:
        directory (str): The path to the parent directory
        patterns (list): List of patterns to recognize in file or folder names
        skip (list): List of patterns used to decide which elements to skip
        func (function): User-defined function that returns classifier(s)

    """

    def __init__(self, directory, patterns, skip=None, function=None):

        self.directory = directory
        self.patterns = patterns
        self.skip = skip
        self.function = function

        self.check_directory(self.directory)

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

        self.check_patterns(self.patterns)

    def check_patterns(self, patterns):

        if not isinstance(patterns, list):
            raise TypeError("patterns must be a list of strings")

        for indx, items in enumerate(patterns):
            if not isinstance(items, str):
                raise TypeError(
                    "all items in the patterns list must be strings, item ["
                    + str(indx)
                    + "] is not class 'str'"
                )

        self.check_skip(self.skip)

    def check_skip(self, skip):

        if not isinstance(skip, list) and (skip is not None):
            raise TypeError("skip must be a list or None")

        if isinstance(skip, list):
            for indx, items in enumerate(skip):
                if not isinstance(items, str):
                    raise TypeError(
                        "all items in the skip list must be strings, item ["
                        + str(indx)
                        + "] is not class 'str'"
                    )

        self.check_func(self.function)

    def check_function(self, function):

        if not isinstance(function, types.FunctionType):
            raise TypeError("this is not a function")
        if function is None:
            raise NameError("you must supply a function that returns a qualifier")

        self.checks_passed()

    def checks_passed(self):

        return {
                "directory": self.directory,
                "patterns": self.patterns,
                "skip": self.skip,
                "function": self.function,
               }
