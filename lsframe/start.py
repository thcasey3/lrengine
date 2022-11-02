"""
start module, creates the core object of lsframe
"""

import os
import pandas as pd
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

    def pipeline(self, pipe):
        """
        Apply adtk, statsmodels, scikit, and custom functions sequentially to use the results across methods

        Args:
            pipe (list or tuple of list-like): See examples for

        Returns:
            Adds adtk_object and/or statsmodels_object and/or scikit_object to the start object
        """
        if not isinstance(pipe, (list, tuple)):
            raise TypeError(
                "you must give a list or tuple of lists(s) or tuple(s), e.g. [('adtk', {'time': column1, 'series': column2}), etc.]"
            )
        adtk_inputs = {
            "time": None,
            "series": None,
            "method": "OutlierDetector",
            "adtk_args": {},
            "select": "all",
            "append_object": True,
            "survey": False,
            "plot": False,
        }
        stats_inputs = {
            "forecast_df": None,
            "time": None,
            "endog": None,
            "exog": None,
            "statsmodels_args": {},
            "model": "ARIMA",
            "steps": 3,
            "select": "all",
            "append_object": True,
            "survey": False,
            "plot": False,
        }
        scikit_inputs = {
            "method": "regress",
            "model": "RandomForest",
            "scikit_args": {},
            "select": "all",
            "append_object": True,
            "survey": False,
            "plot": False,
        }

        for items in pipe:
            if items[0] == "adtk":
                adtk_inputs.update(items[1])
                self.adtk(
                    time=adtk_inputs["time"],
                    series=adtk_inputs["series"],
                    method=adtk_inputs["series"],
                    adtk_args=adtk_inputs["adtk_args"],
                    select=adtk_inputs["select"],
                    append_object=adtk_inputs["append_object"],
                    survey=adtk_inputs["survey"],
                    plot=adtk_inputs["plot"],
                )
            if items[0] == "statsmodels":
                stats_inputs.update(items[1])
                self.statsmodels(
                    forecast_df=stats_inputs["forecast_df"],
                    time=stats_inputs["time"],
                    endog=stats_inputs["endog"],
                    exog=stats_inputs["exog"],
                    statsmodels_args=stats_inputs["statsmodels_args"],
                    model=stats_inputs["model"],
                    steps=stats_inputs["steps"],
                    select=stats_inputs["select"],
                    append_object=stats_inputs["append_object"],
                    survey=stats_inputs["survey"],
                    plot=stats_inputs["plot"],
                )
            if items[0] == "scikit":
                scikit_inputs.update(items[1])
                self.scikit(
                    method=scikit_inputs["method"],
                    model=scikit_inputs["model"],
                    scikit_args=scikit_inputs["scikit_args"],
                    select=scikit_inputs["select"],
                    append_object=scikit_inputs["append_object"],
                    survey=scikit_inputs["survey"],
                    plot=scikit_inputs["plot"],
                )
            if items[0] == "function":
                self.frame = items[1](items[2])

    def adtk(
        self,
        time=None,
        series=None,
        method="OutlierDetector",
        adtk_args={},
        select="all",
        append_object=True,
        survey=False,
        plot=True,
    ):
        """
        Use adtk to detect anomalies in frame (https://adtk.readthedocs.io/en/stable/index.html)

        Args:
            time (str or list-like): frame column name or time index for series
            series (stror list-like): frame column name or series to analyze
            method (str), optional: see adtk docs for available Detectors, enter Detector function name as str
            adtk_args (dict), optional: {'arg': value} to change from the defaults for the given method
            select (dict), optional: {'column': list(values)} to subset and analyze individually from the frame, default is 'all'
            append_object (bool), optional: True means collect selections or False means purge them when select is used
            survey (dict), optional: {'select': {'column': list(values)}, 'survey parameter': list(value range)}, survey['select'] can be 'all' to use entire frame
            plot (bool), optional: True means raise plot after analysis

        Returns:
            Adds adtk_object to the start object
        """
        if time is None or series is None:
            raise KeyError(
                "you must at least give values for the arguments time and series. see documentation"
            )
        else:
            self.adtk_object = tools.utilities.anomalies(
                df=self.frame, time=time, series=series, update_config=adtk_args
            )
            self.adtk_object.detect(
                select=select,
                append_object=append_object,
                method=method,
                survey=survey,
                plot=plot,
            )

    def statsmodels(
        self,
        forecast_df=None,
        time=None,
        endog=None,
        exog=None,
        statsmodels_args={},
        model="ARIMA",
        steps=3,
        select="all",
        append_object=True,
        survey=False,
        plots=True,
    ):
        """
        Use statsmodels to analyze a given series from frame as a timeseries (https://www.statsmodels.org/stable/index.html)

        Args:
            forecast_df (pd.DataFrame), optional: DataFrame of exogenous variables used to forecast out-of-sample
            time (str or list-like): frame column name for the time index
            endog (stror list-like): frame column name for the endogenous variable
            exog (str), optional: frame column name for the exogenous variable(s)
            statsmodels_args (dict), optional: {'arg': value} to change from the defaults for the given model
            select (dict), optional: {'column': list(values)} to subset and analyze individually from the frame, default is 'all'
            model (str), optional: 'ARIMA' is default, 'SARIMAX' is the only other option
            steps (int), optional: forecast steps
            append_object (bool), optional: True means collect selections or False means purge them when select is used
            survey (dict), optional: {'select': {'column': list(values)}, 'survey parameter': list(value range)}, survey['select'] can be 'all' to use entire frame
            plot (bool), optional: True means raise plot after analysis

        Returns:
            Adds statsmodels_object to the start object
        """
        if time is None or endog is None:
            raise KeyError(
                "you must at least give values for the arguments time and endog. see documentation"
            )
        else:
            self.statsmodels_object = tools.timeseries.statsmodels(
                df=self.frame,
                forecast_df=forecast_df,
                time=time,
                endog=endog,
                exog=exog,
                update_config=statsmodels_args,
            )
            self.statsmodels_object.forecast(
                select=select,
                model=model,
                steps=steps,
                append_object=append_object,
                survey=survey,
                plots=plots,
            )

    def scikit(
        self,
        method="regress",
        model="RandomForest",
        scikit_args={},
        select="all",
        append_object=True,
        survey=False,
        plots=True,
    ):
        """
        Use scikit-learn to perform machine learning using frame (https://scikit-learn.org/stable/index.html)

        Args:
            method (str), optional: 'regress', 'classify', 'cluster', 'semi_supervised'
            model (str or list), optional: model used to learn default is 'RandomForest'
            scikit_args (dict), optional: {'arg': value} to change from the defaults for the method given in scikit_args
            select (dict), optional: {'column': list(values)} to subset and analyze individually from the frame, default is 'all'
            append_object (bool), optional: True means collect selections or False means purge them when select is used
            survey (dict), optional: {'select': {'column': list(values)}, 'survey parameter': list(value range)}, survey['select'] can be 'all' to use entire frame
            plot (bool), optional: True means raise plot after analysis

        Returns:
            Adds scikit_object to the start object
        """
        if not all(map(scikit_args.keys().__contains__, ["Xcolumns", "ycolumn"])):
            raise KeyError(
                "you must specify at least 'Xcolumns' and 'ycolumn' in your scikit_args dictionary. see documentation"
            )
        else:
            self.scikit_object = tools.mlearn.scikit(
                df=self.frame, method=method, model=model, update_config=scikit_args
            )
            self.scikit_object.learn(
                select=select, append_object=append_object, survey=survey, plots=plots
            )

    def seaborn(self, kind="relplot", seaborn_args={}):
        """
        Use seaborn relplot to visualize individual series of frame (https://seaborn.pydata.org/generated/seaborn.relplot.html#seaborn.relplot)

        Args:
            seaborn_args (dict): {'arg': value} for seaborn.relplot, 'y', 'x', and 'hue' are required.
        Returns:
            raises seaborn relplot
        """
        if not all(map(seaborn_args.keys().__contains__, ["x", "y"])):
            raise KeyError(
                "you must specify at least 'x', 'y', and 'hue' in your seaborn_args dictionary. see documentation"
            )
        else:
            return tools.seaborn.sea(
                df=self.frame, kind=kind, seaborn_args=seaborn_args
            )

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
            elif kind in ["files", "any"]:
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
