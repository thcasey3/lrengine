"""
intake module, classes for finding dates and patterns
"""

from . import engine
from datetime import date
from dateutil import parser
import re
import numpy as np


class date_injectors:
    """
    class for looking for dates in file or folder names

    Args:
        lrdata (start): start object

    Returns:
        updated start object
    """

    def __init__(self, lrdata):

        if lrdata.date_format:
            if lrdata.date_format == "any":
                self._smart_search_dates(lrdata)
            else:
                self._look_for_date(lrdata)

    def _smart_search_dates(self, lrdata):

        date_list = list(np.zeros(len(lrdata.frame)))
        date_format_list = list(np.zeros(len(lrdata.frame)))
        date_delta_list = list(np.zeros(len(lrdata.frame)))

        for indx, dir in enumerate(lrdata.frame.names):
            possible_date = []
            possible_delta = []
            possible_patt = []
            for _, patt in enumerate(
                [
                    "YYYYMMDD",
                    "YYYYDDMM",
                    "MMDDYYYY",
                    "DDMMYYYY",
                    "YYYY-MM-DD",
                    "YYYY-DD-MM",
                    "MM-DD-YYYY",
                    "DD-MM-YYYY",
                    "YYMMDD",
                    "YYDDMM",
                    "MMDDYY",
                    "DDMMYY",
                    "YY-MM-DD",
                    "YY-DD-MM",
                    "MM-DD-YY",
                    "DD-MM-YY",
                ]
            ):
                date_try = self.look_for_date_string(dir, patt)
                if date_try:
                    try:
                        found_date = self.parse_dates(date_try)
                        found_delta = self.diff_dates(found_date)
                        if (
                            found_delta >= 0
                            and found_date.year > 1900
                            and found_date.year < date.today().year
                        ):
                            possible_date.append(found_date)
                            possible_delta.append(found_delta)
                            possible_patt.append(patt)
                    except ValueError:
                        continue

            if isinstance(possible_date, list):
                possible_date = list(set(possible_date))
                if len(possible_date) == 1:
                    possible_date = possible_date[0]
            if isinstance(possible_delta, list):
                possible_delta = list(set(possible_delta))
                if len(possible_delta) == 1:
                    possible_delta = possible_delta[0]
            if isinstance(possible_patt, list):
                possible_patt = list(set(possible_patt))
                if len(possible_patt) == 1:
                    possible_patt = possible_patt[0]

            if possible_date:
                date_list[indx] = possible_date
                if date_list[indx]:
                    date_delta_list[indx] = possible_delta
                    date_format_list[indx] = possible_patt

        if not date_delta_list:
            print("No dates were found in names")
        else:
            lrdata.frame["date"] = date_list
            lrdata.frame["date_format"] = date_format_list
            lrdata.frame["date_delta"] = date_delta_list

        return lrdata

    def _look_for_date(self, lrdata):

        date_list = list(np.zeros(len(lrdata.frame)))
        date_delta_list = list(np.zeros(len(lrdata.frame)))

        for indx, dir in enumerate(lrdata.frame.names):
            possible_date = self.look_for_date_string(dir, lrdata.date_format)
            if possible_date:
                date_list[indx] = self.parse_dates(possible_date)
                if date_list[indx]:
                    date_delta_list[indx] = self.diff_dates(date_list[indx])

        if sum(date_delta_list) != 0:
            lrdata.frame["date"] = date_list
            lrdata.frame["date_delta"] = date_delta_list
        else:
            print("No dates of format " + lrdata.date_format + " found in names")

        return lrdata

    @staticmethod
    def look_for_date_string(dir, date_format):

        if date_format in ["YYYY-MM-DD", "YYYY-DD-MM"]:
            date_string = re.findall(
                r"[0-9]{4}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}", dir
            )
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]
                date_string.replace("/", "-").replace("_", "-")

                if date_format == "YYYY-MM-DD":
                    date_string = (
                        date_string[0:4]
                        + "-"
                        + date_string[5:7]
                        + "-"
                        + date_string[8:]
                    )

                if date_format == "YYYY-DD-MM":
                    date_string = (
                        date_string[0:4]
                        + "-"
                        + date_string[8:]
                        + "-"
                        + date_string[5:7]
                    )

        if date_format in ["DD-MM-YYYY", "MM-DD-YYYY"]:
            date_string = re.findall(
                r"[0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{4}", dir
            )
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]
                date_string.replace("/", "-").replace("_", "-")

                if date_format == "DD-MM-YYYY":
                    date_string = (
                        date_string[6:]
                        + "-"
                        + date_string[3:5]
                        + "-"
                        + date_string[0:2]
                    )

                if date_format == "MM-DD-YYYY":
                    date_string = (
                        date_string[6:]
                        + "-"
                        + date_string[0:2]
                        + "-"
                        + date_string[3:5]
                    )

        if date_format in ["YYYYMMDD", "YYYYDDMM", "DDMMYYYY", "MMDDYYYY"]:
            date_string = re.findall(r"[0-9]{8}", dir)
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]

                if date_format == "YYYYMMDD":
                    date_string = date_string[0:4] + date_string[4:6] + date_string[6:]

                if date_format == "YYYYDDMM":
                    date_string = date_string[0:4] + date_string[6:] + date_string[4:6]

                if date_format == "DDMMYYYY":
                    date_string = date_string[4:] + date_string[2:4] + date_string[0:2]

                if date_format == "MMDDYYYY":
                    date_string = date_string[4:] + date_string[0:4]

        if date_format in ["YY-MM-DD", "YY-DD-MM", "DD-MM-YY", "MM-DD-YY"]:
            date_string = re.findall(
                r"[0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}", dir
            )
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]
                date_string.replace("/", "-").replace("_", "-")

                if date_format == "YY-MM-DD":
                    if int("20" + date_string[0:2]) > date.today().year:
                        date_string = (
                            "19"
                            + date_string[0:2]
                            + "-"
                            + date_string[3:5]
                            + "-"
                            + date_string[6:]
                        )
                    else:
                        date_string = (
                            "20"
                            + date_string[0:2]
                            + "-"
                            + date_string[3:5]
                            + "-"
                            + date_string[6:]
                        )

                if date_format == "YY-DD-MM":
                    if int("20" + date_string[0:2]) > date.today().year:
                        date_string = (
                            "19"
                            + date_string[0:2]
                            + "-"
                            + date_string[6:]
                            + "-"
                            + date_string[3:5]
                        )
                    else:
                        date_string = (
                            "20"
                            + date_string[0:2]
                            + "-"
                            + date_string[6:]
                            + "-"
                            + date_string[3:5]
                        )

                if date_format == "DD-MM-YY":
                    if int("20" + date_string[6:]) > date.today().year:
                        date_string = (
                            "19"
                            + date_string[6:]
                            + "-"
                            + date_string[3:5]
                            + "-"
                            + date_string[0:2]
                        )
                    else:
                        date_string = (
                            "20"
                            + date_string[6:]
                            + "-"
                            + date_string[3:5]
                            + "-"
                            + date_string[0:2]
                        )

                if date_format == "MM-DD-YY":
                    if int("20" + date_string[6:]) > date.today().year:
                        date_string = (
                            "19"
                            + date_string[6:]
                            + "-"
                            + date_string[0:2]
                            + "-"
                            + date_string[3:5]
                        )
                    else:
                        date_string = (
                            "20"
                            + date_string[6:]
                            + "-"
                            + date_string[0:2]
                            + "-"
                            + date_string[3:5]
                        )

        if date_format in ["YYMMDD", "YYDDMM", "DDMMYY", "MMDDYY"]:
            date_string = re.findall(r"[0-9]{6}", dir)
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]

                if date_format == "YYMMDD":
                    if int("20" + date_string[0:2]) > date.today().year:
                        date_string = (
                            "19" + date_string[0:2] + date_string[2:4] + date_string[4:]
                        )
                    else:
                        date_string = (
                            "20" + date_string[0:2] + date_string[2:4] + date_string[4:]
                        )

                if date_format == "YYDDMM":
                    if int("20" + date_string[0:2]) > date.today().year:
                        date_string = (
                            "19" + date_string[0:2] + date_string[4:] + date_string[2:4]
                        )
                    else:
                        date_string = (
                            "20" + date_string[0:2] + date_string[4:] + date_string[2:4]
                        )

                if date_format == "DDMMYY":
                    if int("20" + date_string[4:]) > date.today().year:
                        date_string = (
                            "19" + date_string[4:] + date_string[2:4] + date_string[0:2]
                        )
                    else:
                        date_string = (
                            "20" + date_string[4:] + date_string[2:4] + date_string[0:2]
                        )

                if date_format == "MMDDYY":
                    if int("20" + date_string[4:]) > date.today().year:
                        date_string = "19" + date_string[4:] + date_string[0:4]
                    else:
                        date_string = "20" + date_string[4:] + date_string[0:4]

        return date_string

    @staticmethod
    def parse_dates(possible_date):
        return parser.isoparse(possible_date).date()

    @staticmethod
    def diff_dates(possible_date):
        delta = date.today() - date(
            possible_date.year, possible_date.month, possible_date.day
        )
        return int(delta.days)


class pattern_injectors:
    """
    class for looking for patterns in file or folder names

    Args:
        lrdata (start): start object

    Returns:
        updated start object
    """

    def __init__(self, lrdata):

        if lrdata.patterns:
            self._look_for_patterns(lrdata)

    def _look_for_patterns(self, lrdata):

        if isinstance(lrdata.patterns, list):
            patterns_to_iterate = lrdata.patterns
        elif isinstance(lrdata.patterns, dict):
            patterns_to_iterate = lrdata.patterns.keys()

        for patt in patterns_to_iterate:
            lrdata.frame[patt] = np.zeros(len(lrdata.frame))
            for indx, dir in enumerate(lrdata.frame["names"]):
                if isinstance(lrdata.patterns, list):
                    if patt in dir:
                        lrdata.frame.loc[indx, patt] = True
                    else:
                        lrdata.frame.loc[indx, patt] = False

                elif isinstance(lrdata.patterns, dict):
                    if lrdata.patterns[patt] is bool:
                        if patt in dir:
                            lrdata.frame.loc[indx, patt] = True
                        else:
                            lrdata.frame.loc[indx, patt] = False
                    else:
                        found = False
                        value = lrdata.patterns[patt]
                        match = re.findall(patt + ".*", dir)
                        if len(match) != 0:
                            find_value = re.findall(patt + value, match[0])
                            if len(find_value) != 0:
                                found = find_value[0].replace(patt, "")

                        lrdata.frame.loc[indx, patt] = found

        return lrdata


class names_filter:
    """
    class for filtering names based on patterns

    Args:
        lrdata (start): start object
        skip (list): patterns to use to decide which names to skip
        keep (list): patterns to use to decide which names to keep
        inplace (bool): pandas "inplace" option for the .drop method

    Returns:
        updated start object
    """

    def __init__(self, lrdata, skip=None, keep=None, inplace=True):

        self._take_out_names(lrdata, skip=skip, keep=keep, inplace=inplace)

    def _take_out_names(self, lrdata, skip=None, keep=None, inplace=True):

        if skip is not None or keep is not None:
            if isinstance(skip, str):
                skip = [skip]
            if isinstance(keep, str):
                keep = [keep]
            skip_indx = []
            for indx, subdir in enumerate(lrdata.frame["names"]):
                if skip is not None and keep is None:
                    if any(map(subdir.__contains__, skip)):
                        skip_indx.append(lrdata.frame.index[indx])
                elif skip is None and keep is not None:
                    if not any(map(subdir.__contains__, keep)):
                        skip_indx.append(lrdata.frame.index[indx])
                elif skip is not None and keep is not None:
                    if any(map(subdir.__contains__, skip)) or not any(
                        map(subdir.__contains__, keep)
                    ):
                        skip_indx.append(lrdata.frame.index[indx])

            if len(skip_indx) == len(lrdata.frame):
                raise TypeError(
                    "You removed all of your names! Try different skip or keep patterns"
                )
            else:
                lrdata.frame.drop(skip_indx, inplace=inplace)

        return lrdata


class dates_filter:
    """
    class for reducing dates lists from the "all" option

    Args:
        lrdata (start): start object
        format (str): format of date to keep

    Returns:
        updated start object
    """

    def __init__(self, lrdata, format=None):

        self._take_out_dates(lrdata, format=format)

    def _take_out_dates(self, lrdata, format=None):

        if format and "date_format" in lrdata.frame.columns:
            new_formats = list(np.zeros(len(lrdata.frame)))
            new_dates = list(np.zeros(len(lrdata.frame)))
            new_deltas = list(np.zeros(len(lrdata.frame)))
            for indx, form_list in enumerate(lrdata.frame["date_format"]):
                if isinstance(form_list, list):
                    for indx2, forms in enumerate(form_list):
                        if forms == format and isinstance(forms, str):
                            new_formats[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_format"
                            ][indx2]
                            new_dates[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date"
                            ][indx2]
                            new_deltas[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_delta"
                            ][indx2]

                elif isinstance(form_list, str):
                    if form_list == format:
                        new_formats[indx] = lrdata.frame.loc[
                            lrdata.frame.index[indx], "date_format"
                        ]
                        new_dates[indx] = lrdata.frame.loc[
                            lrdata.frame.index[indx], "date"
                        ]
                        new_deltas[indx] = lrdata.frame.loc[
                            lrdata.frame.index[indx], "date_delta"
                        ]

            lrdata.frame["date_format"] = new_formats
            lrdata.frame["date"] = new_dates
            lrdata.frame["date_delta"] = new_deltas

        return lrdata
