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
            if lrdata.date_format == "any" or isinstance(lrdata.date_format, list):
                self._smart_search_dates(lrdata)
            elif isinstance(lrdata.date_format, str):
                self._look_for_date(lrdata)

    def _smart_search_dates(self, lrdata):

        date_list = list(np.zeros(len(lrdata.frame)))
        date_format_list = list(np.zeros(len(lrdata.frame)))
        date_delta_list = list(np.zeros(len(lrdata.frame)))

        if lrdata.date_format == "any":
            possible_formats = [
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
                "YYYY:MM:DD",
                "YYYY:DD:MM",
                "MM:DD:YYYY",
                "DD:MM:YYYY",
                "YY:MM:DD",
                "YY:DD:MM",
                "MM:DD:YY",
                "DD:MM:YY",
            ]
        elif isinstance(lrdata.date_format, list):
            possible_formats = lrdata.date_format

        for indx, dir in enumerate(lrdata.frame.names):
            possible_date = []
            possible_delta = []
            possible_patt = []
            for _, patt in enumerate(possible_formats):
                date_try = self._look_for_date_string(dir, patt)
                if date_try:
                    found_date = self.parse_dates(date_try)
                    if found_date is not None:
                        found_delta = self.diff_dates(found_date)
                        if (
                            found_delta >= 0
                            and found_date.year > 1900
                            and found_date.year < date.today().year
                        ):
                            possible_date.append(found_date)
                            possible_delta.append(found_delta)
                            possible_patt.append(patt)
                    else:
                        continue

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

        date_list = [0 for _ in range(len(lrdata.frame))]
        date_delta_list = [0 for _ in range(len(lrdata.frame))]

        for indx, dir in enumerate(lrdata.frame.names):
            possible_date = self._look_for_date_string(dir, lrdata.date_format)
            if possible_date:
                date_list[indx] = self.parse_dates(possible_date)
                if date_list[indx] is not None:
                    if date_list[indx]:
                        date_delta_list[indx] = self.diff_dates(date_list[indx])
                else:
                    continue

        if sum(date_delta_list) != 0:
            lrdata.frame["date"] = date_list
            lrdata.frame["date_delta"] = date_delta_list
        else:
            print("No dates of format " + lrdata.date_format + " found in names")

        return lrdata

    def _look_for_date_string(self, dir, date_format):

        if date_format in [
            "YYYY-MM-DD",
            "YYYY-DD-MM",
            "YYYY/MM/DD",
            "YYYY/DD/MM",
            "YYYY_MM_DD",
            "YYYY_DD_MM",
            "YYYY:MM:DD",
            "YYYY:DD:MM",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{4}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}))", dir
            )
        elif date_format in [
            "DD-MM-YYYY",
            "MM-DD-YYYY",
            "DD/MM/YYYY",
            "MM/DD/YYYY",
            "DD_MM_YYYY",
            "MM_DD_YYYY",
            "DD:MM:YYYY",
            "MM:DD:YYYY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{4}))", dir
            )
        elif date_format in [
            "YY-MM-DD",
            "YY-DD-MM",
            "DD-MM-YY",
            "MM-DD-YY",
            "YY/MM/DD",
            "YY/DD/MM",
            "DD/MM/YY",
            "MM/DD/YY",
            "YY_MM_DD",
            "YY_DD_MM",
            "DD_MM_YY",
            "MM_DD_YY",
            "YY:MM:DD",
            "YY:DD:MM",
            "DD:MM:YY",
            "MM:DD:YY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}))", dir
            )
        elif date_format in ["YYYYMMDD", "YYYYDDMM", "DDMMYYYY", "MMDDYYYY"]:
            matches = re.finditer(r"(?=(\d{8}))", dir)
        elif date_format in ["YYMMDD", "YYDDMM", "DDMMYY", "MMDDYY"]:
            matches = re.finditer(r"(?=(\d{6}))", dir)

        date_string = [match.group(1) for match in matches]

        if date_string and isinstance(date_string, list):
            new_date_string = []
            for ds in date_string:
                ds.replace("/", "-").replace("_", "-").replace(":", "-")
                test_date_string = self._map_date_string(ds, date_format)
                test = self.parse_dates(test_date_string)
                if test is not None:
                    new_date_string.append(test_date_string)
                else:
                    continue
            if new_date_string and len(new_date_string) == 1:
                date_string = new_date_string[0]
            elif new_date_string and len(new_date_string) > 1:
                date_string = [x for x in new_date_string if x != []]

        return date_string

    def _map_date_string(self, date_string, date_format):

        if date_format == "YYYY-MM-DD":
            date_string = (
                date_string[0:4] + "-" + date_string[5:7] + "-" + date_string[8:]
            )

        if date_format == "YYYY-DD-MM":
            date_string = (
                date_string[0:4] + "-" + date_string[8:] + "-" + date_string[5:7]
            )

        if date_format == "DD-MM-YYYY":
            date_string = (
                date_string[6:] + "-" + date_string[3:5] + "-" + date_string[0:2]
            )

        if date_format == "MM-DD-YYYY":
            date_string = (
                date_string[6:] + "-" + date_string[0:2] + "-" + date_string[3:5]
            )

        if date_format == "YYYYMMDD":
            date_string = date_string[0:4] + date_string[4:6] + date_string[6:]

        if date_format == "YYYYDDMM":
            date_string = date_string[0:4] + date_string[6:] + date_string[4:6]

        if date_format == "DDMMYYYY":
            date_string = date_string[4:] + date_string[2:4] + date_string[0:2]

        if date_format == "MMDDYYYY":
            date_string = date_string[4:] + date_string[0:4]

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
        try:
            parsed = parser.isoparse(possible_date).date()
            return parsed
        except:
            return None

    @staticmethod
    def diff_dates(possible_date):
        delta = date.today() - date(
            possible_date.year, possible_date.month, possible_date.day
        )
        return int(delta.days)


class dates_filter:
    """
    class for reducing dates lists from the "all" option

    Args:
        lrdata (start): start object
        remove (str or list): format(s) of dates to remove
        keep (str or list): format(s) of dates to keep

    Returns:
        updated start object
    """

    def __init__(self, lrdata, remove=None, keep=None):

        if remove is not None and keep is not None:
            raise ValueError("please give either remove or keep, do not give both.")

        if isinstance(remove, str):
            remove = [remove]
        if isinstance(keep, str):
            keep = [keep]

        self._take_out_dates(lrdata, remove=remove, keep=keep)

    def _take_out_dates(self, lrdata, remove, keep):

        if (keep or remove) and "date_format" in lrdata.frame.columns:
            new_formats = [0 for _ in range(len(lrdata.frame))]
            new_dates = [0 for _ in range(len(lrdata.frame))]
            new_deltas = [0 for _ in range(len(lrdata.frame))]
            for indx, form_list in enumerate(lrdata.frame["date_format"]):
                if isinstance(form_list, list):
                    if remove is not None and keep is None:
                        new_formats[indx] = [
                            x
                            for x in lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_format"
                            ]
                            if x not in remove
                        ]
                        new_dates[indx] = [
                            lrdata.frame.loc[lrdata.frame.index[indx], "date"][x]
                            for x, form in enumerate(
                                lrdata.frame.loc[
                                    lrdata.frame.index[indx], "date_format"
                                ]
                            )
                            if form not in remove
                        ]
                        new_deltas[indx] = [
                            lrdata.frame.loc[lrdata.frame.index[indx], "date_delta"][x]
                            for x, form in enumerate(
                                lrdata.frame.loc[
                                    lrdata.frame.index[indx], "date_format"
                                ]
                            )
                            if form not in remove
                        ]
                    elif remove is None and keep is not None:
                        new_formats[indx] = [
                            x
                            for x in lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_format"
                            ]
                            if x in keep
                        ]
                        new_dates[indx] = [
                            lrdata.frame.loc[lrdata.frame.index[indx], "date"][x]
                            for x, form in enumerate(
                                lrdata.frame.loc[
                                    lrdata.frame.index[indx], "date_format"
                                ]
                            )
                            if form in keep
                        ]
                        new_deltas[indx] = [
                            lrdata.frame.loc[lrdata.frame.index[indx], "date_delta"][x]
                            for x, form in enumerate(
                                lrdata.frame.loc[
                                    lrdata.frame.index[indx], "date_format"
                                ]
                            )
                            if form in keep
                        ]
                elif isinstance(form_list, str):
                    if remove is not None and keep is None:
                        if form_list not in remove:
                            new_formats[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_format"
                            ]
                            new_dates[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date"
                            ]
                            new_deltas[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_delta"
                            ]
                    elif remove is None and keep is not None:
                        if form_list in keep:
                            new_formats[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_format"
                            ]
                            new_dates[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date"
                            ]
                            new_deltas[indx] = lrdata.frame.loc[
                                lrdata.frame.index[indx], "date_delta"
                            ]

                if not new_formats[indx]:
                    new_formats[indx] = 0
                elif (
                    isinstance(new_formats[indx], list) and len(new_formats[indx]) == 1
                ):
                    new_formats[indx] = new_formats[indx][0]

                if not new_dates[indx]:
                    new_dates[indx] = 0
                elif isinstance(new_dates[indx], list) and len(new_dates[indx]) == 1:
                    new_dates[indx] = new_dates[indx][0]

                if not new_deltas[indx]:
                    new_deltas[indx] = 0
                elif isinstance(new_deltas[indx], list) and len(new_deltas[indx]) == 1:
                    new_deltas[indx] = new_deltas[indx][0]

            lrdata.frame["date_format"] = new_formats
            lrdata.frame["date"] = new_dates
            lrdata.frame["date_delta"] = new_deltas

        return lrdata


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
            lrdata.frame[patt] = [False for _ in range(len(lrdata.frame))]
            for indx, dir in enumerate(lrdata.frame["names"]):
                if isinstance(lrdata.patterns, list):
                    if patt in dir:
                        lrdata.frame.loc[indx, patt] = True

                elif isinstance(lrdata.patterns, dict):
                    if lrdata.patterns[patt] is bool:
                        if patt in dir:
                            lrdata.frame.loc[indx, patt] = True
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


class patterns_filter:
    """
    class for filtering names based on patterns

    Args:
        lrdata (start): start object
        remove (list): patterns to use to decide which names to remove
        keep (list): patterns to use to decide which names to keep
        inplace (bool): pandas "inplace" option for the .drop method

    Returns:
        updated start object
    """

    def __init__(self, lrdata, remove=None, keep=None, inplace=True):

        self._take_out_names(lrdata, remove=remove, keep=keep, inplace=inplace)

    def _take_out_names(self, lrdata, remove, keep, inplace):

        if remove is not None or keep is not None:
            if isinstance(remove, str):
                remove = [remove]
            if isinstance(keep, str):
                keep = [keep]
            remove_indx = []
            for indx, subdir in enumerate(lrdata.frame["names"]):
                if remove is not None and keep is None:
                    if any(map(subdir.__contains__, remove)):
                        remove_indx.append(lrdata.frame.index[indx])
                elif remove is None and keep is not None:
                    if not any(map(subdir.__contains__, keep)):
                        remove_indx.append(lrdata.frame.index[indx])
                elif remove is not None and keep is not None:
                    if any(map(subdir.__contains__, remove)) or not any(
                        map(subdir.__contains__, keep)
                    ):
                        remove_indx.append(lrdata.frame.index[indx])

            if len(remove_indx) == len(lrdata.frame):
                raise TypeError(
                    "You removed all of your names! Try different remove or keep patterns"
                )
            else:
                lrdata.frame.drop(remove_indx, inplace=inplace)

        return lrdata
