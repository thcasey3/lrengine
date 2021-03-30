"""
classes for finding dates and patterns
"""

from . import engine
from datetime import date
from dateutil import parser
import re
import numpy as np


class date_injectors:
    """
    injectors class

    Attributes:
        lrdata (start object): Data object from start class
    """

    def __init__(self, lrdata):

        if lrdata.date_format:
            if lrdata.date_format == "any":
                lrdata = self._smart_search_dates(lrdata)
            else:
                lrdata = self._look_for_date(lrdata)

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
                            and found_date.year > 1970
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

        if date_format == "YYYY-MM-DD" or date_format == "YYYY-DD-MM":
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

        if date_format == "DD-MM-YYYY" or date_format == "MM-DD-YYYY":
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

        if (
            date_format == "YYYYMMDD"
            or date_format == "YYYYDDMM"
            or date_format == "DDMMYYYY"
            or date_format == "MMDDYYYY"
        ):
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

        if (
            date_format == "YY-MM-DD"
            or date_format == "YY-DD-MM"
            or date_format == "DD-MM-YY"
            or date_format == "MM-DD-YY"
        ):
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

        if (
            date_format == "YYMMDD"
            or date_format == "YYDDMM"
            or date_format == "DDMMYY"
            or date_format == "MMDDYY"
        ):
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
    injectors class

    Attributes:
        lrdata (start object): Data object from start class
    """

    def __init__(self, lrdata):

        if lrdata.patterns:
            lrdata = self.look_for_patterns(lrdata)

    def look_for_patterns(self, lrdata):

        temp_dict = {}
        for indx in lrdata.patterns:
            temp_dict[indx] = []

        for _, dir in enumerate(lrdata.frame.names):
            for _, patt in enumerate(temp_dict.keys()):
                if patt in dir:
                    temp_dict[patt].append(1)
                else:
                    temp_dict[patt].append(0)

        for _, cols in enumerate(lrdata.frame.columns):
            for _, keys in enumerate(temp_dict.keys()):
                if keys == cols:
                    lrdata.frame[cols] = temp_dict[keys]

        return lrdata
