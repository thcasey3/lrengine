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
            lrdata = self.look_for_date(lrdata)

    def look_for_date(self, lrdata):

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

            if date_format == "YYYY-DD-MM":
                date_string = (
                    date_string[0:5] + "-" + +date_string[8:] + "-" + +date_string[5:7]
                )

        if date_format == "DD-MM-YYYY" or date_format == "MM-DD-YYYY":
            date_string = re.findall(
                r"[0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{4}", dir
            )
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]

            if date_format == "DD-MM-YYYY":
                date_string = (
                    date_string[6:] + "-" + +date_string[3:5] + "-" + +date_string[0:3]
                )

            if date_format == "MM-DD-YYYY":
                date_string = (
                    date_string[6:] + "-" + +date_string[0:3] + "-" + +date_string[3:5]
                )

        if (
            date_format == "YYYYMMDD"
            or date_format == "DDMMYYYY"
            or date_format == "MMDDYYYY"
            or date_format == "YYYYDDMM"
        ):
            date_string = re.findall(r"[0-9]{8}", dir)
            if date_string and isinstance(date_string, list):
                date_string = date_string[0]

            if date_format == "DDMMYYYY":
                date_string = date_string[5:] + date_string[3:5] + date_string[0:3]
            if date_format == "MMDDYYYY":
                date_string = date_string[5:] + date_string[0:5]
            if date_format == "YYYYDDMM":
                date_string = date_string[0:5] + date_string[7:] + date_string[5:7]

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
