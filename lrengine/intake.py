"""
class for checking and sending the data object to engine
"""

from . import engine
from datetime import date
from dateutil import parser
import re
import numpy as np


class injectors:
    """
    injectors class

    Attributes:
        lrdata (start object): Data object from start class
    """

    def __init__(self, lrdata):

        if "date_format" in lrdata.keys() and lrdata["date_format"]:
            lrdata = self.look_for_dates(lrdata)

        if "patterns" in lrdata.keys() and lrdata["patterns"]:
            lrdata = self.look_for_patterns(lrdata)

    def look_for_dates(self, lrdata):

        date_list = list(np.zeros(len(lrdata["frame"])))
        date_delta_list = list(np.zeros(len(lrdata["frame"])))

        for indx, dir in enumerate(lrdata["frame"]["names"]):
            possible_date = None
            if lrdata["date_format"] == "YYYYMMDD":
                possible_date = self.look_for_YYYYMMDD(dir)
            if lrdata["date_format"] == "YYYY_MM_DD":
                possible_date = self.look_for_YYYY_MM_DD(dir)

            if possible_date:
                date_list[indx] = self.parse_dates(possible_date[0])
                if date_list[indx]:
                    date_delta_list[indx] = self.diff_dates(date_list[indx])

        if sum(date_delta_list) != 0:
            lrdata["frame"]["date"] = date_list
            lrdata["frame"]["date_delta"] = date_delta_list
        else:
            print("No dates of format " + lrdata["date_format"] + " found in names")

        return lrdata

    @staticmethod
    def look_for_YYYY_MM_DD(dir):
        return re.findall(r"[0-9]{4}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}", dir)

    @staticmethod
    def look_for_YYYYMMDD(dir):
        return re.findall(r"[0-9]{8}", dir)

    @staticmethod
    def parse_dates(possible_date):
        return parser.isoparse(possible_date).date()

    @staticmethod
    def diff_dates(possible_date):
        delta = date.today() - date(
            possible_date.year, possible_date.month, possible_date.day
        )
        return int(delta.days)

    def look_for_patterns(self, lrdata):

        temp_dict = {}
        for indx in lrdata["patterns"]:
            temp_dict[indx] = []

        for _, dir in enumerate(lrdata["frame"]["names"]):
            for _, patt in enumerate(temp_dict.keys()):
                if patt in dir:
                    temp_dict[patt].append(1)
                else:
                    temp_dict[patt].append(0)

        for _, cols in enumerate(lrdata["frame"].columns):
            for _, keys in enumerate(temp_dict.keys()):
                if keys == cols:
                    lrdata["frame"][cols] = temp_dict[keys]

        return lrdata
