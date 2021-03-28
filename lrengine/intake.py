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
            sub_patterns = re.split("[^a-zA-Z0-9 \n\.]", dir)
            for possible_date in sub_patterns:
                if possible_date.isdigit():
                    if len(possible_date) == 8:
                        if lrdata["date_format"] == "YYYYMMDD":
                            pass
                        elif lrdata["date_format"] == "DDMMYYYY":
                            possible_date = (
                                possible_date[5:8]
                                + possible_date[3:5]
                                + possible_date[0:3]
                            )
                        elif lrdata["date_format"] == "MMDDYYYY":
                            possible_date = possible_date[5:8] + possible_date[0:5]

                    if len(possible_date) == 6:
                        if lrdata["date_format"] == "MMDDYY":
                            d = possible_date[5:6] + possible_date[0:5]
                            if int(possible_date[5:6]) > 0 and int(
                                possible_date[5:6]
                            ) <= (date.today().year - 2000):
                                possible_date = "20" + d
                            else:
                                possible_date = "19" + d
                        elif lrdata["date_format"] == "DDMMYY":
                            d = (
                                possible_date[5:6]
                                + possible_date[3:5]
                                + possible_date[0:3]
                            )
                            if int(possible_date[5:6]) > 0 and int(
                                possible_date[5:6]
                            ) <= (date.today().year - 2000):
                                possible_date = "20" + d
                            else:
                                possible_date = "19" + d

                    if (
                        (
                            int(possible_date[0:4]) > 1900
                            and int(possible_date[0:4]) <= date.today().year
                        )
                        and (
                            int(possible_date[4:6]) > 0
                            and int(possible_date[4:6]) <= 12
                        )
                        and (
                            int(possible_date[6:8]) > 0
                            and int(possible_date[6:8]) <= 31
                        )
                    ):
                        date_list[indx] = self.parse_dates(possible_date)
                        date_delta_list[indx] = self.diff_dates(date_list[indx])

        if sum(date_delta_list) != 0:
            lrdata["frame"]["date"] = date_list
            lrdata["frame"]["date_delta"] = date_delta_list
        else:
            print("No dates of format " + lrdata["date_format"] + " found in names")

        return lrdata

    @staticmethod
    def diff_dates(possible_date):
        delta = date.today() - date(
            possible_date.year, possible_date.month, possible_date.day
        )
        return int(delta.days)

    @staticmethod
    def parse_dates(possible_date):
        return parser.isoparse(possible_date).date()

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
