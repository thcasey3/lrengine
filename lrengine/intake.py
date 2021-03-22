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
        start object (start object): Data object from start class
    """

    def __init__(self, lrdata):

        lrdata = self.look_for_dates(lrdata)

        if lrdata["patterns"]:
            lrdata = self.look_for_patterns(lrdata)

        engine.cylinders(lrdata)

    def look_for_dates(self, lrdata):

        date = list(np.zeros(len(lrdata["frame"])))
        date_delta = list(np.zeros(len(lrdata["frame"])))

        for indx, dir in enumerate(lrdata["frame"]["Names"]):
            sub_patterns = re.split("[^a-zA-Z0-9 \n\.]", dir)
            if len(sub_patterns) > 1:
                for possible_date in sub_patterns:
                    if len(possible_date) == 8:
                        try:
                            date[indx] = parser.isoparse(possible_date).date()
                            date_delta[indx] = self.diff_dates(date[indx])
                        except ValueError:
                            continue
            elif isinstance(sub_patterns, str) and len(sub_patterns) == 8:
                try:
                    date[indx] = parser.isoparse(sub_patterns).date()
                    date_delta[indx] = self.diff_dates(date[indx])
                except ValueError:
                    continue
            elif isinstance(sub_patterns, list) and len(sub_patterns) == 1:
                if isinstance(sub_patterns[0], str) and len(sub_patterns[0]) == 8:
                    try:
                        date[indx] = parser.isoparse(sub_patterns[0]).date()
                        date_delta[indx] = self.diff_dates(date[indx])
                    except ValueError:
                        continue

        if date:
            lrdata["frame"]["date"] = date
            lrdata["frame"]["date_delta"] = date_delta
        else:
            print("No dates found in Names")

        return lrdata

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

        for _, dir in enumerate(lrdata["frame"]["Names"]):
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
