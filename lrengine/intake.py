"""
class for sending the lrdata object attriburtes to sparks
"""

from . import tools
import datetime
import dateutil
import re
import numpy as np


class injectors:
    def __init__(self, input_dict):

        input_dict = self.look_for_dates(input_dict)
        input_dict = self.look_for_patterns(input_dict)

        tools_obj = tools.tools(input_dict)

        input_dict["frame"][tools_obj.col_name] = tools_obj.measure

    def look_for_dates(self, input_dict):

        date = list(np.zeros(len(input_dict["frame"])))
        date_delta = list(np.zeros(len(input_dict["frame"])))

        for indx, dir in enumerate(input_dict["frame"]["Names"]):
            sub_patterns = re.split("[^a-zA-Z0-9 \n\.]", dir)
            if len(sub_patterns) > 1:
                for possible_date in sub_patterns:
                    if len(possible_date) == 8:
                        try:
                            date[indx] = dateutil.parser.isoparse(possible_date)
                            date_delta[indx] = self.sum_the_dates(date[indx])
                        except ValueError:
                            continue
            elif isinstance(sub_patterns, str) and len(sub_patterns) == 8:
                try:
                    date[indx] = dateutil.parser.isoparse(sub_patterns)
                    date_delta[indx] = self.sum_the_dates(date[indx])
                except ValueError:
                    continue
            elif isinstance(sub_patterns, list) and len(sub_patterns) == 1:
                if isinstance(sub_patterns[0], str) and len(sub_patterns[0]) == 8:
                    try:
                        date[indx] = dateutil.parser.isoparse(sub_patterns[0])
                        date_delta[indx] = self.sum_the_dates(date[indx])
                    except ValueError:
                        continue
            else:
                print("NOTHING")

        input_dict["frame"]["date"] = date
        input_dict["frame"]["date_delta"] = date_delta

        return input_dict

    @staticmethod
    def sum_the_dates(possible_date):
        delta = datetime.date(
            possible_date.year, possible_date.month, possible_date.day
        ) - datetime.date(1900, 1, 1)
        return int(delta.days)

    def look_for_patterns(self, input_dict):

        temp_dict = {}
        for indx in input_dict["patterns"]:
            temp_dict[indx] = []

        for _, dir in enumerate(input_dict["frame"]["Names"]):
            for _, patt in enumerate(temp_dict.keys()):
                if patt in dir:
                    temp_dict[patt].append(1)
                else:
                    temp_dict[patt].append(0)

        for _, cols in enumerate(input_dict["frame"].columns):
            for _, keys in enumerate(temp_dict.keys()):
                if keys == cols:
                    input_dict["frame"][cols] = temp_dict[keys]

        return input_dict
