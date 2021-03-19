"""
class for sending the lrdata object attriburtes to sparks
"""
from . import tools
import os
import time
import re
import pandas as pd
import numpy as np


class injectors:
    def __init__(self, input_dict):

        self.dir_index = os.listdir(input_dict["directory"])

        input_dict["data_frame"] = pd.DataFrame({"Name": self.dir_index})

        # self.look_for_times(input_dict)
        input_dict = self.look_for_patterns(input_dict)

        tools.tools(input_dict)

    def look_for_times(self, input_dict):

        print(time.time())

    @staticmethod
    def sum_the_dates(possible_date):
        return (
            int(possible_date[0:4]) + int(possible_date[4:6]) + int(possible_date[6:8])
        )

    def look_for_patterns(self, input_dict):

        dirs = os.listdir(input_dict["directory"])

        date = list(np.zeros(len(input_dict["data_frame"])))
        date_sum = list(np.zeros(len(input_dict["data_frame"])))

        for indx, dir in enumerate(dirs):
            sub_pat1 = re.sub("[^a-zA-Z0-9 \n\.]", "", dir)
            sub_pat2 = re.split("[a-zA-Z]+", sub_pat1)
            if len(sub_pat2) > 1:
                for _, possible_date in enumerate(sub_pat2):
                    if len(possible_date) == 8:
                        date[indx] = possible_date
                        date_sum[indx] = self.sum_the_dates(possible_date)
            elif isinstance(sub_pat2, str) and len(sub_pat2) == 8:
                date[indx] = sub_pat2
                date_sum[indx] = self.sum_the_dates(sub_pat2)
            elif isinstance(sub_pat2, list) and len(sub_pat2) == 1:
                if isinstance(sub_pat2[0], str) and len(sub_pat2[0]) == 8:
                    date[indx] = sub_pat2[0]
                    date_sum[indx] = self.sum_the_dates(sub_pat2[0])
            else:
                print("NOTHING")

        if date and len(date) == len(input_dict["data_frame"]):
            input_dict["data_frame"]["date"] = date

        if date_sum and len(date_sum) == len(input_dict["data_frame"]):
            input_dict["data_frame"]["date_sum"] = date_sum

        return input_dict
