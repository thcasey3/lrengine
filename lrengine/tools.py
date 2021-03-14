"""
class for the user defined functions
"""
import os


class tools:
    def __init__(self, input_dict):

        self.user1(input_dict)

    def user1(self, input_dict):

        bolts = os.listdir(input_dict["directory"])

        num = 0
        for bolt in bolts:
            if input_dict["patterns"] in bolt:
                num = num + 1

        print("Pattern found in " + str(num) + " names")

        input_dict["recognized"] = num

        return input_dict
