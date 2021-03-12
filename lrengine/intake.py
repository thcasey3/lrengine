"""
class for sending the lrdata object attriburtes to sparks
"""

import pandas as pd
from . import sparks

class injectors:
    
    def __init__(self, input_dict):
        
        self.df = pd.DataFrame(input_dict)

        sparks(self.df)



