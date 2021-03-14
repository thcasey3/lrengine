"""
class for distributing the lrdata object attributes
"""

from . import engine


class ignition:
    def __init__(self, input_df):

        engine.run(input_df)
