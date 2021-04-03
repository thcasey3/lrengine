"""
sq_lite module, for interacting with sqlite3
"""
import pandas as pd


class sql:
    """
    class for interacting with sqlite3

    Args:
        df (DataFrame): "frame" from start object
        args (dict): dict
    Returns:
        sqlite3 result
    """

    def __init__(self, df=None, args):

        self.sql_table(df=df, args)

    def sql_table(self, df, args):
        return args
