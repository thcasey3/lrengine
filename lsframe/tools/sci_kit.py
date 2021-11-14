"""
sci_kit module, for preparing X and y from a lsobject.frame
"""
from dateutil import parser
import pandas as pd


class scikit:
    """
    class for preparing sklearn X and y from lsobject.frame

    Args:
        X_columns (list): columns from frame to include in X
        y_column (list): columns from frame to include in y
    Returns:
        lsdata (object): lsdata object with X, y, and sciframe added
    """

    def __init__(
        self,
        lsdata,
        X_columns=None,
        y_column=None
    ):

        if X_columns is not None or y_column is not None:
            self.Xy_split(lsdata, X_columns=X_columns, y_column=y_column)
        
        lsdata.sciframe = lsdata.frame[X_columns + y_column]

    def Xy_split(self, lsdata, X_columns=None, y_column=None):

        if X_columns is not None and y_column is not None:
            lsdata.X = lsdata.sciframe[X_columns]
            lsdata.y = lsdata.sciframe[y_column]
        elif X_columns is None and y_column is not None:
            lsdata.X = lsdata.sciframe[lsdata.sciframe.columns.values - y_column]
            lsdata.y = lsdata.sciframe[y_column]
        elif X_columns is not None and y_column is None:
            lsdata.X = lsdata.sciframe[X_columns]
            lsdata.y = lsdata.sciframe[lsdata.sciframe.columns.values - X_columns]
