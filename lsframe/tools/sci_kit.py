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
        y_columns (list): columns from frame to include in y
        dtypes (dict): desired data types for each column
        encode (list): non-numeric dtype columns to encode
    Returns:
        lsdata (object): lsdata object with X, y, and sciframe added
    """

    def __init__(
        self,
        lsdata,
        X_columns=None,
        y_columns=None,
        dtypes="native",
        encode="all",
    ):

        lsdata.sciframe = lsdata.frame.copy()
        
        if encode is not None:
            self.numberhot(lsdata, encode=encode)

        if dtypes != "native":
            self.set_dtypes(lsdata, dtypes=dtypes)

        if X_columns is not None or y_columns is not None:
            self.Xy_split(lsdata, X_columns=X_columns, y_columns=y_columns)
            

    def Xy_split(self, lsdata, X_columns=None, y_columns=None):

        if X_columns is not None and y_columns is not None:
            lsdata.X = lsdata.sciframe[X_columns]
            lsdata.y = lsdata.sciframe[y_columns]
        elif X_columns is None and y_columns is not None:
            lsdata.X = lsdata.sciframe[lsdata.sciframe.columns.values - y_columns]
            lsdata.y = lsdata.sciframe[y_columns]
        elif X_columns is not None and y_columns is None:
            lsdata.X = lsdata.sciframe[X_columns]
            lsdata.y = lsdata.sciframe[lsdata.sciframe.columns.values - X_columns]


    def numberhot(self, lsdata, encode="all"):

        if encode == "all":
            columns = lsdata.sciframe.columns.values
        elif isinstance(encode, list):
            columns = encode
        else:
            raise ValueError("encode must be 'all' or a list of column names")

        for col in columns:
            numhot_df = lsdata.sciframe.drop_duplicates([col])[[col]]
            temp_col_name = "_".join([col, "numhot"])
            numhot_df[temp_col_name] = list(range(len(numhot_df)))
            for indx in numhot_df.index:
                lsdata.sciframe[col].replace(
                    numhot_df.loc[indx, col],
                    numhot_df.loc[indx, temp_col_name],
                    inplace=True,
                )


    def set_dtypes(self, lsdata, dtypes={}):

        new_df = {}
        for x in lsdata.sciframe.columns.values:
            if x in dtypes.keys():
                if dtypes[x] == "str":
                    new_df[x] = [
                        str(lsdata.sciframe.loc[indx, x])
                        for indx in lsdata.sciframe.index
                    ]
                if dtypes[x] == "int":
                    new_df[x] = [
                        int(lsdata.sciframe.loc[indx, x])
                        for indx in lsdata.sciframe.index
                    ]
                if dtypes[x] == "float":
                    new_df[x] = [
                        str(lsdata.sciframe.loc[indx, x])
                        for indx in lsdata.sciframe.index
                    ]
                if dtypes[x] == "datetime":
                    new_df[x] = [
                        parser.parse(lsdata.sciframe.loc[indx, x])
                        for indx in lsdata.sciframe.index
                    ]
            else:
                new_df[x] = lsdata.sciframe[x].to_list()

        lsdata.sciframe = pd.DataFrame(new_df)
