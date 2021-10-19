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
        encode (list): non-numeric dtype columns to encode
        dtypes (dict): desired data types for each column
    Returns:
        X (DataFrame): X for inserting in train_test_split
        y (DataFrame): y for inserting in train_test_split
    """

    def __init__(self, lsdata, X_columns=None, y_columns=None, dtypes="native", encode=None, decode=False):
        
        if encode is not None:
            self.numberhot(lsdata, encode=encode)
        elif decode:
            self.unhot(lsdata, orig_values=False)
            
        if dtypes != "native":
            self.set_dtypes(lsdata, dtypes=dtypes)
            
        if X_columns is not None and y_columns is not None:
            self.Xy_split(lsdata, X_columns=X_columns, y_columns=y_columns)
        
    def Xy_split(self, lsdata, X_columns=None, y_columns=None):

        lsdata.X = lsdata.frame[X_columns]
        lsdata.y = lsdata.frame[y_columns]

    def numberhot(self, lsdata, encode="all"):
    
        if encode == "all":
            columns = lsdata.frame.columns.values
        elif isinstance(encode, list):
            columns = encode
        else:
            raise ValueError("encode must be 'all' or a list of column names")
            
        lsdata.original_frame = lsdata.frame.copy()
        for col in lsdata.frame.columns.values:
            numhot_df = lsdata.frame.drop_duplicates([col])[[col]]
            temp_col_name = "_".join([col, "numhot"])
            numhot_df[temp_col_name] = list(range(len(numhot_df)))
            for indx in numhot_df.index:
                lsdata.frame[col].replace(numhot_df.loc[indx, col], numhot_df.loc[indx, temp_col_name], inplace=True)
    
    def unhot(self, lsdata):
    
        lsdata.frame = lsdata.original_frame
        delattr(lsdata, "original_frame")
    
    def set_dtypes(self, lsdata, dtypes="native"):
        
        new_df = {}
        for x in lsdata.frame.columns.values:
            if x in dtypes.keys():
                if dtypes[x] == "str":
                    new_df[x] = [str(lsdata.frame.loc[indx, x]) for indx in lsdata.frame.index]
                if dtypes[x] == "int":
                    new_df[x] = [int(lsdata.frame.loc[indx, x]) for indx in lsdata.frame.index]
                if dtypes[x] == "float":
                    new_df[x] = [str(lsdata.frame.loc[indx, x]) for indx in lsdata.frame.index]
                if dtypes[x] == "datetime":
                    new_df[x] = [parser.parse(lsdata.frame.loc[indx, x]) for indx in lsdata.frame.index]
        
        lsdata.frame = pd.DataFrame(new_df)
