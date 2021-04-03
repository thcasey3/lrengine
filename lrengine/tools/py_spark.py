"""
py_spark module, for interacting with PySpark
"""
import pyspark as ps


class spark:
    """
    class for interacting with PySpark

    Args:
        df (DataFrame): "frame" from start object
        args (dict): dict
    Returns:
        PySpark result
    """

    def __init__(self, df=None, args):

        self.spark_object(df=df, args)

    def spark_object(self, df, args):
        return args
