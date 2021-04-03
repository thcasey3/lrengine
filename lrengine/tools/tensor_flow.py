"""
tensor_flow module, for interacting with tensorflow
"""
import tensorflow as tf


class flow:
    """
    class for interacting with tensorflow

    Args:
        df (DataFrame): "frame" from start object
        args (dict): dict
    Returns:
        tensorflow result
    """

    def __init__(self, df=None, kind=None, args):

        if kind == "modelfit":
            self.model_fit(df=df, args)

    def model_fit(self, df, args):
        return args
