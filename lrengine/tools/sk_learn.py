"""
sk_learn module, for interacting with scikit-learn
"""
import sklearn as sk


class skl:
    """
    class for interacting with scikit-learn

    Args:
        df (DataFrame): "frame" from start object
        args (dict): dict
    Returns:
        sklearn result
    """

    def __init__(self, df=None, kind=None, args):

        if kind == "kmeans":
            self.kmeans(df=df, args)

    def kmeans(self, df, args):
        return args
