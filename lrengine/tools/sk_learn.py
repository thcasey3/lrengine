"""
class for interacting with sklearn
"""


class learn:
    """
    learn class

    Attributes:
        df (DataFrame): "frame" from start object
        kind (str): type of model
        options (dict): dictionary of sklearn options
    """

    def __init__(self, df, kind, options):

        if kind != "KNN":
            raise ValueError("only KNN is currently supported")
        else:
            from sklearn import preprocessing
            from sklearn import metrics
            from sklearn.model_selection import train_test_split
            from sklearn.neighbors import KNeighborsClassifier

            self.frame = df
            self.x_columns = options["x_columns"]
            self.y_columns = options["y_columns"]
            self.test_size = options["test_size"]
            self.random_state = options["random_state"]
            self.k = options["k"]

            self.prepare_to_lrn(options)

    def prepare_to_lrn(self, options):

        x = self.frame[options["x_columns"]].values
        y = self.frame[options["y_columns"]].values

        xset = preprocessing.StandardScaler().fit(x).transform(x.astype(float))
        yset = preprocessing.StandardScaler().fit(y).transform(y.astype(float))

        xtrain, xtest, ytrain, ytest = train_test_split(
            xset, yset, test_size=self.test_size, random_state=self.random_state
        )

        self.neighbors = self.train_lrn(xtrain, ytrain, options)
        self.predictions = self.test_lrn(self.neighbors, test)
        self.metrics = self.metrics_lrn(test_column, test_results)

    def train_lrn(self, xtrain, ytrain, options):
        return KNeighborsClassifier(n_neighbors=options["k"]).fit(xtrain, ytrain)

    def test_lrn(self, neighbors, test):
        return neighbors.predict(test)

    def metrics_lrn(self, test_column, test_results):
        return metrics.accuracy_score(test_column, test_results)
