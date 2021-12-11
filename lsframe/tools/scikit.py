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

    def __init__(self, lsdata, X_columns=None, y_column=None):

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


def scikit_fit(X_train, y_train, scikit_params, method, base_estimator=False):

    if method in ["AdaBoost", "Bagging"]:
        if base_estimator == "DecisionTree":
            base_est = tree.DecisionTreeRegressor(
                random_state=scikit_params["random_state"],
                criterion=scikit_params["criterion"],
            )
        elif base_estimator == "ExtraTree":
            base_est = tree.ExtraTreeRegressor(
                random_state=scikit_params["random_state"],
                criterion=scikit_params["criterion"],
            )
        elif base_estimator == "RandomForest":
            base_est = ensemble.RandomForestRegressor(
                random_state=scikit_params["random_state"],
                n_estimators=scikit_params["n_estimators"],
                criterion=scikit_params["criterion"],
                max_features=scikit_params["max_features"],
                max_samples=scikit_params["max_samples"],
                n_jobs=scikit_params["n_jobs"],
            )
        else:
            base_est = scikit_params["base_estimator"]

        if method == "AdaBoost":
            obj = ensemble.AdaBoostRegressor(
                random_state=scikit_params["random_state"],
                base_estimator=base_est,
                loss=scikit_params["loss"],
                n_estimators=scikit_params["n_estimators"],
            )
        if method == "Bagging":
            obj = ensemble.BaggingRegressor(
                random_state=scikit_params["random_state"],
                base_estimator=base_est,
                n_estimators=scikit_params["n_estimators"],
                n_jobs=scikit_params["n_jobs"],
            )

    elif method == "DecisionTree":
        obj = tree.DecisionTreeRegressor(
            random_state=scikit_params["random_state"],
            criterion=scikit_params["criterion"],
        )
    elif method == "ExtraTree":
        obj = tree.ExtraTreeRegressor(
            random_state=scikit_params["random_state"],
            criterion=scikit_params["criterion"],
        )
    elif method == "ExtraTrees":
        obj = ensemble.ExtraTreesRegressor(
            random_state=scikit_params["random_state"],
            n_estimators=scikit_params["n_estimators"],
            criterion=scikit_params["criterion"],
            max_features=scikit_params["max_features"],
            max_samples=scikit_params["max_samples"],
            n_jobs=scikit_params["n_jobs"],
        )
    elif method == "GradientBoosting":
        obj = ensemble.GradientBoostingRegressor(
            random_state=scikit_params["random_state"],
            n_estimators=scikit_params["n_estimators"],
            criterion=scikit_params["criterion"],
            loss=scikit_params["loss"],
        )
    elif method == "HistGradientBoosting":
        obj = ensemble.HistGradientBoostingRegressor(
            random_state=scikit_params["random_state"],
            loss=scikit_params["loss"],
            l2_regularization=scikit_params["l2_regularization"],
        )
    elif method == "RandomForest":
        obj = ensemble.RandomForestRegressor(
            random_state=scikit_params["random_state"],
            n_estimators=scikit_params["n_estimators"],
            criterion=scikit_params["criterion"],
            max_features=scikit_params["max_features"],
            max_samples=scikit_params["max_samples"],
            n_jobs=scikit_params["n_jobs"],
        )
    elif method == "Stacking":
        obj = ensemble.StackingRegressor(
            estimators=scikit_params["estimators"],
            final_estimator=scikit_params["final_estimator"],
        )
    elif method == "Voting":
        obj = ensemble.VotingRegressor(estimators=scikit_params["estimators"])
    elif method == "SVR":
        obj = svm.SVR()
    elif method == "MLP":
        obj = neural_network.MLPRegressor(
            random_state=scikit_params["random_state"],
            activation=scikit_params["activation"],
            solver=scikit_params["solver"],
            learning_rate=scikit_params["learning_rate"],
            learning_rate_init=scikit_params["learning_rate_init"],
            power_t=scikit_params["power_t"],
        )
    elif method == "KNeighbors":
        obj = neighbors.KNeighborsRegressor(weights=scikit_params["weights"])
    elif method == "Isotonic":
        obj = isotonic.IsotonicRegression()
    elif method == "GaussianProcess":
        obj = gaussian_process.GaussianProcessRegressor(
            random_state=scikit_params["random_state"]
        )
    elif method == "PLS":
        obj = cross_decomposition.PLSRegression()

    obj = obj.fit(X_train, y_train)

    return obj
