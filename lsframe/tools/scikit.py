"""
scikit module, for using scikit-learn with lsobject.frame
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn import (
    metrics,
    tree,
    ensemble,
    svm,
    neural_network,
    neighbors,
    isotonic,
    gaussian_process,
    cross_decomposition,
    covariance,
    cluster,
    multioutput,
    linear_model,
    kernel_ridge,
    semi_supervised,
    multiclass,
    naive_bayes,
)

_scikit_params = {
    "base_estimator": None,
    "estimator": None,
    "estimators": None,
    "final_estimator": None,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": False,
}


class scikit:
    """
    class for using scikit-learn with lsobject.frame

    Args:
        X_columns (list): columns from frame to include in X
        y_column (list): columns from frame to include in y
    Returns:
         (object):  object with X, y, and sciframe added
    """

    def __init__(
        self,
        df=None,
        X_columns=None,
        y_column=None,
        method="Linear",
        type="regressor",
        ordinal_encode=True,
        label_encode=True,
        fit_outliers=True,
        scikit_params={},
    ):

        sci_params = _scikit_params.update(scikit_params)

        if df is not None and (X_columns is not None or y_column is not None):
            scikit_object = {}

            if isinstance(y_column, str):
                y_column = [y_column]

            if X_columns is None:
                X_columns = list(set(df.columns) - set(y_column))
            if y_column is None:
                y_column = list(set(df.columns) - set(X_columns))
                if len(y_column) > 1:
                    raise ValueError(
                        "y must be a 1D array, consider giving an input for y_column="
                    )

            scikit_object["X"], scikit_object["y"] = self._Xy_split(
                df, X_columns=X_columns, y_column=y_column
            )
            scikit_object["sciframe"] = df[X_columns + y_column]

            if fit_outliers:
                scikit_object["sciframe"] = self._fit_outliers(
                    df, y_column, fit_outliers, sci_params
                )
                scikit_object["X"] = scikit_object["sciframe"][X_columns]
                scikit_object["y"] = scikit_object["sciframe"][y_column]

            if ordinal_encode:
                enc = OrdinalEncoder()
                scikit_object["OrdinalEncoder"] = enc.fit(scikit_object["X"])
                scikit_object["X"] = enc.transform(scikit_object["X"])

            if label_encode:
                enc = LabelEncoder()
                scikit_object["LabelEncoder"] = enc.fit(scikit_object["y"])
                scikit_object["y"] = enc.transform(scikit_object["y"])

            X_train, X_test, y_train, y_test = train_test_split(
                scikit_object["X"],
                scikit_object["y"],
                test_size=sci_params["test_size"],
                random_state=sci_params["random_state"],
            )

            if type == "regressor":
                (
                    scikit_object["fit"],
                    scikit_object["predict"],
                    scikit_object["score"],
                ) = self._scikit_regressors(
                    self, X_train, X_test, y_train, y_test, method, sci_params
                )
            elif type == "classifier":
                (
                    scikit_object["fit"],
                    scikit_object["predict"],
                    scikit_object["score"],
                ) = self._scikit_classifiers(
                    self, X_train, X_test, y_train, y_test, method, sci_params
                )
            elif type == "neural_network":
                (
                    scikit_object["fit"],
                    scikit_object["predict"],
                    scikit_object["score"],
                ) = self._scikit_neural_network(
                    self, X_train, X_test, y_train, y_test, method, sci_params
                )
            elif type == "cluster":
                (
                    scikit_object["fit"],
                    scikit_object["predict"],
                    scikit_object["score"],
                ) = self._scikit_clustering(
                    self, X_train, X_test, y_train, y_test, method, sci_params
                )
            elif type == "semi_supervised":
                (
                    scikit_object["fit"],
                    scikit_object["predict"],
                    scikit_object["score"],
                ) = self._scikit_semi_supervised(
                    self, X_train, X_test, y_train, y_test, method, sci_params
                )

            return scikit_object
        else:
            raise ValueError(
                "You must give at least a DataFrame and either X_columns or y_column"
            )

    @staticmethod
    def _Xy_split(df, X_columns=None, y_column=None):

        if X_columns is not None and y_column is not None:
            X = df[X_columns]
            y = df[y_column]
        elif X_columns is None and y_column is not None:
            X = df[df.columns.values - y_column]
            y = df[y_column]
        elif X_columns is not None and y_column is None:
            X = df[X_columns]
            y = df[df.columns.values - X_columns]

        return X, y

    def _fit_outliers(self, df, ycol, which, scikit_params):

        try:
            return self._outlier_filter(
                df,
                ycol,
                which,
                scikit_params["outlier_method"],
                scikit_params["contamination"],
                None,
            )
        except ValueError:
            s_frac = 0
            while s_frac <= 1:
                try:
                    s_frac += 0.05
                    return self._outlier_filter(
                        df,
                        ycol,
                        which,
                        scikit_params["outlier_method"],
                        scikit_params["contamination"],
                        s_frac,
                    )
                except ValueError:
                    continue
            else:
                return df

    @staticmethod
    def _outlier_filter(df, ycol, which, method, threshold, s_frac):

        if method == "EllipticEnvelope":
            out = covariance.EllipticEnvelope(
                contamination=threshold,
                support_fraction=s_frac,
                random_state=_scikit_params["random_state"],
            )
        elif method == "OneClassSVM":
            out = svm.OneClassSVM(
                contamination=threshold,
                support_fraction=s_frac,
                random_state=_scikit_params["random_state"],
            )
        elif method == "LocalOutlierFactor":
            out = neighbors.LocalOutlierFactor(
                contamination=threshold,
                support_fraction=s_frac,
                random_state=_scikit_params["random_state"],
            )

        subjects = df[ycol].to_numpy().reshape(-1, 1)
        result = out.fit_predict(subjects)
        df["outliers"] = result
        if which == "remove":
            return df[df["outliers"] == 1].drop(labels="outliers", axis=1)
        elif which == "keep":
            return df[df["outliers"] == -1].drop(labels="outliers", axis=1)
        else:
            return df

    def _scikit_clustering(
        self, X_train, X_test, y_train, y_test, method, scikit_params
    ):

        if method == "Spectral":
            obj = cluster.SpectralClustering(
                n_clusters=8,
                eigen_solver=None,
                n_components=None,
                random_state=scikit_params["random_state"],
                n_init=10,
                gamma=1.0,
                affinity="rbf",
                n_neighbors=10,
                eigen_tol=0.0,
                assign_labels="kmeans",
                degree=3,
                coef0=1,
                kernel_params=None,
                n_jobs=scikit_params["n_jobs"],
                verbose=scikit_params["verbose"],
            )
        elif method == "SpectralBi":
            obj = cluster.SpectralBiclustering(
                n_clusters=3,
                method="bistochastic",
                n_components=6,
                n_best=3,
                svd_method="randomized",
                n_svd_vecs=None,
                mini_batch=False,
                init="k-means++",
                n_init=10,
                random_state=scikit_params["random_state"],
            )
        elif method == "SpectralCo":
            obj = cluster.SpectralCoclustering(
                n_clusters=3,
                svd_method="randomized",
                n_svd_vecs=None,
                mini_batch=False,
                init="k-means++",
                n_init=10,
                random_state=scikit_params["random_state"],
            )
        elif method == "KMeans":
            obj = cluster.KMeans(
                n_clusters=8,
                init="k-means++",
                n_init=10,
                max_iter=300,
                tol=0.0001,
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
                copy_x=True,
                algorithm="auto",
            )
        elif method == "Agglomerative":
            obj = cluster.AgglomerativeClustering(
                n_clusters=2,
                affinity="euclidean",
                memory=None,
                connectivity=None,
                compute_full_tree="auto",
                linkage="ward",
                distance_threshold=None,
                compute_distances=False,
            )
        elif method == "DBSCAN":
            obj = cluster.DBSCAN(
                eps=0.5,
                min_samples=5,
                metric="euclidean",
                metric_params=None,
                algorithm="auto",
                leaf_size=30,
                p=None,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "OPTICS":
            obj = cluster.OPTICS(
                min_samples=5,
                max_eps=np.inf,
                metric="minkowski",
                p=2,
                metric_params=None,
                cluster_method="xi",
                eps=None,
                xi=0.05,
                predecessor_correction=True,
                min_cluster_size=None,
                algorithm="auto",
                leaf_size=30,
                memory=None,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "AffinityPropagation":
            obj = cluster.AffinityPropagation(
                damping=0.5,
                max_iter=200,
                convergence_iter=15,
                copy=True,
                preference=None,
                affinity="euclidean",
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
            )
        elif method == "Birch":
            obj = cluster.Birch(
                threshold=0.5,
                branching_factor=50,
                n_clusters=3,
                compute_labels=True,
                copy=True,
            )
        elif method == "MiniBatchKMeans":
            obj = cluster.MiniBatchKMeans(
                n_clusters=8,
                init="k-means++",
                max_iter=100,
                batch_size=1024,
                verbose=scikit_params["verbose"],
                compute_labels=True,
                random_state=scikit_params["random_state"],
                tol=0.0,
                max_no_improvement=10,
                init_size=None,
                n_init=3,
                reassignment_ratio=0.01,
            )
        elif method == "FeatureAgglomeration":
            obj = cluster.FeatureAgglomeration(
                n_clusters=2,
                affinity="euclidean",
                memory=None,
                connectivity=None,
                compute_full_tree="auto",
                linkage="ward",
                pooling_func=np.mean,
                distance_threshold=None,
                compute_distances=False,
            )
        elif method == "MeanShift":
            obj = cluster.MeanShift(
                bandwidth=None,
                seeds=None,
                bin_seeding=False,
                min_bin_freq=1,
                cluster_all=True,
                n_jobs=scikit_params["n_jobs"],
                max_iter=300,
            )

        return self.scikit_obj(obj, X_train, X_test, y_train, y_test, "rand")

    def _scikit_regressors(
        self, X_train, X_test, y_train, y_test, method, scikit_params
    ):

        if method == "AdaBoost":
            obj = ensemble.AdaBoostRegressor(
                base_estimator=scikit_params["base_estimator"],
                n_estimators=50,
                learning_rate=1.0,
                loss="linear",
                random_state=scikit_params["random_state"],
            )
        if method == "Bagging":
            obj = ensemble.BaggingRegressor(
                base_estimator=scikit_params["base_estimator"],
                n_estimators=10,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                oob_score=False,
                warm_start=False,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                verbose=scikit_params["verbose"],
            )

        elif method == "DecisionTree":
            obj = tree.DecisionTreeRegressor(
                criterion="squared_error",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=None,
                random_state=scikit_params["random_state"],
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                ccp_alpha=0.0,
            )
        elif method == "ExtraTree":
            obj = tree.ExtraTreeRegressor(
                criterion="squared_error",
                splitter="random",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                random_state=scikit_params["random_state"],
                min_impurity_decrease=0.0,
                max_leaf_nodes=None,
                ccp_alpha=0.0,
            )
        elif method == "ExtraTrees":
            obj = ensemble.ExtraTreesRegressor(
                n_estimators=100,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                oob_score=False,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                verbose=scikit_params["verbose"],
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
            )
        elif method == "GradientBoosting":
            obj = ensemble.GradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.1,
                n_estimators=100,
                subsample=1.0,
                criterion="friedman_mse",
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_depth=3,
                min_impurity_decrease=0.0,
                init=None,
                random_state=scikit_params["random_state"],
                max_features=None,
                alpha=0.9,
                verbose=scikit_params["verbose"],
                max_leaf_nodes=None,
                warm_start=False,
                validation_fraction=0.1,
                n_iter_no_change=None,
                tol=0.0001,
                ccp_alpha=0.0,
            )
        elif method == "HistGradientBoosting":
            obj = ensemble.HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.1,
                max_iter=100,
                max_leaf_nodes=31,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=0.0,
                max_bins=255,
                categorical_features=None,
                monotonic_cst=None,
                warm_start=False,
                early_stopping="auto",
                scoring="loss",
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-07,
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
            )
        elif method == "RandomForest":
            obj = ensemble.RandomForestRegressor(
                n_estimators=100,
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                verbose=scikit_params["verbose"],
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
            )
        elif method == "Stacking":
            obj = ensemble.StackingRegressor(
                estimators=scikit_params["estimators"],
                final_estimator=scikit_params["final_estimator"],
                cv=None,
                n_jobs=scikit_params["n_jobs"],
                passthrough=False,
                verbose=scikit_params["verbose"],
            )
        elif method == "Voting":
            obj = ensemble.VotingRegressor(
                estimators=scikit_params["estimators"],
                weights=None,
                n_jobs=scikit_params["n_jobs"],
                verbose=scikit_params["verbose"],
            )
        elif method == "MultiOutput":
            obj = multioutput.MultiOutputRegressor(
                estimator=scikit_params["estimator"], n_jobs=scikit_params["n_jobs"]
            )
        elif method == "Chain":
            obj = multioutput.RegressorChain(
                base_estimator=scikit_params["base_estimator"],
                order=None,
                cv=None,
                random_state=scikit_params["random_state"],
            )
        elif method == "LinearSVR":
            obj = svm.LinearSVR(
                epsilon=0.0,
                tol=0.0001,
                C=1.0,
                loss="epsilon_insensitive",
                fit_intercept=True,
                intercept_scaling=1.0,
                dual=True,
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
                max_iter=1000,
            )
        elif method == "NuSVR":
            obj = svm.NuSVR(
                nu=0.5,
                C=1.0,
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                shrinking=True,
                tol=0.001,
                cache_size=200,
                verbose=scikit_params["verbose"],
                max_iter=-1,
            )
        elif method == "SVR":
            obj = svm.SVR(
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                tol=0.001,
                C=1.0,
                epsilon=0.1,
                shrinking=True,
                cache_size=200,
                verbose=scikit_params["verbose"],
                max_iter=-1,
            )
        elif method == "MLP":
            obj = neural_network.MLPRegressor(
                hidden_layer_sizes=(100),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="constant",
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=200,
                shuffle=True,
                random_state=scikit_params["random_state"],
                tol=0.0001,
                verbose=scikit_params["verbose"],
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=10,
                max_fun=15000,
            )
        elif method == "KNeighbors":
            obj = neighbors.KNeighborsRegressor(
                n_neighbors=5,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "RadiusNeighbors":
            obj = neighbors.RadiusNeighborsRegressor(
                radius=1.0,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "Isotonic":
            obj = isotonic.IsotonicRegression(
                y_min=None, y_max=None, increasing=True, out_of_bounds="nan"
            )
        elif method == "GaussianProcess":
            obj = gaussian_process.GaussianProcessRegressor(
                kernel=None,
                alpha=1e-10,
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=0,
                normalize_y=False,
                copy_X_train=True,
                random_state=scikit_params["random_state"],
            )
        elif method == "PLS":
            obj = cross_decomposition.PLSRegression(
                n_components=2, scale=True, max_iter=500, tol=1e-06, copy=True
            )
        elif method == "Linear":
            obj = linear_model.LinearRegression(
                fit_intercept=True,
                normalize="deprecated",
                copy_X=True,
                n_jobs=scikit_params["n_jobs"],
                positive=False,
            )
        elif method == "SGD":
            obj = linear_model.SGDRegressor(
                loss="squared_error",
                penalty="l2",
                alpha=0.0001,
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=1000,
                tol=0.001,
                shuffle=True,
                verbose=scikit_params["verbose"],
                epsilon=0.1,
                random_state=scikit_params["random_state"],
                learning_rate="invscaling",
                eta0=0.01,
                power_t=0.25,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=5,
                warm_start=False,
                average=False,
            )
        elif method == "ARD":
            obj = linear_model.ARDRegression(
                n_iter=300,
                tol=0.001,
                alpha_1=1e-06,
                alpha_2=1e-06,
                lambda_1=1e-06,
                lambda_2=1e-06,
                compute_score=False,
                threshold_lambda=10000.0,
                fit_intercept=True,
                normalize="deprecated",
                copy_X=True,
                verbose=scikit_params["verbose"],
            )
        elif method == "BayesianRidge":
            obj = linear_model.BayesianRidge(
                n_iter=300,
                tol=0.001,
                alpha_1=1e-06,
                alpha_2=1e-06,
                lambda_1=1e-06,
                lambda_2=1e-06,
                alpha_init=None,
                lambda_init=None,
                compute_score=False,
                fit_intercept=True,
                normalize="deprecated",
                copy_X=True,
                verbose=scikit_params["verbose"],
            )
        elif method == "Huber":
            obj = linear_model.HuberRegressor(
                epsilon=1.35,
                max_iter=100,
                alpha=0.0001,
                warm_start=False,
                fit_intercept=True,
                tol=1e-05,
            )
        elif method == "TheilSen":
            obj = linear_model.TheilSenRegressor(
                fit_intercept=True,
                copy_X=True,
                max_subpopulation=10000.0,
                n_subsamples=None,
                max_iter=300,
                tol=0.001,
                random_state=scikit_params["random_state"],
                n_jobs=scikit_params["n_jobs"],
                verbose=scikit_params["verbose"],
            )
        elif method == "Poisson":
            obj = linear_model.PoissonRegressor(
                alpha=1.0,
                fit_intercept=True,
                max_iter=100,
                tol=0.0001,
                warm_start=False,
                verbose=scikit_params["verbose"],
            )
        elif method == "Tweedie":
            obj = linear_model.TweedieRegressor(
                power=0.0,
                alpha=1.0,
                fit_intercept=True,
                link="auto",
                max_iter=100,
                tol=0.0001,
                warm_start=False,
                verbose=scikit_params["verbose"],
            )
        elif method == "Gamma":
            obj = linear_model.GammaRegressor(
                alpha=1.0,
                fit_intercept=True,
                max_iter=100,
                tol=0.0001,
                warm_start=False,
                verbose=scikit_params["verbose"],
            )
        elif method == "PassiveAggressive":
            obj = linear_model.PassiveAggressiveRegressor(
                C=1.0,
                fit_intercept=True,
                max_iter=1000,
                tol=0.001,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=5,
                shuffle=True,
                verbose=scikit_params["verbose"],
                loss="epsilon_insensitive",
                epsilon=0.1,
                random_state=scikit_params["random_state"],
                warm_start=False,
                average=False,
            )
        elif method == "KernelRidge":
            obj = kernel_ridge.KernelRidge(
                alpha=1,
                kernel="linear",
                gamma=None,
                degree=3,
                coef0=1,
                kernel_params=None,
            )
        # elif method == "Polynomial":
        #     obj = custom_estimators.PolynomialRegression()
        # elif method == "Exponential":
        #     obj = custom_estimators.ExponentialRegression()
        # elif method == "BiExponential":
        #     obj = custom_estimators.BiExponentialRegression()

        return self.scikit_obj(obj, X_train, X_test, y_train, y_test, "mape")

    def _scikit_classifiers(
        self, X_train, X_test, y_train, y_test, method, scikit_params
    ):

        if method == "AdaBoost":
            obj = ensemble.AdaBoostClassifier(
                base_estimator=scikit_params["base_estimator"],
                n_estimators=50,
                learning_rate=1.0,
                algorithm="SAMME.R",
                random_state=scikit_params["random_state"],
            )
        elif method == "Bagging":
            obj = ensemble.BaggingClassifier(
                base_estimator=scikit_params["base_estimator"],
                n_estimators=10,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                bootstrap_features=False,
                oob_score=False,
                warm_start=False,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                verbose=scikit_params["verbose"],
            )

        elif method == "DecisionTree":
            obj = tree.DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=None,
                random_state=scikit_params["random_state"],
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                class_weight=None,
                ccp_alpha=0.0,
            )
        elif method == "ExtraTree":
            obj = tree.ExtraTreeClassifier(
                criterion="gini",
                splitter="random",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                random_state=scikit_params["random_state"],
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                class_weight=None,
                ccp_alpha=0.0,
            )
        elif method == "ExtraTrees":
            obj = ensemble.ExtraTreesClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=False,
                oob_score=False,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                verbose=scikit_params["verbose"],
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None,
            )
        elif method == "GradientBoosting":
            obj = ensemble.GradientBoostingClassifier(
                loss="deviance",
                learning_rate=0.1,
                n_estimators=100,
                subsample=1.0,
                criterion="friedman_mse",
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_depth=3,
                min_impurity_decrease=0.0,
                init=None,
                random_state=scikit_params["random_state"],
                max_features=None,
                verbose=scikit_params["verbose"],
                max_leaf_nodes=None,
                warm_start=False,
                validation_fraction=0.1,
                n_iter_no_change=None,
                tol=0.0001,
                ccp_alpha=0.0,
            )
        elif method == "HistGradientBoosting":
            obj = ensemble.HistGradientBoostingClassifier(
                loss="auto",
                learning_rate=0.1,
                max_iter=100,
                max_leaf_nodes=31,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=0.0,
                max_bins=255,
                categorical_features=None,
                monotonic_cst=None,
                warm_start=False,
                early_stopping="auto",
                scoring="loss",
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-07,
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
            )
        elif method == "RandomForest":
            obj = ensemble.RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features="auto",
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                verbose=scikit_params["verbose"],
                warm_start=False,
                class_weight=None,
                ccp_alpha=0.0,
                max_samples=None,
            )
        elif method == "Stacking":
            obj = ensemble.StackingClassifier(
                scikit_params["estimators"],
                final_estimator=None,
                cv=None,
                stack_method="auto",
                n_jobs=scikit_params["n_jobs"],
                passthrough=False,
                verbose=scikit_params["verbose"],
            )
        elif method == "Voting":
            obj = ensemble.VotingClassifier(
                scikit_params["estimators"],
                voting="hard",
                weights=None,
                n_jobs=scikit_params["n_jobs"],
                flatten_transform=True,
                verbose=scikit_params["verbose"],
            )
        elif method == "MultiOutput":
            obj = multioutput.MultiOutputClassifier(
                estimator=scikit_params["estimator"], n_jobs=scikit_params["n_jobs"]
            )
        elif method == "Chain":
            obj = multioutput.ClassifierChain(
                scikit_params["base_estimator"],
                order=None,
                cv=None,
                random_state=scikit_params["random_state"],
            )
        elif method == "MLP":
            obj = neural_network.MLPClassifier(
                hidden_layer_sizes=(100),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="constant",
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=200,
                shuffle=True,
                random_state=scikit_params["random_state"],
                tol=0.0001,
                verbose=scikit_params["verbose"],
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=10,
                max_fun=15000,
            )
        elif method == "KNeighbors":
            obj = neighbors.KNeighborsClassifier(
                n_neighbors=5,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                metric_params=None,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "RadiusNeighbors":
            obj = neighbors.RadiusNeighborsClassifier(
                radius=1.0,
                weights="uniform",
                algorithm="auto",
                leaf_size=30,
                p=2,
                metric="minkowski",
                outlier_label=None,
                metric_params=None,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "GaussianProcess":
            obj = gaussian_process.GaussianProcessClassifier(
                kernel=None,
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=0,
                max_iter_predict=100,
                warm_start=False,
                copy_X_train=True,
                random_state=scikit_params["random_state"],
                multi_class="one_vs_rest",
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "LinearSVC":
            obj = svm.LinearSVC(
                penalty="l2",
                loss="squared_hinge",
                dual=True,
                tol=0.0001,
                C=1.0,
                multi_class="ovr",
                fit_intercept=True,
                intercept_scaling=1,
                class_weight=None,
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
                max_iter=1000,
            )
        elif method == "NuSVC":
            obj = svm.NuSVC(
                nu=0.5,
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                shrinking=True,
                probability=False,
                tol=0.001,
                cache_size=200,
                class_weight=None,
                verbose=scikit_params["verbose"],
                max_iter=-1,
                decision_function_shape="ovr",
                break_ties=False,
                random_state=scikit_params["random_state"],
            )
        elif method == "SVC":
            obj = svm.SVC(
                C=1.0,
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                shrinking=True,
                probability=False,
                tol=0.001,
                cache_size=200,
                class_weight=None,
                verbose=scikit_params["verbose"],
                max_iter=-1,
                decision_function_shape="ovr",
                break_ties=False,
                random_state=scikit_params["random_state"],
            )
        elif method == "Perceptron":
            obj = linear_model.Perceptron(
                penalty=None,
                alpha=0.0001,
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=1000,
                tol=0.001,
                shuffle=True,
                verbose=scikit_params["verbose"],
                eta0=1.0,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=5,
                class_weight=None,
                warm_start=False,
            )
        elif method == "SGD":
            obj = linear_model.SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=0.0001,
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=1000,
                tol=0.001,
                shuffle=True,
                verbose=scikit_params["verbose"],
                epsilon=0.1,
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                learning_rate="optimal",
                eta0=0.0,
                power_t=0.5,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=5,
                class_weight=None,
                warm_start=False,
                average=False,
            )
        elif method == "Ridge":
            obj = linear_model.RidgeClassifier(
                alpha=1.0,
                fit_intercept=True,
                normalize="deprecated",
                copy_X=True,
                max_iter=None,
                tol=0.001,
                class_weight=None,
                solver="auto",
                positive=False,
                random_state=scikit_params["random_state"],
            )
        elif method == "RidgeCV":
            obj = linear_model.RidgeClassifierCV(
                alphas=(0.1, 1.0, 10.0),
                fit_intercept=True,
                normalize="deprecated",
                scoring=None,
                cv=None,
                class_weight=None,
                store_cv_values=False,
            )
        elif method == "PassiveAggressive":
            obj = linear_model.PassiveAggressiveClassifier(
                C=1.0,
                fit_intercept=True,
                max_iter=1000,
                tol=0.001,
                early_stopping=False,
                validation_fraction=0.1,
                n_iter_no_change=5,
                shuffle=True,
                verbose=scikit_params["verbose"],
                loss="hinge",
                n_jobs=scikit_params["n_jobs"],
                random_state=scikit_params["random_state"],
                warm_start=False,
                class_weight=None,
                average=False,
            )
        elif method == "LogisticRegression":
            obj = linear_model.LogisticRegression(
                penalty="l2",
                dual=False,
                tol=0.0001,
                C=1.0,
                fit_intercept=True,
                intercept_scaling=1,
                class_weight=None,
                random_state=scikit_params["random_state"],
                solver="lbfgs",
                max_iter=100,
                multi_class="auto",
                verbose=scikit_params["verbose"],
                warm_start=False,
                n_jobs=scikit_params["n_jobs"],
                l1_ratio=None,
            )
        elif method == "LogisticRegressionCV":
            obj = linear_model.LogisticRegressionCV(
                Cs=10,
                fit_intercept=True,
                cv=None,
                dual=False,
                penalty="l2",
                scoring=None,
                solver="lbfgs",
                tol=0.0001,
                max_iter=100,
                class_weight=None,
                n_jobs=scikit_params["n_jobs"],
                verbose=scikit_params["verbose"],
                refit=True,
                intercept_scaling=1.0,
                multi_class="auto",
                random_state=scikit_params["random_state"],
                l1_ratios=None,
            )
        elif method == "OneVsRest":
            obj = multiclass.OneVsRestClassifier(
                scikit_params["estimator"], n_jobs=scikit_params["n_jobs"]
            )
        elif method == "OneVsOne":
            obj = multiclass.OneVsOneClassifier(
                scikit_params["estimator"], n_jobs=scikit_params["n_jobs"]
            )
        elif method == "LogisticRegressionCV":
            obj = multiclass.OutputCodeClassifier(
                scikit_params["estimator"],
                code_size=1.5,
                random_state=scikit_params["random_state"],
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "BernoulliNB":
            obj = naive_bayes.BernoulliNB(
                alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None
            )
        elif method == "CategoricalNB":
            obj = naive_bayes.CategoricalNB(
                alpha=1.0, fit_prior=True, class_prior=None, min_categories=None
            )
        elif method == "ComplementNB":
            obj = naive_bayes.ComplementNB(
                alpha=1.0, fit_prior=True, class_prior=None, norm=False
            )
        elif method == "GaussianNB":
            obj = naive_bayes.GaussianNB(priors=None, var_smoothing=1e-09)
        elif method == "MultinomialNB":
            obj = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

        return self.scikit_obj(obj, X_train, X_test, y_train, y_test, "jaccard")

    def _scikit_neural_network(
        self, X_train, X_test, y_train, y_test, method, scikit_params
    ):

        if method == "BernoulliRBM":
            obj = neural_network.BernoulliRBM(
                n_components=256,
                learning_rate=0.1,
                batch_size=10,
                n_iter=10,
                verbose=scikit_params["verbose"],
                random_state=scikit_params["random_state"],
            )

        return self.scikit_obj(obj, X_train, X_test, y_train, y_test, "score_samples")

    def _scikit_semi_supervised(
        self, X_train, X_test, y_train, y_test, method, scikit_params
    ):

        if method == "SelfTrainingClassifier":
            obj = semi_supervised.SelfTrainingClassifier(
                scikit_params["base_estimator"],
                threshold=0.75,
                criterion="threshold",
                k_best=10,
                max_iter=10,
                verbose=scikit_params["verbose"],
            )
        elif method == "LabelPropagation":
            obj = semi_supervised.LabelPropagation(
                kernel="rbf",
                gamma=20,
                n_neighbors=7,
                max_iter=1000,
                tol=0.001,
                n_jobs=scikit_params["n_jobs"],
            )
        elif method == "LabelSpreading":
            obj = semi_supervised.LabelSpreading(
                kernel="rbf",
                gamma=20,
                n_neighbors=7,
                alpha=0.2,
                max_iter=30,
                tol=0.001,
                n_jobs=scikit_params["n_jobs"],
            )

        return self.scikit_obj(obj, X_train, X_test, y_train, y_test, "score")

    @staticmethod
    def scikit_obj(obj, X_train, X_test, y_train, y_test, score_type):

        obj_fit = obj.fit(X_train, y_train)
        obj_predict = {"train": obj.predict(X_train), "test": obj.predict(X_test)}

        if score_type == "mape":
            obj_score = {
                "train": metrics.mean_absolute_percentage_error(
                    y_train, obj_predict["train"]
                ),
                "test": metrics.mean_absolute_percentage_error(
                    y_test, obj_predict["test"]
                ),
            }
        elif score_type == "jaccard":
            obj_score = {
                "train": metrics.jaccard_score(y_train, obj_predict["train"]),
                "test": metrics.jaccard_score(y_test, obj_predict["test"]),
            }
        elif score_type == "rand":
            obj_score = {
                "train": metrics.cluster.rand_score(y_train, obj_predict["train"]),
                "test": metrics.cluster.rand_score(y_test, obj_predict["test"]),
            }
        elif score_type == "score":
            obj_score = {
                "train": obj.score(X_train, y_train),
                "test": obj.score(X_test, y_test),
            }
        elif score_type == "score_samples":
            obj_score = {
                "train": obj.score_samples(X_train),
                "test": obj.score_samples(X_test),
            }

        return obj_fit, obj_predict, obj_score
