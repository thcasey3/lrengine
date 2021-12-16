from sklearn import (
    neighbors,
    ensemble,
    tree,
    svm,
    neural_network,
    isotonic,
    gaussian_process,
    cross_decomposition,
    linear_model,
    kernel_ridge,
    multioutput,
)


def scikit_regression(X_train, y_train, method, scikit_params, base_estimator=False):

    if isinstance(scikit_params["n_estimators"], dict):
        n_est = int(scikit_params["n_estimators"]["auto"] * len(X_train))
    else:
        n_est = scikit_params["n_estimators"]

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
                n_estimators=n_est,
                criterion=scikit_params["criterion"],
                max_features=scikit_params["max_features"],
                max_samples=scikit_params["max_samples"],
                min_samples_split=scikit_params["min_samples_split"],
                min_samples_leaf=scikit_params["min_samples_leaf"],
                max_depth=scikit_params["max_depth"],
                n_jobs=scikit_params["n_jobs"],
                bootstrap=scikit_params["bootstrap"],
                min_weight_fraction_leaf=scikit_params["min_weight_fraction_leaf"],
                max_leaf_nodes=scikit_params["max_leaf_nodes"],
                min_impurity_decrease=scikit_params["min_impurity_decrease"],
                oob_score=scikit_params["oob_score"],
                verbose=scikit_params["verbose"],
                warm_start=scikit_params["warm_start"],
                ccp_alpha=scikit_params["ccp_alpha"],
            )
        else:
            base_est = scikit_params["base_estimator"]

        if method == "AdaBoost":
            obj = ensemble.AdaBoostRegressor(
                random_state=scikit_params["random_state"],
                base_estimator=base_est,
                loss=scikit_params["loss_a"],
                n_estimators=n_est,
                learning_rate=scikit_params["learning_rate"],
            )
        if method == "Bagging":
            obj = ensemble.BaggingRegressor(
                random_state=scikit_params["random_state"],
                base_estimator=base_est,
                n_estimators=n_est,
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
            n_estimators=n_est,
            criterion=scikit_params["criterion"],
            max_features=scikit_params["max_features"],
            max_samples=scikit_params["max_samples"],
            n_jobs=scikit_params["n_jobs"],
            bootstrap=scikit_params["bootstrap"],
        )
    elif method == "GradientBoosting":
        obj = ensemble.GradientBoostingRegressor(
            random_state=scikit_params["random_state"],
            n_estimators=n_est,
            criterion=scikit_params["criterion"],
            loss=scikit_params["loss_g"],
        )
    elif method == "HistGradientBoosting":
        obj = ensemble.HistGradientBoostingRegressor(
            random_state=scikit_params["random_state"],
            loss=scikit_params["loss_h"],
            l2_regularization=scikit_params["l2_regularization"],
        )
    elif method == "RandomForest":
        obj = ensemble.RandomForestRegressor(
            random_state=scikit_params["random_state"],
            n_estimators=n_est,
            criterion=scikit_params["criterion"],
            max_features=scikit_params["max_features"],
            max_samples=scikit_params["max_samples"],
            min_samples_split=scikit_params["min_samples_split"],
            min_samples_leaf=scikit_params["min_samples_leaf"],
            max_depth=scikit_params["max_depth"],
            n_jobs=scikit_params["n_jobs"],
            bootstrap=scikit_params["bootstrap"],
            min_weight_fraction_leaf=scikit_params["min_weight_fraction_leaf"],
            max_leaf_nodes=scikit_params["max_leaf_nodes"],
            min_impurity_decrease=scikit_params["min_impurity_decrease"],
            oob_score=scikit_params["oob_score"],
            verbose=scikit_params["verbose"],
            warm_start=scikit_params["warm_start"],
            ccp_alpha=scikit_params["ccp_alpha"],
        )
    elif method == "Stacking":
        obj = ensemble.StackingRegressor(
            estimators=scikit_params["estimators"],
            final_estimator=scikit_params["final_estimator"],
        )
    elif method == "Voting":
        obj = ensemble.VotingRegressor(estimators=scikit_params["estimators"])
    elif method == "MLP":
        obj = neural_network.MLPRegressor(
            random_state=scikit_params["random_state"],
            activation=scikit_params["activation"],
            solver=scikit_params["solver"],
            learning_rate=scikit_params["learning_rate"],
            learning_rate_init=scikit_params["learning_rate_init"],
            power_t=scikit_params["power_t"],
            hidden_layer_sizes=scikit_params["hidden_layer_sizes"],
            alpha=scikit_params["alpha"],
            batch_size=scikit_params["batch_size"],
            max_iter=scikit_params["max_iter"],
            shuffle=scikit_params["shuffle"],
            tol=scikit_params["tol"],
            verbose=scikit_params["verbose"],
            warm_start=scikit_params["warm_start"],
            momentum=scikit_params["momentum"],
            nesterovs_momentum=scikit_params["nesterovs_momentum"],
            early_stopping=scikit_params["early_stopping"],
            validation_fraction=scikit_params["validation_fraction"],
            beta_1=scikit_params["beta_1"],
            beta_2=scikit_params["beta_2"],
            epsilon=scikit_params["epsilon"],
            n_iter_no_change=scikit_params["n_iter_no_change"],
            max_fun=scikit_params["max_fun"],
        )
    elif method == "KNeighbors":
        obj = neighbors.KNeighborsRegressor(weights=scikit_params["weights"])
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

    obj = obj.fit(X_train, y_train)

    return obj
