from sklearn import (
    neighbors,
    ensemble,
    tree,
    svm,
    neural_network,
    gaussian_process,
    linear_model,
    multioutput,
    naive_bayes,
    multiclass,
)


def scikit_classification(
    X_train, y_train, method, scikit_params, base_estimator=False
):
    if isinstance(scikit_params["n_estimators"], dict):
        n_est = int(scikit_params["n_estimators"]["auto"] * len(X_train))
    else:
        n_est = scikit_params["n_estimators"]

    if method in ["AdaBoost", "Bagging"]:
        if base_estimator == "DecisionTree":
            base_est = tree.DecisionTreeClassifier(
                random_state=scikit_params["random_state"],
                criterion=scikit_params["criterion"],
            )
        elif base_estimator == "ExtraTree":
            base_est = tree.ExtraTreeClassifier(
                random_state=scikit_params["random_state"],
                criterion=scikit_params["criterion"],
            )
        elif base_estimator == "RandomForest":
            base_est = ensemble.RandomForestClassifier(
                random_state=scikit_params["random_state"],
                n_estimators=n_est,
                criterion=scikit_params["criterion"],
                max_features=scikit_params["max_features"],
                max_samples=scikit_params["max_samples"],
                n_jobs=scikit_params["n_jobs"],
                bootstrap=scikit_params["bootstrap"],
            )
        else:
            base_est = scikit_params["base_estimator"]

        if method == "AdaBoost":
            obj = ensemble.AdaBoostClassifier(
                random_state=scikit_params["random_state"],
                base_estimator=base_est,
                loss=scikit_params["loss_a"],
                n_estimators=n_est,
            )
        if method == "Bagging":
            obj = ensemble.BaggingClassifier(
                random_state=scikit_params["random_state"],
                base_estimator=base_est,
                n_estimators=n_est,
                n_jobs=scikit_params["n_jobs"],
            )

    elif method == "DecisionTree":
        obj = tree.DecisionTreeClassifier(
            random_state=scikit_params["random_state"],
            criterion=scikit_params["criterion"],
        )
    elif method == "ExtraTree":
        obj = tree.ExtraTreeClassifier(
            random_state=scikit_params["random_state"],
            criterion=scikit_params["criterion"],
        )
    elif method == "ExtraTrees":
        obj = ensemble.ExtraTreesClassifier(
            random_state=scikit_params["random_state"],
            n_estimators=n_est,
            criterion=scikit_params["criterion"],
            max_features=scikit_params["max_features"],
            max_samples=scikit_params["max_samples"],
            n_jobs=scikit_params["n_jobs"],
            bootstrap=scikit_params["bootstrap"],
        )
    elif method == "GradientBoosting":
        obj = ensemble.GradientBoostingClassifier(
            random_state=scikit_params["random_state"],
            n_estimators=n_est,
            criterion=scikit_params["criterion_g"],
            loss=scikit_params["loss_g"],
        )
    elif method == "HistGradientBoosting":
        obj = ensemble.HistGradientBoostingClassifier(
            random_state=scikit_params["random_state"],
            loss=scikit_params["loss_h"],
            l2_regularization=scikit_params["l2_regularization"],
        )
    elif method == "RandomForest":
        obj = ensemble.RandomForestClassifier(
            random_state=scikit_params["random_state"],
            n_estimators=n_est,
            criterion=scikit_params["criterion"],
            max_features=scikit_params["max_features"],
            max_samples=scikit_params["max_samples"],
            n_jobs=scikit_params["n_jobs"],
            bootstrap=scikit_params["bootstrap"],
        )
    elif method == "Stacking":
        obj = ensemble.StackingClassifier(
            estimators=scikit_params["estimators"],
            final_estimator=scikit_params["final_estimator"],
        )
    elif method == "Voting":
        obj = ensemble.VotingClassifier(estimators=scikit_params["estimators"])
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
            random_state=scikit_params["random_state"],
            activation=scikit_params["activation"],
            solver=scikit_params["solver"],
            learning_rate=scikit_params["learning_rate"],
            learning_rate_init=scikit_params["learning_rate_init"],
            power_t=scikit_params["power_t"],
        )
    elif method == "KNeighbors":
        obj = neighbors.KNeighborsClassifier(weights=scikit_params["weights"])
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
            random_state=scikit_params["random_state"]
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

    obj = obj.fit(X_train, y_train)

    return obj
