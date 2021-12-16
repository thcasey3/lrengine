from sklearn import semi_supervised


def scikit_semi_supervised(X_train, y_train, method, scikit_params):

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

    obj = obj.fit(X_train, y_train)

    return obj
