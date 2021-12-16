from sklearn import cluster


def scikit_clustering(X_train, y_train, method, scikit_params):

    if method == "Spectral":
        obj = cluster.SpectralClustering(
            n_clusters=scikit_params["n_clusters"],
            eigen_solver=scikit_params["eigen_solver"],
            n_components=scikit_params["n_components"],
            random_state=scikit_params["random_state"],
            n_init=scikit_params["n_init"],
            gamma=scikit_params["gamma"],
            affinity=scikit_params["affinity"],
            n_neighbors=scikit_params["n_neighbors"],
            eigen_tol=scikit_params["eigen_tol"],
            assign_labels=scikit_params["assign_labels"],
            degree=scikit_params["degree"],
            coef0=scikit_params["coef0"],
            kernel_params=scikit_params["kernel_params"],
            n_jobs=scikit_params["n_jobs"],
            verbose=scikit_params["verbose"],
        )
    elif method == "SpectralBi":
        obj = cluster.SpectralBiclustering(
            n_clusters=scikit_params["n_clusters"],
            method=scikit_params["method"],
            n_components=scikit_params["n_components"],
            n_best=scikit_params["n_best"],
            svd_method=scikit_params["svd_method"],
            n_svd_vecs=scikit_params["n_svd_vecs"],
            mini_batch=scikit_params["mini_batch"],
            init=scikit_params["init"],
            n_init=scikit_params["n_init"],
            random_state=scikit_params["random_state"],
        )
    elif method == "SpectralCo":
        obj = cluster.SpectralCoclustering(
            n_clusters=scikit_params["n_clusters"],
            svd_method=scikit_params["svd_method"],
            n_svd_vecs=scikit_params["n_svd_vecs"],
            mini_batch=scikit_params["mini_batch"],
            init=scikit_params["init"],
            n_init=scikit_params["n_init"],
            random_state=scikit_params["random_state"],
        )
    elif method == "KMeans":
        obj = cluster.KMeans(
            n_clusters=scikit_params["n_clusters"],
            init=scikit_params["init"],
            n_init=scikit_params["n_init"],
            max_iter=scikit_params["max_iter"],
            tol=scikit_params["tol"],
            verbose=scikit_params["verbose"],
            random_state=scikit_params["random_state"],
            copy_x=scikit_params["copy_x"],
            algorithm=scikit_params["algorithm"],
        )
    elif method == "Agglomerative":
        obj = cluster.AgglomerativeClustering(
            n_clusters=scikit_params["n_clusters"],
            affinity=scikit_params["affinity"],
            memory=scikit_params["memory"],
            connectivity=scikit_params["connectivity"],
            compute_full_tree=scikit_params["compute_full_tree"],
            linkage=scikit_params["linkage"],
            distance_threshold=scikit_params["distance_threshold"],
            compute_distances=scikit_params["compute_distances"],
        )
    elif method == "DBSCAN":
        obj = cluster.DBSCAN(
            eps=scikit_params["eps"],
            min_samples=scikit_params["min_samples"],
            metric=scikit_params["metric"],
            metric_params=scikit_params["metric_params"],
            algorithm=scikit_params["algorithm"],
            leaf_size=scikit_params["leaf_size"],
            p=scikit_params["p"],
            n_jobs=scikit_params["n_jobs"],
        )
    elif method == "OPTICS":
        obj = cluster.OPTICS(
            min_samples=scikit_params["min_samples"],
            max_eps=scikit_params["max_eps"],
            metric=scikit_params["metric"],
            p=scikit_params["p"],
            metric_params=scikit_params["metric_params"],
            cluster_method=scikit_params["cluster_method"],
            eps=scikit_params["eps"],
            xi=scikit_params["xi"],
            predecessor_correction=scikit_params["predecessor_correction"],
            min_cluster_size=scikit_params["min_cluster_size"],
            algorithm=scikit_params["algorithm"],
            leaf_size=scikit_params["leaf_size"],
            memory=scikit_params["memory"],
            n_jobs=scikit_params["n_jobs"],
        )
    elif method == "AffinityPropagation":
        obj = cluster.AffinityPropagation(
            damping=scikit_params["damping"],
            max_iter=scikit_params["max_iter"],
            convergence_iter=scikit_params["convergence_iter"],
            copy=scikit_params["copy"],
            preference=scikit_params["preference"],
            affinity=scikit_params["affinity"],
            verbose=scikit_params["verbose"],
            random_state=scikit_params["random_state"],
        )
    elif method == "Birch":
        obj = cluster.Birch(
            threshold=scikit_params["threshold"],
            branching_factor=scikit_params["branching_factor"],
            n_clusters=scikit_params["n_clusters"],
            compute_labels=scikit_params["compute_labels"],
            copy=scikit_params["copy"],
        )
    elif method == "MiniBatchKMeans":
        obj = cluster.MiniBatchKMeans(
            n_clusters=scikit_params["n_clusters"],
            init=scikit_params["init"],
            max_iter=scikit_params["max_iter"],
            batch_size=scikit_params["batch_size"],
            verbose=scikit_params["verbose"],
            compute_labels=scikit_params["compute_labels"],
            random_state=scikit_params["random_state"],
            tol=scikit_params["tol"],
            max_no_improvement=scikit_params["max_no_improvement"],
            init_size=scikit_params["init_size"],
            n_init=scikit_params["n_init"],
            reassignment_ratio=scikit_params["reassignment_ratio"],
        )
    elif method == "FeatureAgglomeration":
        obj = cluster.FeatureAgglomeration(
            n_clusters=scikit_params["n_clusters"],
            affinity=scikit_params["affinity"],
            memory=scikit_params["memory"],
            connectivity=scikit_params["connectivity"],
            compute_full_tree=scikit_params["compute_full_tree"],
            linkage=scikit_params["linkage"],
            pooling_func=scikit_params["pooling_func"],
            distance_threshold=scikit_params["distance_threshold"],
            compute_distances=scikit_params["compute_distances"],
        )
    elif method == "MeanShift":
        obj = cluster.MeanShift(
            bandwidth=scikit_params["bandwidth"],
            seeds=scikit_params["seeds"],
            bin_seeding=scikit_params["bin_seeding"],
            min_bin_freq=scikit_params["min_bin_freq"],
            cluster_all=scikit_params["cluster_all"],
            n_jobs=scikit_params["n_jobs"],
            max_iter=scikit_params["max_iter"],
        )

    obj = obj.fit(X_train, y_train)

    return obj
