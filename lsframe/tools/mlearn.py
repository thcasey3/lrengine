"""
mlearn module, for calling scikit-learn
"""
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from . import config, utilities, scikit_models


class scikit:
    """
    Create scikit object (https://scikit-learn.org/stable/index.html)

    Args:
        df (pd.DataFrame): DataFrame with data to analyze
        method (str), optional: 'regress', 'classify', 'cluster', 'semi_supervised'
        model (str or list), optional: model used to learn default is 'RandomForest'
        update_config (dict), optional: {'arg': value} to change from the defaults for the chosen model

    Returns:
        Instantiates scikit object
    """

    def __init__(self, df, method="regress", model="RandomForest", update_config=False):

        if method == "classify":
            self.scikit_params = {**config.LEARNING_PARAMS, **config.CLASSIFYING_PARAMS}
        elif method == "regress":
            self.scikit_params = {**config.LEARNING_PARAMS, **config.REGRESSING_PARAMS}
        elif method == "cluster":
            self.scikit_params = {**config.LEARNING_PARAMS, **config.CLUSTERING_PARAMS}
        elif method == "semi_supervised":
            self.scikit_params = {
                **config.LEARNING_PARAMS,
                **config.SEMI_SUPERVISED_PARAMS,
            }

        self.scikit_params.update({"model": model, "method": method})

        if update_config:
            self.scikit_params.update(update_config)
        self.scikit_object = {}
        self.df = df

    def learn(self, select="all", append_object=False, survey=False, plots=False):
        """
        Calls scikit-learn

        Args:
            select (dict), optional: {'column': list(values)} to subset and analyze individually from the frame, default is 'all'
            append_object (bool), optional: True means collect selections or False means purge them when select is used
            survey (dict), optional: {'select': {'column': list(values)}, 'survey parameter': list(value range)}, survey['select'] can be 'all' to use entire frame
            plot (bool), optional: True means raise plot after analysis

        Returns:
            Populates scikit object with results
        """
        self.scores_dict = {"subject": [], "score": [], "model": []}

        if survey:
            survey_par = [x for x in list(survey.keys()) if x != "select"][0]
            if survey["select"] == "all":
                itm_col = "all"
                survey["select"] = {"all": ["all"]}
            else:
                itm_col = list(survey["select"].keys())[0]
                if isinstance(survey["select"][itm_col], str):
                    survey["select"][itm_col] = [survey["select"][itm_col]]
            self.scores_dict[survey_par] = []
            for itm in survey["select"][itm_col]:
                if not append_object:
                    self.scikit_object = {}
                for par in survey[survey_par]:
                    self.scikit_params.update({survey_par: par})
                    self.prepare(select=itm, column=itm_col)
                    self.train(select=itm)
                    self.predict_score(select=itm)
                    if plots:
                        self.plots(select=itm)
                    self.scores_dict[survey_par].append(par)
        else:
            if select == "all":
                itm_col = "all"
                select = {"all": ["all"]}
            else:
                itm_col = list(select.keys())[0]
                if isinstance(select[itm_col], str):
                    select[itm_col] = [select[itm_col]]
            for itm in select[itm_col]:
                if not append_object:
                    self.scikit_object = {}
                self.prepare(select=itm, column=itm_col)
                self.train(select=itm)
                self.predict_score(select=itm)
                if plots:
                    self.plots(select=itm)

    def prepare(self, select="all", column=False):

        if select != "all":
            df = self.df[self.df[column] == select]
        elif select == "all":
            df = self.df.copy()

        if len(df) > 0:
            self.scikit_object[select] = {}

            if self.scikit_params["outliers"] and len(df) > 1:
                inputs = list(self.scikit_params["outliers"].items())
                df = utilities.fit_outliers(
                    df, inputs[0][0], inputs[0][1], self.scikit_params["ycolumn"]
                )

            xcol = self.scikit_params["Xcolumns"]
            df = utilities.handle_na(df, self.scikit_params["handle_na"])

            if self.scikit_params["drop_uniform"]:
                df, cols = utilities.drop_uniform(df)
                if len(cols) > 0:
                    self.scikit_object[select]["uniform_columns"] = cols
                    xcol = [x for x in self.scikit_params["Xcolumns"] if x not in cols]

            if self.scikit_params["split_column"]:
                for col in self.scikit_params["split_column"].keys():
                    df, wrd_lst = utilities.split_cols(
                        df=df,
                        base_column=col,
                        use_top=self.scikit_params["split_column"][col],
                    )
                    xcol = xcol + wrd_lst

            train_df, predict_df = self.train_predict_split(df=df)
            X_predict_df = predict_df[xcol]

            self.scikit_object[select]["df"] = df.copy()

            X_df = df[xcol]
            X_train_df = train_df[xcol]
            if self.scikit_params["encode_X"]:
                X_df, self.scikit_object[select]["X_encoder"] = utilities.encoder(
                    X_df, "ordinal"
                )
                if len(X_train_df) > 0:
                    X_train_df = self.scikit_object[select]["X_encoder"].transform(
                        X_train_df
                    )
                if len(X_predict_df) > 0:
                    X_predict_df = self.scikit_object[select]["X_encoder"].transform(
                        X_predict_df
                    )
            else:
                self.scikit_object[select]["X_encoder"] = None

            if self.scikit_params["scale"]:
                if self.scikit_params["scale"] == "robust":
                    X_df, self.scikit_object[select]["scaler"] = utilities.scaler(
                        X_df, "robust"
                    )
                else:
                    X_df, self.scikit_object[select]["scaler"] = utilities.scaler(
                        X_df, "standard"
                    )

                if len(X_train_df) > 0:
                    X_train_df = self.scikit_object[select]["scaler"].transform(
                        X_train_df
                    )
                if len(X_predict_df) > 0:
                    X_predict_df = self.scikit_object[select]["scaler"].transform(
                        X_predict_df
                    )
            else:
                self.scikit_object[select]["scaler"] = None

            if self.scikit_params["normalize"]:
                X_df, self.scikit_object[select]["normalizer"] = utilities.normalizer(
                    X_df
                )
                if len(X_train_df) > 0:
                    X_train_df = self.scikit_object[select]["normalizer"].transform(
                        X_train_df
                    )
                if len(X_predict_df) > 0:
                    X_predict_df = self.scikit_object[select]["normalizer"].transform(
                        X_predict_df
                    )
            else:
                self.scikit_object[select]["normalizer"] = None

            if len(X_train_df) > 0:
                self.scikit_object[select]["X"] = pd.DataFrame(
                    data=X_train_df, columns=xcol
                )
                if self.scikit_params["encode_y"]:
                    (
                        self.scikit_object[select]["y"],
                        self.scikit_object[select]["y_encoder"],
                    ) = utilities.encoder(
                        train_df[[self.scikit_params["ycolumn"]]], "label"
                    )
                else:
                    self.scikit_object[select]["y"] = train_df[
                        [self.scikit_params["ycolumn"]]
                    ].to_numpy()

            if len(X_predict_df) > 0:
                self.scikit_object[select]["X_predict"] = pd.DataFrame(
                    data=X_predict_df, columns=xcol
                )
        else:
            self.scikit_object[select] = {select: None}

    def train(self, select=False):

        model, _ = self.pre_qualify_lists(self.scikit_params["model"], [])
        if all(map(list(self.scikit_object[select].keys()).__contains__, ["X", "y"])):
            if len(self.scikit_object[select]["X"]) > 10:
                (
                    self.scikit_object[select]["X_train"],
                    self.scikit_object[select]["X_test"],
                    y_train,
                    y_test,
                ) = train_test_split(
                    self.scikit_object[select]["X"],
                    self.scikit_object[select]["y"],
                    test_size=self.scikit_params["test_size"],
                    random_state=1,
                )
                self.scikit_object[select]["y_train"] = y_train.reshape(-1)
                self.scikit_object[select]["y_test"] = y_test.reshape(-1)

            else:
                self.scikit_object[select]["X_train"] = self.scikit_object[select]["X"]
                self.scikit_object[select]["y_train"] = self.scikit_object[select]["y"]

            for meth in model:
                if isinstance(meth, dict):
                    for key in meth.keys():
                        self.scikit_object[select][key] = {}
                        for val in meth[key]:
                            self.scikit_object[select][key][val] = self.scikit_fit(
                                self.scikit_object[select]["X_train"],
                                self.scikit_object[select]["y_train"],
                                key,
                                base_estimator=val,
                            )
                else:
                    self.scikit_object[select][meth] = self.scikit_fit(
                        self.scikit_object[select]["X_train"],
                        self.scikit_object[select]["y_train"],
                        meth,
                    )

    def scikit_fit(self, X_train, y_train, model, base_estimator=False):

        if self.scikit_params["method"] == "regress":
            return scikit_models.regression_methods.scikit_regression(
                X_train,
                y_train,
                model,
                self.scikit_params,
                base_estimator=base_estimator,
            )
        elif self.scikit_params["method"] == "classify":
            return scikit_models.classification_methods.scikit_classification(
                X_train,
                y_train,
                model,
                self.scikit_params,
                base_estimator=base_estimator,
            )
        elif self.scikit_params["method"] == "cluster":
            return scikit_models.clustering_methods.scikit_clustering(
                X_train, y_train, model, self.scikit_params
            )

        elif self.scikit_params["method"] == "semi_supervised":
            return scikit_models.semi_supervised_methods.scikit_semi_supervised(
                X_train, y_train, model, self.scikit_params
            )

    def predict_score(self, select=False):

        model, metric = self.pre_qualify_lists(
            self.scikit_params["model"], self.scikit_params["metric"]
        )

        self.scikit_object[select]["scores"] = {}
        self.scikit_object[select]["predictions"] = {}
        if all(
            map(
                list(self.scikit_object[select].keys()).__contains__,
                ["X_test", "y_test"],
            )
        ):
            for meth in model:
                if isinstance(meth, dict):
                    for key in meth.keys():
                        self.scikit_object[select]["scores"][key] = {}
                        self.scikit_object[select]["predictions"][key] = {}
                        for val in meth[key]:
                            self.scikit_object[select]["scores"][key][val] = {}
                            self.scikit_object[select]["predictions"][key][val] = {
                                "train": self.scikit_object[select][key][val].predict(
                                    self.scikit_object[select]["X_train"]
                                ),
                                "test": self.scikit_object[select][key][val].predict(
                                    self.scikit_object[select]["X_test"]
                                ),
                            }
                            if "X_predict" in self.scikit_object[select].keys():
                                self.scikit_object[select]["predictions"][key][
                                    val
                                ].update(
                                    {
                                        "sample": self.scikit_object[select][key][
                                            val
                                        ].predict(
                                            self.scikit_object[select]["X_predict"]
                                        )
                                    }
                                )

                            for metr in metric:
                                self.scikit_object[select]["scores"][key][val][metr] = {
                                    "train": utilities.return_metric(
                                        self.scikit_object[select][key][val],
                                        X_true=self.scikit_object[select]["X_train"],
                                        y_true=self.scikit_object[select]["y_train"],
                                        y_pred=self.scikit_object[select][
                                            "predictions"
                                        ][key][val]["train"],
                                        metric=metr,
                                    ),
                                    "test": utilities.return_metric(
                                        self.scikit_object[select][key][val],
                                        X_true=self.scikit_object[select]["X_test"],
                                        y_true=self.scikit_object[select]["y_test"],
                                        y_pred=self.scikit_object[select][
                                            "predictions"
                                        ][key][val]["test"],
                                        metric=metr,
                                    ),
                                }
                                self.scores_dict["subject"].append(select)
                                self.scores_dict["score"].append(
                                    self.scikit_object[select]["scores"][key][val][metr]
                                )
                                self.scores_dict["model"].append(
                                    "_".join([key, val, metr])
                                )
                                print(
                                    f"{select} score: "
                                    + str(
                                        self.scikit_object[select]["scores"][key][val][
                                            metr
                                        ]
                                    )
                                    + " "
                                    + key
                                    + "_"
                                    + val
                                    + "_"
                                    + metr
                                )

                elif meth in self.scikit_object[select].keys():
                    self.scikit_object[select]["predictions"][meth] = {
                        "train": self.scikit_object[select][meth].predict(
                            self.scikit_object[select]["X_train"]
                        ),
                        "test": self.scikit_object[select][meth].predict(
                            self.scikit_object[select]["X_test"]
                        ),
                    }
                    if "X_predict" in self.scikit_object[select].keys():
                        self.scikit_object[select]["predictions"][meth].update(
                            {
                                "sample": self.scikit_object[select][meth].predict(
                                    self.scikit_object[select]["X_predict"]
                                )
                            }
                        )
                    self.scikit_object[select]["scores"][meth] = {}
                    for metr in metric:
                        self.scikit_object[select]["scores"][meth][metr] = {
                            "train": utilities.return_metric(
                                self.scikit_object[select][meth],
                                X_true=self.scikit_object[select]["X_train"],
                                y_true=self.scikit_object[select]["y_train"],
                                y_pred=self.scikit_object[select]["predictions"][meth][
                                    "train"
                                ],
                                metric=metr,
                            ),
                            "test": utilities.return_metric(
                                self.scikit_object[select][meth],
                                X_true=self.scikit_object[select]["X_test"],
                                y_true=self.scikit_object[select]["y_test"],
                                y_pred=self.scikit_object[select]["predictions"][meth][
                                    "test"
                                ],
                                metric=metr,
                            ),
                        }
                        self.scores_dict["subject"].append(select)
                        self.scores_dict["score"].append(
                            self.scikit_object[select]["scores"][meth][metr]
                        )
                        self.scores_dict["model"].append("_".join([meth, metr]))
                        print(
                            f"{select} score: "
                            + str(self.scikit_object[select]["scores"][meth][metr])
                            + " "
                            + meth
                            + "_"
                            + metr
                        )

                else:
                    sc = None
                    mt = meth
                    self.scikit_object[select]["scores"][meth] = None
                    self.scikit_object[select]["y_predict"][meth] = None

                    print(f"{select} score: {sc} {mt}")
        else:
            self.scikit_object[select]["scores"] = None
            self.scikit_object[select]["y_predict"] = None
            print(f"{select} None")

        if self.scikit_params["decode"]:
            if self.scikit_params["encode_X"]:
                for x in ["X", "X_train", "X_test", "X_predict"]:
                    if (
                        x in self.scikit_object[select].keys()
                        and self.scikit_object[select][x] is not None
                    ):
                        self.scikit_object[select][x] = utilities.decoder(
                            self.scikit_object[select][x],
                            self.scikit_object[select]["X_encoder"],
                        )
            if self.scikit_params["encode_y"]:
                for y in ["y", "y_train", "y_test", "y_predict"]:
                    if (
                        y in self.scikit_object[select].keys()
                        and self.scikit_object[select][y] is not None
                    ):
                        self.scikit_object[select][y] = utilities.decoder(
                            self.scikit_object[select][y],
                            self.scikit_object[select]["y_encoder"],
                        )

    def plots(self, select):

        model, _ = self.pre_qualify_lists(self.scikit_params["model"], [])
        if self.scikit_object[select]["predictions"] and all(
            [x in self.scikit_object[select].keys() for x in ["y_test", "y_train"]]
        ):
            for meth in model:
                if isinstance(meth, dict):
                    for key in meth.keys():
                        if key in self.scikit_object[select]["predictions"].keys():
                            for val in meth[key]:
                                if self.scikit_object[select]["predictions"][key] and (
                                    val
                                    in self.scikit_object[select]["predictions"][
                                        key
                                    ].keys()
                                ):
                                    plt.figure(figsize=(12, 6))
                                    plt.subplot(121)
                                    plt.plot(
                                        self.scikit_object[select]["y_train"],
                                        "o",
                                        label="Train points",
                                        markerfacecolor="#a80f8c",
                                        markeredgecolor=None,
                                        alpha=0.5,
                                        markersize=6,
                                    )
                                    plt.plot(
                                        self.scikit_object[select]["predictions"][key][
                                            val
                                        ]["train"],
                                        "x",
                                        color="#a80f8c",
                                        label="Predictions",
                                        markersize=5,
                                    )
                                    plt.title(
                                        select
                                        + ": "
                                        + "_".join([key, val])
                                        + " (Train)"
                                    )
                                    plt.legend()
                                    plt.subplot(122)
                                    plt.plot(
                                        self.scikit_object[select]["y_test"],
                                        "o",
                                        label="Test points",
                                        markerfacecolor="#a80f8c",
                                        markeredgecolor=None,
                                        alpha=0.5,
                                        markersize=6,
                                    )
                                    plt.plot(
                                        self.scikit_object[select]["predictions"][key][
                                            val
                                        ]["test"],
                                        "x",
                                        color="#a80f8c",
                                        label="Predictions",
                                        markersize=5,
                                    )
                                    plt.title(
                                        select + ": " + "_".join([key, val]) + " (Test)"
                                    )
                                    plt.legend()
                                    plt.show()
                else:
                    if meth in self.scikit_object[select]["predictions"].keys():
                        plt.figure(figsize=(12, 6))
                        plt.subplot(121)
                        plt.plot(
                            self.scikit_object[select]["y_train"],
                            "o",
                            label="Train points",
                            markerfacecolor="#a80f8c",
                            markeredgecolor=None,
                            alpha=0.5,
                            markersize=6,
                        )
                        plt.plot(
                            self.scikit_object[select]["predictions"][meth]["train"],
                            "x",
                            color="#a80f8c",
                            label="Predictions",
                            markersize=5,
                        )
                        plt.title(select + ": " + meth + " (Train)")
                        plt.legend()
                        plt.subplot(122)
                        plt.plot(
                            self.scikit_object[select]["y_test"],
                            "o",
                            label="Test points",
                            markerfacecolor="#a80f8c",
                            markeredgecolor=None,
                            alpha=0.5,
                            markersize=6,
                        )
                        plt.plot(
                            self.scikit_object[select]["predictions"][meth]["test"],
                            "x",
                            color="#a80f8c",
                            label="Predictions",
                            markersize=5,
                        )
                        plt.title(select + ": " + meth + " (Test)")
                        plt.legend()
                        plt.show()

    @staticmethod
    def train_predict_split(
        df=None, date_column="since_start", cost_column="TotActCosts"
    ):

        predict_df_1 = df[df[date_column] <= 0]
        predict_df_2 = df[df[cost_column] <= 0]
        predict_df = predict_df_1.append(predict_df_2, ignore_index=True)

        train_df = df[df[date_column] > 0]
        train_df = train_df[train_df[cost_column] > 0]

        return train_df.drop_duplicates(), predict_df.drop_duplicates()

    @staticmethod
    def pre_qualify_lists(model, metric):

        if not isinstance(model, list):
            model = [model]

        for indx, meth in enumerate(model):
            if isinstance(meth, dict):
                for key in meth.keys():
                    if not isinstance(meth[key], list):
                        model[indx][key] = [meth[key]]

        if not isinstance(metric, list):
            metric = [metric]

        return model, metric

    def scores_to_df(self, save=False):

        self.scores_df = utilities.scores_dict_to_df(self.scores_dict)

        self.scores_df["metric"] = [x.split("_")[-1] for x in self.scores_dict["model"]]

        method_lst = []
        base_est_lst = []
        for x in self.scores_dict["model"]:
            splt = x.split("_")
            method_lst.append(splt[0])
            if len(splt) > 2:
                base_est_lst.append(splt[1])
            else:
                base_est_lst.append("")

        self.scores_df["model"] = method_lst
        self.scores_df["base_estimator"] = base_est_lst

        self.scores_df.sort_values(by="subject", ascending=False, inplace=True)

        if save and isinstance(save, str):
            self.scores_df.to_csv(save, index=False)
