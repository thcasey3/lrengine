import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing, metrics, covariance
from sklearn.model_selection import train_test_split

from . import utilities, scikit_models
from scikit_models import config


class scikit:
    def __init__(self, update_config=False):

        if "model_selection" in update_config.keys():
            model_select = update_config["model_selection"]
        else:
            model_select = config.LEARNING_PARAMS["model_selection"]

        if model_select == "classify":
            self.scikit_params = {**config.LEARNING_PARAMS, **config.CLASSIFYING_PARAMS}
        elif model_select == "regress":
            self.scikit_params = {**config.LEARNING_PARAMS, **config.REGRESSING_PARAMS}
        elif model_select == "cluster":
            self.scikit_params = {**config.LEARNING_PARAMS, **config.CLUSTERING_PARAMS}
        elif model_select == "semi_supervised":
            self.scikit_params = {
                **config.LEARNING_PARAMS,
                **config.SEMI_SUPERVISED_PARAMS,
            }

        if update_config:
            self.scikit_params.update(update_config)
        self.scikit_object = {}
        self.df = {}

    def learn(
        self,
        select="all",
        append_object=False,
        plot="plot",
        survey=False,
        aggregate=False,
    ):

        self.scores_dict = {"Name": [], "score": [], "method": []}
        if select == "all":
            self.prepare(select="all", aggregate=aggregate)
            self.train(select="all")
            self.predict_score(select="all")
            if plot == "plot":
                self.plot(select="all")
            elif plot == "scatter":
                self.scatter(select="all")
        else:
            itm_col = list(select.keys())[0]
            if isinstance(select[itm_col], str):
                select[itm_col] = [select[itm_col]]

            if survey:
                survey_par = [x for x in list(survey.keys()) if x != "item"][0]
                for par in survey[survey_par]:
                    self.scikit_params.update({survey_par: par})
                    self.prepare(
                        select=survey["item"], column=itm_col, aggregate=aggregate
                    )
                    self.train(select=survey["item"])
                    self.predict_score(select=survey["item"])
                self.scores_dict[survey_par] = survey[survey_par]
            else:
                for itm in select[itm_col]:
                    if not append_object:
                        self.scikit_object = {}
                    self.prepare(select=itm, column=itm_col, aggregate=aggregate)
                    self.train(select=itm)
                    self.predict_score(select=itm)
                    if plot:
                        self.plot(select=itm)

    def prepare(self, select="all", column=False, aggregate=False):

        if select != "all":
            df = self.df[self.df[column] == select]
        elif select == "all":
            df = self.df.copy()

        if aggregate:
            self.scikit_params["Xcolumns"] = [aggregate[0]]

        if self.scikit_params["handle_na"] == "fill":
            for col in df.columns:
                if df[col].dtype == "float64":
                    df[col].fillna(value=42.42, inplace=True)
                elif df[col].dtype == "int64":
                    df[col].fillna(value=4242, inplace=True)
                elif df[col].dtype == "object":
                    df[col].fillna(value="4242", inplace=True)
                elif df[col].dtype == "datetime64[ns]":
                    df[col].fillna(
                        value=datetime(1942, 4, 2).replace(tzinfo=None), inplace=True
                    )

        elif self.scikit_params["handle_na"] == "drop":
            df.dropna(subset=self.scikit_params["Xcolumns"], how="any", inplace=True)

        if len(df) > 0:
            self.scikit_object[select] = {}

            if self.scikit_params["outliers"] and len(df) > 1:
                inputs = list(self.scikit_params["outliers"].items())
                df = self.fit_outliers(df, inputs[0][0], inputs[0][1])

            if aggregate:
                xcol = [aggregate[0]]
                train_df, predict_df = self.train_predict_split(df=df)
                train_df, agg_lst = utilities.aggregate(
                    train_df,
                    self.scikit_params["ycolumn"][0],
                    aggregate[0],
                    aggregate[1],
                )
                X_predict_df = pd.DataFrame({xcol[0]: agg_lst})
            else:
                xcol = self.scikit_params["Xcolumns"]
                if self.scikit_params["drop_uniform"]:
                    df, cols = self.drop_uniform(df)
                    if len(cols) > 0:
                        self.scikit_object[select]["uniform_columns"] = cols
                        xcol = [
                            x for x in self.scikit_params["Xcolumns"] if x not in cols
                        ]

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
                enc = preprocessing.OrdinalEncoder()
                self.scikit_object[select]["X_encoder"] = enc.fit(X_df)
                X_df = self.scikit_object[select]["X_encoder"].transform(X_df)
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
                    scl = preprocessing.RobustScaler()
                else:
                    scl = preprocessing.StandardScaler()

                self.scikit_object[select]["scaler"] = scl.fit(X_df)
                X_df = self.scikit_object[select]["scaler"].transform(X_df)
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
                nrm = preprocessing.Normalizer()
                self.scikit_object[select]["normalizer"] = nrm.fit(X_df)
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
                    enc = preprocessing.LabelEncoder()
                    self.scikit_object[select]["y_encoder"] = enc.fit(
                        train_df[self.scikit_params["ycolumn"]]
                    )
                    self.scikit_object[select]["y"] = self.scikit_object[select][
                        "y_encoder"
                    ].transform(train_df[self.scikit_params["ycolumn"]])
                else:
                    self.scikit_object[select]["y"] = train_df[
                        self.scikit_params["ycolumn"]
                    ].to_numpy()

            if len(X_predict_df) > 0:
                self.scikit_object[select]["X_predict"] = pd.DataFrame(
                    data=X_predict_df, columns=xcol
                )
        else:
            self.scikit_object[select] = {select: None}

    def fit_outliers(self, df, which, threshold):

        try:
            return self.outlier_filter(df, which, threshold, None, self.scikit_params)
        except ValueError:
            s_frac = 0
            while s_frac <= 1:
                try:
                    s_frac += 0.05
                    return self.outlier_filter(
                        df, which, threshold, s_frac, self.scikit_params
                    )
                except ValueError:
                    continue
            else:
                return df

    @staticmethod
    def outlier_filter(df, which, threshold, s_frac, scikit_params):

        out = covariance.EllipticEnvelope(
            contamination=threshold, support_fraction=s_frac, random_state=42
        )
        subjects = df[scikit_params["ycolumn"]].to_numpy().reshape(-1, 1)
        result = out.fit_predict(subjects)
        df["outliers"] = result
        if which == "remove":
            return df[df["outliers"] == 1].drop(labels="outliers", axis=1)
        elif which == "select":
            return df[df["outliers"] == -1].drop(labels="outliers", axis=1)

    def train(self, select=False):

        method, _ = self.pre_qualify_lists(self.scikit_params["method"], [])
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

            for meth in method:
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

    def scikit_fit(self, X_train, y_train, method, base_estimator=False):

        if self.scikit_params["model_selection"] == "regress":
            return scikit_models.regression_methods.scikit_regression(
                X_train,
                y_train,
                method,
                self.scikit_params,
                base_estimator=base_estimator,
            )
        elif self.scikit_params["model_selection"] == "classify":
            return scikit_models.classification_methods.scikit_classification(
                X_train,
                y_train,
                method,
                self.scikit_params,
                base_estimator=base_estimator,
            )
        elif self.scikit_params["model_selection"] == "cluster":
            return scikit_models.clustering_methods.scikit_clustering(
                X_train, y_train, method, self.scikit_params
            )

        elif self.scikit_params["model_selection"] == "semi_supervised":
            return scikit_models.semi_supervised_methods.scikit_semi_supervised(
                X_train, y_train, method, self.scikit_params
            )

    @staticmethod
    def return_score(obj, X_true=None, y_true=None, y_pred=None, metric="mape"):

        try:
            if metric in ["r2", "score"]:
                return obj.score(X_true, y_true)
            elif metric == "score_samples":
                return obj.score_samples(X_true)
            elif metric == "mape":
                return metrics.mean_absolute_percentage_error(y_true, y_pred)
            elif metric == "mae":
                return metrics.mean_absolute_error(y_true, y_pred)
            elif metric == "mse":
                return metrics.mean_squared_error(y_true, y_pred)
            elif metric == "jaccard":
                return metrics.jaccard_score(y_true, y_pred)
            elif metric == "adjusted_rand":
                return metrics.cluster.adjusted_rand_score(y_true, y_pred)
        except ValueError:
            return False

    def predict_score(self, select=False):

        method, metric = self.pre_qualify_lists(
            self.scikit_params["method"], self.scikit_params["score_metric"]
        )

        self.scikit_object[select]["scores"] = {}
        self.scikit_object[select]["predictions"] = {}
        if all(
            map(
                list(self.scikit_object[select].keys()).__contains__,
                ["X_test", "y_test"],
            )
        ):
            for meth in method:
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
                                    "train": self.return_score(
                                        self.scikit_object[select][key][val],
                                        X_true=self.scikit_object[select]["X_train"],
                                        y_true=self.scikit_object[select]["y_train"],
                                        y_pred=self.scikit_object[select][
                                            "predictions"
                                        ][key][val]["train"],
                                        metric=metr,
                                    ),
                                    "test": self.return_score(
                                        self.scikit_object[select][key][val],
                                        X_true=self.scikit_object[select]["X_test"],
                                        y_true=self.scikit_object[select]["y_test"],
                                        y_pred=self.scikit_object[select][
                                            "predictions"
                                        ][key][val]["test"],
                                        metric=metr,
                                    ),
                                }
                                self.scores_dict["Var"].append(select)
                                self.scores_dict["score"].append(
                                    self.scikit_object[select]["scores"][key][val][metr]
                                )
                                self.scores_dict["method"].append(
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
                            "train": self.return_score(
                                self.scikit_object[select][meth],
                                X_true=self.scikit_object[select]["X_train"],
                                y_true=self.scikit_object[select]["y_train"],
                                y_pred=self.scikit_object[select]["predictions"][meth][
                                    "train"
                                ],
                                metric=metr,
                            ),
                            "test": self.return_score(
                                self.scikit_object[select][meth],
                                X_true=self.scikit_object[select]["X_test"],
                                y_true=self.scikit_object[select]["y_test"],
                                y_pred=self.scikit_object[select]["predictions"][meth][
                                    "test"
                                ],
                                metric=metr,
                            ),
                        }
                        self.scores_dict["Var"].append(select)
                        self.scores_dict["score"].append(
                            self.scikit_object[select]["scores"][meth][metr]
                        )
                        self.scores_dict["method"].append("_".join([meth, metr]))
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

    def plot(self, select):

        method, _ = self.pre_qualify_lists(self.scikit_params["method"], [])
        if self.scikit_object[select]["predictions"] and all(
            [x in self.scikit_object[select].keys() for x in ["y_test", "y_train"]]
        ):
            for meth in method:
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
                                        range(
                                            len(self.scikit_object[select]["y_train"])
                                        ),
                                        self.scikit_object[select]["y_train"],
                                        "o",
                                        label="Train points",
                                    )
                                    plt.plot(
                                        range(
                                            len(self.scikit_object[select]["y_train"])
                                        ),
                                        self.scikit_object[select]["predictions"][key][
                                            val
                                        ]["train"],
                                        "x",
                                        label="Predictions",
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
                                        range(
                                            len(self.scikit_object[select]["y_test"])
                                        ),
                                        self.scikit_object[select]["y_test"],
                                        "o",
                                        label="Test points",
                                    )
                                    plt.plot(
                                        range(
                                            len(self.scikit_object[select]["y_test"])
                                        ),
                                        self.scikit_object[select]["predictions"][key][
                                            val
                                        ]["test"],
                                        "x",
                                        label="Predictions",
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
                            range(len(self.scikit_object[select]["y_train"])),
                            self.scikit_object[select]["y_train"],
                            "o",
                            label="Train points",
                        )
                        plt.plot(
                            range(len(self.scikit_object[select]["y_train"])),
                            self.scikit_object[select]["predictions"][meth]["train"],
                            "x",
                            label="Predictions",
                        )
                        plt.title(select + ": " + meth + " (Train)")
                        plt.legend()
                        plt.subplot(122)
                        plt.plot(
                            range(len(self.scikit_object[select]["y_test"])),
                            self.scikit_object[select]["y_test"],
                            "o",
                            label="Test points",
                        )
                        plt.plot(
                            range(len(self.scikit_object[select]["y_test"])),
                            self.scikit_object[select]["predictions"][meth]["test"],
                            "x",
                            label="Predictions",
                        )
                        plt.title(select + ": " + meth + " (Test)")
                        plt.legend()
                        plt.show()

    def scatter(self, select):

        method, _ = self.pre_qualify_lists(self.scikit_params["method"], [])
        if self.scikit_object[select]["predictions"] and all(
            [x in self.scikit_object[select].keys() for x in ["y_test", "y_train"]]
        ):
            for meth in method:
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
                                    plt.scatter(
                                        self.scikit_object[select]["X_train"][
                                            self.scikit_object[select][
                                                "X_train"
                                            ].columns[0]
                                        ],
                                        self.scikit_object[select]["y_train"],
                                        label="Train points",
                                    )
                                    plt.scatter(
                                        self.scikit_object[select]["X_train"][
                                            self.scikit_object[select][
                                                "X_train"
                                            ].columns[0]
                                        ],
                                        self.scikit_object[select]["predictions"][key][
                                            val
                                        ]["train"],
                                        label="Predictions",
                                    )
                                    plt.title(
                                        select
                                        + ": "
                                        + "_".join([key, val])
                                        + " (Train)"
                                    )
                                    plt.legend()
                                    plt.subplot(122)
                                    plt.scatter(
                                        self.scikit_object[select]["X_test"][
                                            self.scikit_object[select][
                                                "X_test"
                                            ].columns[0]
                                        ],
                                        self.scikit_object[select]["y_test"],
                                        label="Test points",
                                    )
                                    plt.scatter(
                                        self.scikit_object[select]["X_test"][
                                            self.scikit_object[select][
                                                "X_test"
                                            ].columns[0]
                                        ],
                                        self.scikit_object[select]["predictions"][key][
                                            val
                                        ]["test"],
                                        label="Predictions",
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
                        plt.scatter(
                            self.scikit_object[select]["X_train"][
                                self.scikit_object[select]["X_train"].columns[0]
                            ],
                            self.scikit_object[select]["y_train"],
                            label="Train points",
                        )
                        plt.scatter(
                            self.scikit_object[select]["X_train"][
                                self.scikit_object[select]["X_train"].columns[0]
                            ],
                            self.scikit_object[select]["predictions"][meth]["train"],
                            label="Predictions",
                        )
                        plt.title(select + ": " + meth + " (Train)")
                        plt.legend()
                        plt.subplot(122)
                        plt.scatter(
                            self.scikit_object[select]["X_test"][
                                self.scikit_object[select]["X_test"].columns[0]
                            ],
                            self.scikit_object[select]["y_test"],
                            label="Test points",
                        )
                        plt.scatter(
                            self.scikit_object[select]["X_test"][
                                self.scikit_object[select]["X_test"].columns[0]
                            ],
                            self.scikit_object[select]["predictions"][meth]["test"],
                            label="Predictions",
                        )
                        plt.title(select + ": " + meth + " (Test)")
                        plt.legend()
                        plt.show()

    @staticmethod
    def drop_uniform(df):

        cols = []
        for ix in df.columns.values:
            if len(df.drop_duplicates([ix])[ix]) <= 1:
                cols.append(ix)
                df.drop(labels=[ix], axis=1, inplace=True)

        return df, cols

    @staticmethod
    def train_predict_split(df=None, date_column="", cost_column=""):

        predict_df_1 = df[df[date_column] <= 0]
        predict_df_2 = df[df[cost_column] <= 0]
        predict_df = predict_df_1.append(predict_df_2, ignore_index=True)

        train_df = df[df[date_column] > 0]
        train_df = train_df[train_df[cost_column] > 0]

        return train_df.drop_duplicates(), predict_df.drop_duplicates()

    @staticmethod
    def pre_qualify_lists(method, metric):

        if not isinstance(method, list):
            method = [method]

        for indx, meth in enumerate(method):
            if isinstance(meth, dict):
                for key in meth.keys():
                    if not isinstance(meth[key], list):
                        method[indx][key] = [meth[key]]

        if not isinstance(metric, list):
            metric = [metric]

        return method, metric

    def scores_to_df(self, save=False):

        self.scores_df = pd.DataFrame(
            {
                "Var": self.scores_dict["Var"],
                "train_score": [x["train"] for x in self.scores_dict["score"]],
                "test_score": [x["test"] for x in self.scores_dict["score"]],
            }
        )

        self.scores_df.sort_values(by="Var", ascending=False, inplace=True)
        self.scores_df["metric"] = [
            x.split("_")[-1] for x in self.scores_dict["method"]
        ]

        method_lst = []
        base_est_lst = []
        for x in self.scores_dict["method"]:
            splt = x.split("_")
            method_lst.append(splt[0])
            if len(splt) > 2:
                base_est_lst.append(splt[1])
            else:
                base_est_lst.append("")

        self.scores_df["method"] = method_lst
        self.scores_df["base_estimator"] = base_est_lst

        self.scores_df.astype(
            {
                "Var": "str",
                "train_score": "float64",
                "test_score": "float64",
                "metric": "str",
                "method": "str",
                "base_estimator": "str",
            },
            errors="ignore",
        )

        if save and isinstance(save, str):
            self.scores_df.to_csv(save, index=False)
