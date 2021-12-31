from statsmodels import api, tsa, stats, graphics
from scipy.optimize import brute
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from . import config, utilities


class statsmodels:
    def __init__(
        self,
        df=None,
        forecast_df=None,
        time=None,
        endog=None,
        exog=None,
        update_config={},
    ):

        self.statsmodels_params = config.TIMESERIES_PARAMS
        if update_config:
            self.statsmodels_params.update(update_config)

        self.stats_object = {}
        self.stats_object["time"] = time
        self.stats_object["endog"] = endog
        self.stats_object["exog"] = exog

        self.df = df
        if exog is not None:
            self.forecast_df = forecast_df
        else:
            self.forecast_df = None

    def forecast(
        self,
        select="all",
        model="ARIMA",
        steps=3,
        append_object=False,
        survey=False,
        plots=False,
    ):

        self.scores_dict = {"subject": [], "score": [], "method": []}

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
                    self.stats_object = {
                        key: item
                        for key, item in self.stats_object.items()
                        if key in ["time", "endog", "exog", "formula"]
                    }
                for par in survey[survey_par]:
                    self.statsmodels_params.update({survey_par: par})
                    self.prepare(select=itm, column=itm_col)
                    self.run_fit(select=itm, model=model, test_steps=steps)
                    self.run_forecast(select=itm, model=model, steps=steps)
                    self.invert_log_transforms(select=itm)
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
                    self.stats_object = {
                        key: item
                        for key, item in self.stats_object.items()
                        if key in ["time", "endog", "exog", "formula"]
                    }
                self.prepare(select=itm, column=itm_col)
                self.run_fit(select=itm, model=model, test_steps=steps)
                self.run_forecast(select=itm, model=model, steps=steps)
                self.invert_log_transforms(select=itm)
                if plots:
                    self.plots(select=itm)

    def prepare(self, select="all", column=None):

        self.stats_object[select] = {}
        self.stats_object[select]["statistics"] = {}
        self.stats_object[select]["results"] = {}

        if select != "all":
            df = self.df[self.df[column] == select]
        elif select == "all":
            df = self.df.copy()

        if not df[self.stats_object["endog"]].dtype == "float64":
            df = df.astype({self.stats_object["endog"]: "float64"})

        if self.statsmodels_params["scale"]:
            vals, self.y_scaler = utilities.scaler(
                df[self.stats_object["endog"]].values.reshape(-1, 1), "robust"
            )
            df[self.stats_object["endog"]] = vals.reshape(-1)

        if self.statsmodels_params["normalize"]:
            vals, self.y_normalizer = utilities.normalizer(
                df[self.stats_object["endog"]].values.reshape(-1, 1), "minmax"
            )
            df[self.stats_object["endog"]] = vals.reshape(-1)

        if self.statsmodels_params["outliers"]:
            inputs = list(self.statsmodels_params["outliers"].items())
            df = utilities.fit_outliers(
                df, inputs[0][0], inputs[0][1], self.stats_object["endog"]
            )

        if self.stats_object["exog"] is not None:
            if self.forecast_df is not None:
                if select != "all":
                    foredf = self.forecast_df[self.forecast_df[column] == select]
                elif select == "all":
                    foredf = self.forecast_df.copy()
                df["past_or_future"] = "past"
                foredf["past_or_future"] = "future"
                df = df.append(foredf, ignore_index=True)

            if self.statsmodels_params["encode"]:
                values_exog, self.stats_object[select]["X_encoder"] = utilities.encoder(
                    df[self.stats_object["exog"]], "ordinal"
                )
            else:
                values_exog = df[self.stats_object["exog"]].values
            if self.statsmodels_params["scale"]:
                values_scl, self.X_scaler = utilities.scaler(values_exog, "robust")
            else:
                values_scl = values_exog

            exog_df = pd.DataFrame(data=values_scl, columns=self.stats_object["exog"])
            if self.forecast_df is not None:
                exog_df = exog_df.reset_index(drop=True).join(df[["past_or_future"]])

            df = (
                df[[self.stats_object["time"], self.stats_object["endog"]]]
                .reset_index(drop=True)
                .join(exog_df)
            )
            if self.forecast_df is not None:
                fore_df = df[df["past_or_future"] == "future"].drop(
                    labels=["past_or_future", self.stats_object["endog"]], axis=1
                )
                (
                    self.stats_object[select]["horizon_df"],
                    _,
                ) = utilities.prepare_timeseries(
                    fore_df,
                    self.stats_object["time"],
                    self.statsmodels_params["time_range"],
                    self.statsmodels_params["group"],
                    self.statsmodels_params["aggregate"],
                    self.statsmodels_params["resample"],
                    self.statsmodels_params["freq"],
                    self.statsmodels_params["interpolate"],
                )
                df = df[df["past_or_future"] == "past"].drop(
                    labels=["past_or_future"], axis=1
                )
        else:
            df = df[[self.stats_object["time"], self.stats_object["endog"]]]

        df, time_index = utilities.prepare_timeseries(
            df,
            self.stats_object["time"],
            self.statsmodels_params["time_range"],
            self.statsmodels_params["group"],
            self.statsmodels_params["aggregate"],
            self.statsmodels_params["resample"],
            self.statsmodels_params["freq"],
            self.statsmodels_params["interpolate"],
        )

        self.stats_object[select]["results"][
            "seasonal_decomposition"
        ] = self.make_seasonal_decomp(
            df[self.stats_object["endog"]], self.statsmodels_params
        )

        df[self.stats_object["endog"]] = self.log_transform(
            select, df[self.stats_object["endog"]], self.statsmodels_params["target"]
        )
        df.dropna(inplace=True)

        self.stats_object[select]["time"] = df.index
        self.stats_object[select]["endog_df"] = df[[self.stats_object["endog"]]]

        self.stats_object[select]["statistics"]["acf"] = tsa.stattools.acf(
            self.stats_object[select]["endog_df"].values
        )
        self.stats_object[select]["statistics"]["pacf"] = tsa.stattools.pacf(
            self.stats_object[select]["endog_df"].values
        )

        if self.stats_object["exog"] is not None:
            self.stats_object[select]["exog_df"] = df[self.stats_object["exog"]]
            self.stats_object[select]["df"] = self.stats_object[select][
                "endog_df"
            ].join(self.stats_object[select]["exog_df"])
        else:
            self.stats_object[select]["df"] = self.stats_object[select]["endog_df"]
            self.stats_object[select]["exog_df"] = None

    def plots(self, select="all"):

        print(f"adfuller pvalue: {self.stats_object[select]['statistics']['adfuller']}")
        if "best_order" in self.stats_object[select]["results"].keys():
            print(
                f"order (p,d,q): {self.stats_object[select]['results']['best_order']} was found using scipy.brute"
            )
        else:
            print(f"order (p,d,q): {self.statsmodels_params['order']} was used")
        print(
            f"durbin_watson result: {self.stats_object[select]['statistics']['durbin_watson']}"
        )

        fit_met = self.stats_object[select]["statistics"][
            f"fit_{self.statsmodels_params['metric']}"
        ]
        print(f"Train fit {self.statsmodels_params['metric']}: {fit_met}")

        test_met = self.stats_object[select]["statistics"][
            f"test_prediction_{self.statsmodels_params['metric']}"
        ]
        print(f"Test prediction {self.statsmodels_params['metric']}: {test_met}")

        if (
            f"forecast_fit_{self.statsmodels_params['metric']}"
            in self.stats_object[select]["statistics"].keys()
        ):
            fore_met = self.stats_object[select]["statistics"][
                f"forecast_fit_{self.statsmodels_params['metric']}"
            ]
            print(f"Forecast fit {self.statsmodels_params['metric']}: {fore_met}")

        plt.rc("figure", figsize=(10, 8))
        plt.rc("font", size=9)

        plt.subplot(4, 1, 1)
        plt.plot(
            self.stats_object[select]["results"]["seasonal_decomposition"].observed
        )
        plt.ylabel(f"{self.stats_object['endog']}")
        plt.title(f"{select}")
        plt.subplot(4, 1, 2)
        plt.plot(self.stats_object[select]["results"]["seasonal_decomposition"].trend)
        plt.ylabel("Trend")
        plt.subplot(4, 1, 3)
        plt.plot(
            self.stats_object[select]["results"]["seasonal_decomposition"].seasonal
        )
        plt.ylabel("Seasonal")
        plt.subplot(4, 1, 4)
        plt.plot(self.stats_object[select]["results"]["seasonal_decomposition"].resid)
        plt.ylabel("Residuals")

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        fig = graphics.tsaplots.plot_acf(
            self.stats_object[select]["endog_df"],
            alpha=self.statsmodels_params["alpha"],
            ax=ax1,
        )
        plt.title(f"{select} Autocorrelation")
        ax2 = fig.add_subplot(122)
        graphics.tsaplots.plot_pacf(
            self.stats_object[select]["endog_df"],
            alpha=self.statsmodels_params["alpha"],
            ax=ax2,
        )
        plt.show()

        if "forecast" in self.stats_object[select]["results"].keys():
            plt.subplot(2, 1, 1)
        plt.plot(self.stats_object[select]["endog_df"], color="black", label="Train")

        plt.plot(
            self.stats_object[select]["results"]["fit"],
            color="#a80f8c",
            label="Prediction",
        )
        plt.fill_between(
            self.stats_object[select]["statistics"]["fit_conf_int"].index,
            self.stats_object[select]["statistics"]["fit_conf_int"].values[:, 0],
            self.stats_object[select]["statistics"]["fit_conf_int"].values[:, 1],
            alpha=0.15,
            color="#a80f8c",
            label=f"Fit {int(100-(100*self.statsmodels_params['alpha']))} % CI",
        )

        plt.plot(
            self.stats_object[select]["results"]["test_prediction"],
            color="green",
            label="Test prediction",
        )
        plt.fill_between(
            self.stats_object[select]["statistics"]["test_conf_int"].index,
            self.stats_object[select]["statistics"]["test_conf_int"].values[:, 0],
            self.stats_object[select]["statistics"]["test_conf_int"].values[:, 1],
            alpha=0.15,
            color="green",
            label=f"Test {int(100-(100*self.statsmodels_params['alpha']))} % CI",
        )
        plt.title(f"{select} Test {self.statsmodels_params['metric']}: {test_met}")
        plt.legend(loc="best")

        if "forecast" in self.stats_object[select]["results"].keys():
            plt.subplot(2, 1, 2)
            plt.plot(
                self.stats_object[select]["endog_df"],
                color="black",
                label=self.stats_object["endog"],
            )
            plt.plot(
                self.stats_object[select]["results"]["forecast"],
                color="#a80f8c",
                label="Forecast",
            )
            plt.fill_between(
                self.stats_object[select]["statistics"]["forecast_conf_int"].index,
                self.stats_object[select]["statistics"]["forecast_conf_int"].values[
                    :, 0
                ],
                self.stats_object[select]["statistics"]["forecast_conf_int"].values[
                    :, 1
                ],
                alpha=0.15,
                color="#a80f8c",
                label=f"Forecast {int(100-(100*self.statsmodels_params['alpha']))} % CI",
            )
            plt.legend(loc="best")

        plt.show()

    def run_fit(self, select="all", model="ARIMA", test_steps=3, s_params={}):

        if hasattr(self, "statsmodels_params"):
            self.statsmodels_params.update(s_params)
        else:
            self.statsmodels_params = s_params

        train_df = self.stats_object[select]["df"].iloc[:-test_steps]
        train_time = train_df.index
        test_df = self.stats_object[select]["df"].iloc[-test_steps:]
        if self.stats_object[select]["exog_df"] is None:
            self.stats_object[select]["results"]["fit_model"] = self.build_model(
                select=select,
                time=train_time,
                endogenous=train_df,
                exogenous=None,
                model=model,
                spars=self.statsmodels_params,
            )
            train_endog = train_df
            test_endog = test_df
            test_exog = None
        else:
            train_endog = train_df[[self.stats_object["endog"]]]
            train_exog = train_df[self.stats_object["exog"]]
            self.stats_object[select]["results"]["fit_model"] = self.build_model(
                select=select,
                time=train_time,
                endogenous=train_endog,
                exogenous=train_exog,
                model=model,
                spars=self.statsmodels_params,
            )
            test_endog = test_df[[self.stats_object["endog"]]]
            test_exog = test_df[self.stats_object["exog"]]

        (
            self.stats_object[select]["results"]["fit_results"],
            self.stats_object[select]["results"]["fit"],
            self.stats_object[select]["results"]["fit_summary"],
        ) = self.get_fit_prediction(
            self.stats_object[select]["results"]["fit_model"],
            train_endog.index[0],
            train_endog.index[-1],
            model,
        )
        self.stats_object[select]["statistics"]["fit_conf_int"] = self.stats_object[
            select
        ]["results"]["fit"].conf_int(alpha=self.statsmodels_params["alpha"])

        if test_exog is not None:
            pred = self.get_forecast(
                self.stats_object[select]["results"]["fit_results"],
                test_steps,
                test_exog,
            )
        else:
            pred = self.get_forecast(
                self.stats_object[select]["results"]["fit_results"], test_steps, None
            )
        self.stats_object[select]["results"]["test_prediction"] = pred.predicted_mean
        self.stats_object[select]["statistics"]["test_conf_int"] = pred.conf_int(
            alpha=self.statsmodels_params["alpha"]
        )
        try:
            self.stats_object[select]["statistics"][
                f"fit_{self.statsmodels_params['metric']}"
            ] = utilities.return_metric(
                None,
                y_true=train_endog.values,
                y_pred=self.stats_object[select]["results"][
                    "fit"
                ].predicted_mean.values,
                metric=self.statsmodels_params["metric"],
            )
            self.stats_object[select]["statistics"][
                f"test_prediction_{self.statsmodels_params['metric']}"
            ] = utilities.return_metric(
                None,
                y_true=test_endog.values,
                y_pred=self.stats_object[select]["results"]["test_prediction"].values,
                metric=self.statsmodels_params["metric"],
            )
            self.scores_dict["subject"].append(select)
            self.scores_dict["score"].append(
                {
                    "train": self.stats_object[select]["statistics"][
                        f"fit_{self.statsmodels_params['metric']}"
                    ],
                    "test": self.stats_object[select]["statistics"][
                        f"test_prediction_{self.statsmodels_params['metric']}"
                    ],
                }
            )
            self.scores_dict["method"].append(model)
        except:
            self.stats_object[select]["statistics"][
                f"test_prediction_{self.statsmodels_params['metric']}"
            ] = False

    def run_forecast(
        self,
        select="all",
        steps=1,
        model="ARIMA",
        s_params={},
    ):

        if hasattr(self, "statsmodels_params"):
            self.statsmodels_params.update(s_params)
        else:
            self.statsmodels_params = s_params

        if "fit_model" not in self.stats_object[select]["results"].keys():
            raise TypeError("you must use fit before you can forecast")

        if self.stats_object[select]["exog_df"] is None:
            self.stats_object[select]["results"]["forecast_model"] = self.stats_object[
                select
            ]["results"]["fit_model"].clone(
                self.stats_object[select]["df"], dates=self.stats_object[select]["time"]
            )
        else:
            self.stats_object[select]["results"]["forecast_model"] = self.stats_object[
                select
            ]["results"]["fit_model"].clone(
                self.stats_object[select]["endog_df"],
                exog=self.stats_object[select]["exog_df"],
                dates=self.stats_object[select]["time"],
            )

        (
            self.stats_object[select]["results"]["forecast_fit_results"],
            fit,
            _,
        ) = self.get_fit_prediction(
            self.stats_object[select]["results"]["forecast_model"],
            self.stats_object[select]["endog_df"].index[0],
            self.stats_object[select]["endog_df"].index[-1],
            model,
        )
        self.stats_object[select]["results"]["forecast_fit"] = fit.predicted_mean
        self.stats_object[select]["statistics"]["forecast_fit_conf_int"] = fit.conf_int(
            alpha=self.statsmodels_params["alpha"]
        )
        self.stats_object[select]["statistics"][
            f"forecast_fit_{self.statsmodels_params['metric']}"
        ] = utilities.return_metric(
            None,
            y_true=self.stats_object[select]["endog_df"].values,
            y_pred=self.stats_object[select]["results"]["forecast_fit"].values,
            metric=self.statsmodels_params["metric"],
        )

        if "horizon_df" in self.stats_object[select].keys():
            self.stats_object[select]["results"]["forecast"] = self.get_forecast(
                self.stats_object[select]["results"]["forecast_fit_results"],
                len(self.stats_object[select]["horizon_df"]),
                self.stats_object[select]["horizon_df"],
            )
        else:
            if self.stats_object[select]["exog_df"] is not None:
                self.stats_object[select]["results"]["forecast"] = self.get_forecast(
                    self.stats_object[select]["results"]["forecast_fit_results"],
                    steps,
                    self.stats_object[select]["exog_df"].iloc[:steps],
                )
            else:
                self.stats_object[select]["results"]["forecast"] = self.get_forecast(
                    self.stats_object[select]["results"]["forecast_fit_results"],
                    steps,
                    None,
                )

        self.stats_object[select]["statistics"][
            "forecast_conf_int"
        ] = self.stats_object[select]["results"]["forecast"].conf_int(
            alpha=self.statsmodels_params["alpha"]
        )
        self.stats_object[select]["results"]["forecast_summary"] = self.stats_object[
            select
        ]["results"]["forecast_fit_results"].summary()

    def get_fit_prediction(self, smod, start, stop, model):
        if model == "SARIMAX":
            fit = smod.fit(disp=False)
        else:
            fit = smod.fit()
        prediction = fit.get_prediction(start=start, stop=stop)
        summary = fit.summary()

        return fit, prediction, summary

    def get_forecast(self, fit, steps, exog):
        if exog is not None:
            return fit.get_forecast(steps=steps, exog=exog)
        else:
            return fit.get_forecast(steps=steps)

    @staticmethod
    def get_d(d, target, series):
        if d[0] == 0:
            adf = tsa.stattools.adfuller(series)[1]
        else:
            adf = tsa.stattools.adfuller(series.diff(d[0]).dropna())[1]
        return target + adf

    @staticmethod
    def get_pq(pq, d, sorder, trend, endog, exog, time, model, ic):
        if model == "ARIMA":
            result = tsa.arima.model.ARIMA(
                endog,
                exog=exog,
                dates=time,
                order=(pq[0], d, pq[1]),
                seasonal_order=sorder,
                trend=trend,
                enforce_invertibility=False,
            ).fit()
        elif model == "SARIMAX":
            result = api.tsa.statespace.SARIMAX(
                endog,
                exog=exog,
                dates=time,
                order=(pq[0], d, pq[1]),
                seasonal_order=sorder,
                trend=trend,
                enforce_invertibility=False,
            ).fit(disp=False)

        if ic == "aic":
            return result.aic
        elif ic == "aicc":
            return result.aicc
        elif ic == "bic":
            return result.bic
        elif ic == "hqic":
            return result.hqic
        elif ic == "mae":
            return result.mae
        elif ic == "mse":
            return result.mse
        elif ic == "sse":
            return result.sse

    @staticmethod
    def get_seasonal_order(sorder, order, trend, endog, exog, time, model, ic):
        if model == "ARIMA":
            result = tsa.arima.model.ARIMA(
                endog,
                exog=exog,
                dates=time,
                order=order,
                seasonal_order=sorder,
                trend=trend,
                enforce_invertibility=False,
            ).fit()
        elif model == "SARIMAX":
            result = api.tsa.statespace.SARIMAX(
                endog,
                exog=exog,
                dates=time,
                order=order,
                seasonal_order=sorder,
                trend=trend,
                enforce_invertibility=False,
            ).fit(disp=False)

        if ic == "aic":
            return result.aic
        elif ic == "aicc":
            return result.aicc
        elif ic == "bic":
            return result.bic
        elif ic == "hqic":
            return result.hqic
        elif ic == "mae":
            return result.mae
        elif ic == "mse":
            return result.mse
        elif ic == "sse":
            return result.sse

    @staticmethod
    def get_trend(t_list, order, sorder, endog, exog, time, model, ic):
        result = []
        for x in t_list:
            if model == "ARIMA":
                mod = tsa.arima.model.ARIMA(
                    endog,
                    exog=exog,
                    dates=time,
                    order=order,
                    seasonal_order=sorder,
                    trend=x,
                    enforce_invertibility=False,
                ).fit()
            elif model == "SARIMAX":
                mod = api.tsa.statespace.SARIMAX(
                    endog,
                    exog=exog,
                    dates=time,
                    order=order,
                    seasonal_order=sorder,
                    trend=x,
                    enforce_invertibility=False,
                ).fit(disp=False)

            if ic == "aic":
                result.append(mod.aic)
            elif ic == "aicc":
                result.append(mod.aicc)
            elif ic == "bic":
                result.append(mod.bic)
            elif ic == "hqic":
                result.append(mod.hqic)
            elif ic == "mae":
                result.append(mod.mae)
            elif ic == "mse":
                result.append(mod.mse)
            elif ic == "sse":
                result.append(mod.sse)

        return t_list[np.argmin(result)]

    def build_model(
        self, select, endogenous, exogenous=None, time=None, model="ARIMA", spars={}
    ):

        if spars["opt_order"]:
            print("Optimizing order...")
            d_list = (
                slice(spars["pdq_limits"][1][0], spars["pdq_limits"][1][1] + 1, 1),
                slice(0, 1, 1),
            )
            result_d = brute(
                self.get_d,
                d_list,
                args=(0, endogenous),
                finish=None,
            )
            pq_list = (
                slice(spars["pdq_limits"][0][0], spars["pdq_limits"][0][1] + 1, 1),
                slice(spars["pdq_limits"][2][0], spars["pdq_limits"][2][1] + 1, 1),
            )
            result_pq = brute(
                self.get_pq,
                pq_list,
                args=(
                    int(result_d[0]),
                    spars["seasonal_order"],
                    spars["trend"],
                    endogenous,
                    exogenous,
                    time,
                    model,
                    spars["ic"],
                ),
                finish=None,
            )
            self.stats_object[select]["results"]["best_order"] = (
                int(result_pq[0]),
                int(result_d[0]),
                int(result_pq[1]),
            )
            use_order = self.stats_object[select]["results"]["best_order"]

        else:
            use_order = spars["order"]

        if spars["opt_seasonal_order"]:
            print("Optimizing seasonal order...")
            sorders_list = (
                slice(spars["PDQs_limits"][0][0], spars["PDQs_limits"][0][1] + 1, 1),
                slice(spars["PDQs_limits"][1][0], spars["PDQs_limits"][1][1] + 1, 1),
                slice(spars["PDQs_limits"][2][0], spars["PDQs_limits"][2][1] + 1, 1),
                slice(spars["PDQs_limits"][3][0], spars["PDQs_limits"][3][1] + 1, 2),
            )
            result = brute(
                self.get_seasonal_order,
                sorders_list,
                args=(
                    use_order,
                    spars["trend"],
                    endogenous,
                    exogenous,
                    time,
                    model,
                    spars["ic"],
                ),
                finish=None,
            )
            self.stats_object[select]["results"]["best_seasonal_order"] = (
                int(result[0]),
                int(result[1]),
                int(result[2]),
                int(result[3]),
            )
            season_order = self.stats_object[select]["results"]["best_seasonal_order"]

        else:
            season_order = spars["seasonal_order"]

        if spars["opt_trend"]:
            print("Optimizing trend...")
            self.stats_object[select]["results"]["best_trend"] = self.get_trend(
                spars["trend_list"],
                use_order,
                season_order,
                endogenous,
                exogenous,
                time,
                model,
                spars["ic"],
            )
            use_trend = self.stats_object[select]["results"]["best_trend"]

        else:
            use_trend = spars["trend"]

        if model == "SARIMAX":
            smod = api.tsa.statespace.SARIMAX(
                endogenous,
                exog=exogenous,
                dates=time,
                order=use_order,
                seasonal_order=season_order,
                trend=use_trend,
            )
            self.test_correlation(select, smod, model)

        elif model == "ARIMA":
            smod = tsa.arima.model.ARIMA(
                endogenous,
                exog=exogenous,
                dates=time,
                order=use_order,
                seasonal_order=season_order,
                trend=use_trend,
            )
            self.test_correlation(select, smod, model)

        return smod

    @staticmethod
    def make_transform(series, target, log_lam):

        if target == "log":
            if any([x < 0 for x in series]):
                lam = min(series) + log_lam
            elif any([x == 0 for x in series]):
                lam = log_lam
            else:
                lam = 0
            return (
                "log",
                np.log(series + lam),
                tsa.stattools.adfuller(np.log(series + lam))[1],
                lam,
            )
        else:
            return ("data", series, tsa.stattools.adfuller(series)[1], log_lam)

    def log_transform(self, select, series, target):

        if target == "infer":
            pval_lst = []
            for x in ["data", "log"]:
                try:
                    tup = self.make_transform(
                        series, x, self.statsmodels_params["log_lambda"]
                    )
                    if not np.isnan(tup[2]):
                        pval_lst.append(tup)
                except:
                    continue
            if not pval_lst:
                raise ValueError(
                    "adfuller pvalue could not be determined for any target method, try dropna() on input data"
                )
            else:
                best_lst = pval_lst[np.argmin(np.array([x[2] for x in pval_lst]))]
                self.stats_object[select]["statistics"]["adfuller"] = {
                    item[0]: item[2] for item in pval_lst
                }
        elif target == "log":
            best_lst = self.make_transform(
                series, "log", self.statsmodels_params["log_lambda"]
            )
        else:
            best_lst = self.make_transform(series, "data", 0)
        self.stats_object[select]["statistics"]["adfuller"] = {best_lst[0]: best_lst[2]}

        self.stats_object[select]["statistics"]["data_version"] = best_lst[0]
        self.stats_object[select]["results"]["log_lambda"] = best_lst[3]

        return best_lst[1]

    def invert_log_transforms(self, select="all"):

        self.stats_object[select]["endog_df"] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["endog_df"][
                self.stats_object["endog"]
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["endog_df"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        )
        self.stats_object[select]["results"]["forecast"] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["results"][
                "forecast"
            ].predicted_mean.values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["results"]["forecast"].predicted_mean.index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        )
        self.stats_object[select]["df"][
            self.stats_object["endog"]
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["df"][self.stats_object["endog"]].values.reshape(
                -1
            ),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["df"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["forecast_fit_conf_int"][
            f"lower {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["forecast_fit_conf_int"][
                f"lower {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["forecast_fit_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["forecast_fit_conf_int"][
            f"upper {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["forecast_fit_conf_int"][
                f"upper {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["forecast_fit_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["forecast_conf_int"][
            f"lower {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["forecast_conf_int"][
                f"lower {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["forecast_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["forecast_conf_int"][
            f"upper {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["forecast_conf_int"][
                f"upper {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["forecast_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values

        self.stats_object[select]["results"]["fit"] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["results"]["fit"].predicted_mean.values.reshape(
                -1
            ),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["results"]["fit"].predicted_mean.index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        )

        self.stats_object[select]["results"][
            "test_prediction"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["results"]["test_prediction"].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["results"]["test_prediction"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        )
        self.stats_object[select]["statistics"]["fit_conf_int"][
            f"lower {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["fit_conf_int"][
                f"lower {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["fit_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["fit_conf_int"][
            f"upper {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["fit_conf_int"][
                f"upper {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["fit_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["test_conf_int"][
            f"lower {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["test_conf_int"][
                f"lower {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["test_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values
        self.stats_object[select]["statistics"]["test_conf_int"][
            f"upper {self.stats_object['endog']}"
        ] = utilities.recover_transform(
            self.stats_object[select]["statistics"]["data_version"],
            self.stats_object[select]["statistics"]["test_conf_int"][
                f"upper {self.stats_object['endog']}"
            ].values.reshape(-1),
            self.stats_object[select]["results"]["log_lambda"],
            None,
            None,
            self.stats_object[select]["statistics"]["test_conf_int"].index,
            None,
            self.stats_object["time"],
            self.stats_object["endog"],
        ).values

    def test_correlation(self, select, smod, model):

        if model == "SARIMAX":
            fit = smod.fit(disp=False)
        else:
            fit = smod.fit()
        self.stats_object[select]["statistics"][
            "durbin_watson"
        ] = stats.stattools.durbin_watson(fit.resid.values)

    @staticmethod
    def make_seasonal_decomp(series, s_params):

        return tsa.seasonal.seasonal_decompose(
            series,
            model=s_params["model"],
            filt=s_params["filt"],
            period=s_params["period"],
            two_sided=s_params["two_sided"],
            extrapolate_trend=s_params["extrapolate_trend"],
        )

    def build_formula(self):
        if isinstance(self.stats_object["exog"], str):
            self.stats_object["exog"] = [self.stats_object["exog"]]
        return "".join(
            [
                self.stats_object["endog"],
                " ~ ",
                " ".join([f"{x} +" for x in self.stats_object["exog"]]).strip(" +"),
            ]
        )

    def scores_to_df(self, save=False):

        self.scores_df = utilities.scores_dict_to_df(self.scores_dict)

        self.scores_df["method"] = self.scores_dict["method"]

        self.scores_df.sort_values(by="subject", ascending=False, inplace=True)

        if save and isinstance(save, str):
            self.scores_df.to_csv(save, index=False)
