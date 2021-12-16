from statsmodels import api, tsa
import pandas as pd
import matplotlib.pyplot as plt

from . import utilities


class statsmodels:
    def __init__(
        self,
        df=None,
        formula=None,
        time=None,
        group=False,
        resample=False,
        time_range=None,
        exog=None,
        endog=None,
        freq="infer",
        stats_params={},
    ):

        if formula:
            if isinstance(endog, str):
                endog = [endog]
            if isinstance(exog, str):
                exog = [exog]
            df = df[list(set(endog) + set(exog))]
        else:
            df = df[[time, endog]]

            if df[time].dtype != "datetime64[ns]":
                df[time] = pd.to_datetime(
                    df[time], infer_datetime_format=True
                ).sort_values(ascending=True)

            if time_range:
                df = utilities.time_filter(df, time, (time_range[0], time_range[1]))
            if group:
                df, group_indx = utilities.group(df, endog, time, kind=group)
            if resample:
                new_df, df_indx = utilities.datetime_index(df, time, freq="infer")
                df, resamp_indx = utilities.resample(
                    new_df, endog, time, freq, kind=resample
                )

            try:
                new_df, df_indx = utilities.datetime_index(df, time, freq=freq)
                if not df_indx.freq:
                    try:
                        df, resamp_indx = utilities.resample(
                            new_df, endog, time, freq, kind=resample
                        )
                        df, df_indx = utilities.datetime_index(df, time, freq=freq)
                    except:
                        pass
                    if not df_indx.freq:
                        print("not able to assign freq to DatetimeIndex")
                else:
                    df = new_df.copy()
            except:
                df, df_indx = utilities.datetime_index(df, time, freq="infer")
                if not df_indx.freq:
                    print("not able to assign freq to DatetimeIndex")

            df.drop(time, axis=1, inplace=True)

        self.df = df.copy()

        s_params = {
            "order": (3, 2, 1),
            "lags": 3,
            "exog": None,
            "exog_oos": None,
            "model": "additive",
            "filt": None,
            "period": None,
            "two_sided": True,
            "extrapolate_trend": 0,
            "model_kwargs": None,
            "seasonal": 7,
            "trend": None,
            "low_pass": None,
            "seasonal_deg": 1,
            "trend_deg": 1,
            "low_pass_deg": 1,
            "robust": False,
            "seasonal_jump": 1,
            "trend_jump": 1,
            "low_pass_jump": 1,
            "steps": 1,
        }
        s_params.update(stats_params)
        self.statsmodels_params = s_params

        if formula is not None:
            if formula == "build":
                formula = self.build_formula(endog[0], exog)
            self.formula = formula

    def fit_formula(self, model="AutoReg", plot=False):

        self._stats_timeseries_formula(
            self.df, self.formula, model, self.statsmodels_params, plot
        )

    def fit(self, model="ARIMA", target="data", plot=False, s_params={}):

        if hasattr(self, "statsmodels_params"):
            self.statsmodels_params.update(s_params)
        else:
            self.statsmodels_params = s_params

        self.seasonal_decomposition = self.make_seasonal_decomp(
            self.df, self.statsmodels_params
        )
        self._stats_timeseries(self.df, model, target, self.statsmodels_params, plot)

    def forecast(
        self, target="data", type="STLForecast", model="ARIMA", s_params={}, plot=False
    ):

        if hasattr(self, "statsmodels_params"):
            self.statsmodels_params.update(s_params)
        else:
            self.statsmodels_params = s_params

        self.seasonal_decomposition = self.make_seasonal_decomp(
            self.df, self.statsmodels_params
        )
        if target == "seasonal":
            df = self.seasonal_decomposition.seasonal
        elif target == "trend":
            df = self.seasonal_decomposition.trend
        elif target == "resid":
            df = self.seasonal_decomposition.resid
        else:
            df = self.df

        if type == "predict":
            if hasattr(self, "fit"):
                self.forecast = self.fit.forecast(
                    steps=self.statsmodels_params["steps"]
                )

        elif type == "STLForecast":
            if model == "ARIMA":
                mod = tsa.arima.model.ARIMA

            fcast = tsa.api.STLForecast(
                df,
                mod,
                model_kwargs=self.statsmodels_params["model_kwargs"],
                period=self.statsmodels_params["period"],
                seasonal=self.statsmodels_params["seasonal"],
                trend=self.statsmodels_params["trend"],
                low_pass=self.statsmodels_params["low_pass"],
                seasonal_deg=self.statsmodels_params["seasonal_deg"],
                trend_deg=self.statsmodels_params["trend_deg"],
                low_pass_deg=self.statsmodels_params["low_pass_deg"],
                robust=self.statsmodels_params["robust"],
                seasonal_jump=self.statsmodels_params["seasonal_jump"],
                trend_jump=self.statsmodels_params["trend_jump"],
                low_pass_jump=self.statsmodels_params["low_pass_jump"],
            )

            self.forecast_fit = fcast.fit()
            self.forecast = self.forecast_fit.forecast(
                steps=self.statsmodels_params["steps"]
            )

            if plot:
                plt.plot(df, color="g", label="Data")
                plt.plot(self.forecast, color="r", label="Forecast")
                plt.legend()
                plt.show()

    @staticmethod
    def make_seasonal_decomp(df, s_params):

        return tsa.seasonal.seasonal_decompose(
            df,
            model=s_params["model"],
            filt=s_params["filt"],
            period=s_params["period"],
            two_sided=s_params["two_sided"],
            extrapolate_trend=s_params["extrapolate_trend"],
        )

    @staticmethod
    def build_formula(endog, exog):
        return "".join([endog, " ~ ", " ".join([f"{x} +" for x in exog]).strip(" +")])

    def _stats_timeseries_formula(self, df, formula, model, s_params, plot):

        if model == "AutoReg":
            smod = tsa.ar_model.AutoReg.from_formula(formula=formula, data=df)

        elif model == "ARIMA":
            smod = tsa.arima_model.ARIMA.from_formula(formula=formula, data=df)

        self._fit_predict(smod, s_params, plot)

    def _stats_timeseries(self, df, model, target, s_params, plot):

        if target == "seasonal":
            df = self.seasonal_decomposition.seasonal
        elif target == "trend":
            df = self.seasonal_decomposition.trend
        elif target == "resid":
            df = self.seasonal_decomposition.resid

        if model == "AutoReg":
            smod = tsa.ar_model.AutoReg(df, s_params["lags"])

        elif model == "SARIMAX":
            smod = api.tsa.statespace.SARIMAX(df, order=s_params["order"])

        elif model == "ARIMA":
            smod = tsa.arima.model.ARIMA(df, order=s_params["order"])

        self._fit_predict(smod, s_params, plot)

    def _fit_predict(self, smod, s_params, plot):

        self.fit = smod.fit()
        self.predict = self.fit.predict()
        self.summary = self.fit.summary()

        if plot:
            self.plot()

    def plot(self, df=None, label=None):

        plt.figure(figsize=(10, 6))
        if df is None:
            plt.plot(self.df, label="Data")
            plt.plot(self.predict, label="Prediction")
        else:
            if isinstance(df, (list, tuple)):
                for ind, x in enumerate(df):
                    plt.plot(x, label=label[ind])
            else:
                plt.plot(df, label=label)
        plt.legend()
        plt.show()

        if hasattr(self, "seasonal_decomposition"):
            plt.rc("figure", figsize=(10, 6))
            plt.rc("font", size=12)
            self.seasonal_decomposition.plot()
            plt.show()
