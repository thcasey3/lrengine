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
        interpolate="time",
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
                df, _ = utilities.group(df, endog, time, kind=group)
            if resample:
                new_df, _ = utilities.datetime_index(df, time, freq="infer")
                df, _ = utilities.resample(
                    new_df, endog, time, freq, interpolate, kind=resample
                )

            try:
                df, df_indx = utilities.datetime_index(df, time, freq=freq)
            except:
                df, flag = self.try_fix_freq(df, time)
                if flag:
                    print(f"your index matched '{flag}' freq, setting to {freq[-1]}")
                else:
                    print(
                        "Your index freq doesn't match any common frequencies, try using resample"
                    )
                    df, df_indx = utilities.datetime_index(df, time, freq="infer")

            if not df.index.freq:
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

    @staticmethod
    def try_fix_freq(df, time):

        flag = []
        for f in ["N", "U", "L", "S", "T", "H", "D", "W", "M", "Q", "Y"]:
            try:
                test_df, _ = utilities.datetime_index(df, time, freq=f)
                flag.append(f)
            except:
                continue

        if flag:
            df, _ = utilities.datetime_index(df, time, freq=flag[-1])

        return df, flag

    def fit_formula(self, model="AutoReg"):

        self._stats_timeseries_formula(
            self.df, self.formula, model, self.statsmodels_params
        )

    def fit(self, model="ARIMA", target="data", plot=False, s_params={}):

        if hasattr(self, "statsmodels_params"):
            self.statsmodels_params.update(s_params)
        else:
            self.statsmodels_params = s_params

        try:
            self.seasonal_decomposition = self.make_seasonal_decomp(
                self.df, self.statsmodels_params
            )
        except:
            print("seasonal decomposition failed.")
            pass
        if plot and hasattr(self, "seasonal_decomposition"):
            plt.rc("figure", figsize=(10, 6))
            plt.rc("font", size=12)
            self.seasonal_decomposition.plot()
            plt.show()

        if not hasattr(self, "seasonal_decomposition") or target not in [
            "seasonal",
            "seasonal",
            "resid",
        ]:
            df = self.df
        elif target == "seasonal":
            df = self.seasonal_decomposition.seasonal
        elif target == "trend":
            df = self.seasonal_decomposition.trend
        elif target == "resid":
            df = self.seasonal_decomposition.resid

        self._stats_timeseries(df, model, self.statsmodels_params)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df, label="Data")
            plt.plot(self.predict, label="Prediction")
            plt.legend()
            plt.show()

    def forecast(
        self, target="data", type="STLForecast", model="ARIMA", s_params={}, plot=False
    ):

        if hasattr(self, "statsmodels_params"):
            self.statsmodels_params.update(s_params)
        else:
            self.statsmodels_params = s_params

        try:
            self.seasonal_decomposition = self.make_seasonal_decomp(
                self.df, self.statsmodels_params
            )
        except:
            print("seasonal decomposition failed.")
            pass

        if type == "predict":
            if hasattr(self, "fit") and isinstance(target, (list, tuple, pd.DataFrame)):
                if isinstance(target, (list, tuple)):
                    self.forecast = self.fit.predict(start=target[0], end=target[1])
                elif isinstance(target, pd.DataFrame):
                    self.forecast = self.fit.predict(
                        start=target.index[0], end=target.index[-1]
                    )
                self.forecast_fit = self.fit
                self.forecast_summary = self.forecast_fit.summary()
                df = target
            else:
                raise TypeError(
                    "you must use fit first, and target must be a list, tuple, or pd.DateFrame"
                )

        elif type == "STLForecast":

            if not hasattr(self, "seasonal_decomposition") or target not in [
                "seasonal",
                "seasonal",
                "resid",
            ]:
                df = self.df
            elif target == "seasonal":
                df = self.seasonal_decomposition.seasonal
            elif target == "trend":
                df = self.seasonal_decomposition.trend
            elif target == "resid":
                df = self.seasonal_decomposition.resid

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
            self.forecast_summary = self.forecast_fit.summary()

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

    def _stats_timeseries_formula(self, df, formula, model, s_params):

        if model == "AutoReg":
            smod = tsa.ar_model.AutoReg.from_formula(formula=formula, data=df)

        elif model == "ARIMA":
            smod = tsa.arima_model.ARIMA.from_formula(formula=formula, data=df)

        self._fit_predict(smod, s_params)

    def _stats_timeseries(self, df, model, s_params):

        if model == "AutoReg":
            smod = tsa.ar_model.AutoReg(df, s_params["lags"])

        elif model == "SARIMAX":
            smod = api.tsa.statespace.SARIMAX(df, order=s_params["order"])

        elif model == "ARIMA":
            smod = tsa.arima.model.ARIMA(df, order=s_params["order"])

        self._fit_predict(smod, s_params)

    def _fit_predict(self, smod, s_params):

        self.fit = smod.fit()
        self.predict = self.fit.predict()
        self.summary = self.fit.summary()
