"""
statsmodels module, for using statsmodels with lsobject.frame
"""
from statsmodels import api, tsa
import pandas as pd
import matplotlib.pyplot as plt


class statsmodels:
    """
    class for using statsmodels with lsobject.frame

    Args:
        series (list): columns from frame to include in series
        exog (list): columns from frame to include in exog
    Returns:
         (object):  statsmodels_object
    """

    def __init__(
        self,
        formula=None,
        df=None,
        time=None,
        freq=None,
        endog=None,
        exog=None,
        model="AutoReg",
        stats_params={},
        plot=True,
    ):

        if isinstance(time, str) and isinstance(df, pd.DataFrame):
            if not df[time].dtype == "datetime64[ns]":
                try:
                    df[time] = pd.to_datetime(
                        df[time], infer_datetime_format=True, errors="coerce"
                    )
                except:
                    pass
            df.set_index(time, inplace=True)
            if freq is None:
                df.index = pd.DatetimeIndex(df.index, freq="infer")
            else:
                df.index = pd.DatetimeIndex(df.index, freq=freq)
            df.dropna(inplace=True)
        else:
            raise ValueError("you must specify the time column using time")

        if endog and not exog:
            if isinstance(endog, str):
                endog = [endog]
            exog = list(set(df.columns) - set(endog))
        elif not endog and not exog:
            raise ValueError("need to give at least the endog column")

        self.statsmodels_object = {}
        self.statsmodels_object["model"] = model
        self.statsmodels_object["endog"] = endog
        self.statsmodels_object["exog"] = exog
        self.statsmodels_object["df"] = df

        s_params = {
            "order": (0, 0, 0),
            "forecast_steps": 1,
            "lags": 3,
            "exog": df[exog].iloc[0],
            "exog_oos": df[exog].iloc[0],
        }
        s_params.update(stats_params)
        self.statsmodels_object["statsmodels_params"] = s_params

        if formula is not None:
            self.statsmodels_object["formula"] = formula
            self._stats_timeseries(formula, df, endog, exog, model, s_params, plot)
        else:
            self._stats_timeseries(df, endog, exog, model, s_params, plot)

    def _stats_timeseries(self, df, endog, exog, model, s_params, plot):

        if model == "AutoReg":
            smod = tsa.ar_model.AutoReg(df[endog], s_params["lags"], exog=df[exog])

        elif model == "SARIMAX":
            smod = api.tsa.statespace.SARIMAX(
                df[endog], df[exog], order=s_params["order"]
            )

        elif model == "ARIMA":
            smod = tsa.arima.model.ARIMA(df[endog], df[exog], order=s_params["order"])

        # elif model == "ThetaModel":
        #     smod = tsa.forecasting.theta.ThetaModel(data)

        self._fit_predict(smod, s_params, plot)

    def _stats_timeseries_formula(
        self, formula, data, endog, exog, model, s_params, plot
    ):
        pass

    def _fit_predict(self, smod, s_params, plot):

        self.statsmodels_object["fit"] = smod.fit()
        self.statsmodels_object["summary"] = self.statsmodels_object["fit"].summary()
        if s_params["exog_oos"] is not None:
            self.statsmodels_object["prediction"] = self.statsmodels_object[
                "fit"
            ].predict(exog_oos=s_params["exog_oos"])
        if s_params["exog"] is not None:
            self.statsmodels_object["forecast"] = self.statsmodels_object[
                "fit"
            ].forecast(steps=s_params["forecast_steps"], exog=s_params["exog"])

        if plot:
            self.plot()

    def plot(self, x=None, y=None, label=None):

        if x is None and y is None:
            plt.plot(
                self.statsmodels_object["df"].index,
                self.statsmodels_object["df"][self.statsmodels_object["endog"]],
                label="series",
                color="b",
            )
            if "prediction" in self.statsmodels_object.keys():
                plt.plot(
                    self.statsmodels_object["prediction"].index,
                    self.statsmodels_object["prediction"].values,
                    label="prediction",
                    color="r",
                )
            if "forecast" in self.statsmodels_object.keys():
                plt.plot(
                    self.statsmodels_object["forecast"].index,
                    self.statsmodels_object["forecast"].values,
                    label="forecast",
                    marker="x",
                    color="g",
                )
        else:
            if isinstance(y, dict):
                for key, val in y.items():
                    plt.plot(val.index, val.values, label=key)
            else:
                plt.plot(x, y, label=label)

        plt.legend()
        plt.show()
