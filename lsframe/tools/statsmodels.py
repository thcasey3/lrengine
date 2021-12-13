"""
statsmodels module, for using statsmodels with lsobject.frame
"""
from statsmodels import api, tsa, regression, miscmodels, multivariate

_stats_params = {
    "fit_method": "bfgs",
    "disp": False,
    "constraints": None,
    "weights": None,
    "sigma": None,
    "window": 60,
    "min_nobs": 12,
    "expanding": True,
    "distr": "probit",
    "xvals": None,
    "old_names": False,
    "cov_type": "HC0",
}


class statsmodels:
    """
    class for using statsmodels with lsobject.frame

    Args:
        endog (list): columns from frame to include in endog
        exog (list): columns from frame to include in exog
    Returns:
         (object):  statsmodels_object
    """

    def __init__(
        self, endog, exog, formula="", data=None, model="OLS", stats_params={}
    ):

        s_params = _stats_params.update(stats_params)

        self.statsmodels_object = {}

        self._stats(self, endog, exog, model, s_params)

        return self.statsmodels_object

    def _stats_linear_regression(self, endog, exog, model, s_params):

        if model == "OLS":
            smod = api.sm.OLS(endog, exog)
        elif model == "WLS":
            smod = api.sm.WLS(endog, exog, weights=s_params["weights"])
        elif model == "GLS":
            smod = api.sm.GLS(endog, exog, sigma=s_params["sigma"])
        elif model == "GLSAR":
            smod = api.sm.GLSAR(endog, exog)
        elif model == "RecursiveLS":
            smod = api.sm.RecursiveLS(endog, exog, constraints=s_params["constraints"])
        elif model == "RollingOLS":
            smod = regression.rolling.RollingOLS(
                endog,
                exog,
                window=s_params["window"],
                min_nobs=s_params["min_nobs"],
                expanding=s_params["expanding"],
            )

        self._fit_predict(smod, s_params)

    def _stats_discrete_choice(self, endog, exog, model, s_params):

        if model == "OrderedModel":
            smod = miscmodels.ordinal_model.OrderedModel(
                endog, exog, distr=s_params["distr"]
            )

        self._fit_predict(smod, s_params)

    def _stats_nonparametric(self, endog, exog, model, s_params):

        if model == "LOWESS":
            smod = api.sm.nonparametric.lowess(
                exog=exog, endog=endog, xvals=s_params["xvals"]
            )

        self._fit_predict(smod, s_params)

    def _stats_timeseries(self, data, model, s_params):

        if model == "AutoReg":

            dp = tsa.deterministic.DeterministicProcess(
                data.index, constant=True, period=12, fourier=2
            )
            smod = tsa.ar_model.AutoReg(
                data, 2, trend="n", seasonal=False, deterministic=dp
            )

            smod = tsa.ar_model.AutoReg(
                data, s_params["autoreg_num"], old_names=s_params["old_names"]
            )

            sel = tsa.ar_model.ar_select_order(
                data, s_params["autoreg_num"], old_names=s_params["old_names"]
            )
            res = sel.model.fit()

        elif model == "ThetaModel":
            smod = tsa.forecasting.theta.ThetaModel(data)

        elif model == "SARIMAX":

            smod = tsa.statespace.SARIMAX(data, order=(1, 0, 1))

        elif model == "ARIMA":
            smod = tsa.arima.ARIMA(data.endog, order=(1, 0, 0), trend="n")

        elif model == "ETSModel":

            smod = tsa.exponential_smoothing.ets.ETSModel(data)

        self._fit_predict(smod, s_params)

    def _stats_pca(self, data, model, s_params):

        smod = multivariate.pca.PCA(data, standardize=False, demean=True)

        self._fit_predict(smod, s_params)

    def _fit_predict(self, smod, s_params):

        self.statsmodels_object["fit"] = smod.fit(
            method=s_params["fit_method"],
            disp=s_params["disp"],
            cov_type=s_params["cov_type"],
        )
        self.statsmodels_object["summary"] = self.statsmodels_object["fit"].summary()
        self.statsmodels_object["prediction"] = self.statsmodels_object["fit"].predict()
