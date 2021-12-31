"""
anomalies module, for calling adtk
"""
from adtk import data, transformer, detector, pipe, visualization
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

from . import config, utilities


class adtk:
    """
    Creates object for calling adtk (https://adtk.readthedocs.io/en/stable/index.html)

    Args:
        df (pd.DataFrame): DataFrame with data to analyze
        time (str or list-like): df column name or time index for series
        series (stror list-like): df column name or series to analyze
        update_config (dict), optional: {'arg': value} to change from the defaults for the method given to detect

    Returns:
        Instantiates adtk object
    """

    def __init__(
        self,
        df=None,
        time=None,
        series=None,
        update_config={},
    ):

        self.adtk_params = config.ANOMALIES_PARAMS
        if update_config:
            self.adtk_params.update(update_config)

        a_args = {
            "regressor": LinearRegression(),
            "mincluster_model": KMeans(
                n_clusters=self.adtk_params["n_clusters"],
                random_state=self.adtk_params["random_state"],
            ),
            "outlier_model": EllipticEnvelope(
                contamination=self.adtk_params["contamination"],
                support_fraction=self.adtk_params["support_fraction"],
                random_state=self.adtk_params["random_state"],
            ),
            "steps": [
                ("deseasonal", transformer.ClassicSeasonalDecomposition()),
                (
                    "quantile_ad",
                    detector.QuantileAD(
                        high=self.adtk_params["high"], low=self.adtk_params["low"]
                    ),
                ),
            ],
        }
        self.adtk_params.update(a_args)

        self.df = df
        self.adtk_object = {}
        self.adtk_object["time"] = time
        self.adtk_object["series"] = series

    def detect(
        self,
        select="all",
        append_object=False,
        method="OutlierDetector",
        survey=False,
        plot=True,
    ):
        """
        Calls adtk detector

        Args:
            select (dict), optional: {'column': list(values)} to subset and analyze individually from the frame, default is 'all'
            append_object (bool), optional: True means collect selections or False means purge them when select is used
            survey (dict), optional: {'select': {'column': list(values)}, 'survey parameter': list(value range)}, survey['select'] can be 'all' to use entire frame
            plot (bool), optional: True means raise plot after analysis

        Returns:
            Populates adtk object with results
        """
        if survey:
            survey_par = [x for x in list(survey.keys()) if x != "select"][0]
            if survey["select"] == "all":
                itm_col = "all"
                survey["select"] = {"all": ["all"]}
            else:
                itm_col = list(survey["select"].keys())[0]
                if isinstance(survey["select"][itm_col], str):
                    survey["select"][itm_col] = [survey["select"][itm_col]]
            for itm in select[itm_col]:
                if not append_object:
                    self.adtk_object = {
                        key: item
                        for key, item in self.stats_object.items()
                        if key in ["time", "series"]
                    }
                self.adtk_object[itm] = {}
                self.adtk_object[itm]["survey"] = {}
                self.adtk_object[itm]["survey"][survey_par] = {}
                for par in survey[survey_par]:
                    self.adtk_params.update({survey_par: par})
                    self.prepare(select=itm, column=itm_col)
                    self.run_adtk(select=itm, method=method, plot=plot)
                    self.adtk_object[itm]["survey"][survey_par][
                        str(par)
                    ] = self.adtk_object[itm]["anomalies"]
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
                    self.adtk_object = {
                        key: item
                        for key, item in self.stats_object.items()
                        if key in ["time", "series"]
                    }
                self.adtk_object[itm] = {}
                self.prepare(select=itm, column=itm_col)
                self.run_adtk(select=itm, method=method, plot=plot)

    def prepare(self, select="all", column=None):

        if select != "all":
            df = self.df[self.df[column] == select]
        elif select == "all":
            df = self.df.copy()

        if isinstance(self.adtk_object["time"], list) and isinstance(
            self.adtk_object["series"], list
        ):
            df = pd.DataFrame(
                {"time": self.adtk_object["time"], "series": self.adtk_object["series"]}
            )
            time_col = "time"
            series_col = "series"
        else:
            time_col = self.adtk_object["time"]
            series_col = self.adtk_object["series"]

        if df is not None:
            if not df[time_col].dtype == "datetime64[ns]":
                try:
                    new_time = pd.to_datetime(
                        df[time_col], infer_datetime_format=True, errors="coerce"
                    )
                except:
                    new_time = df[time_col]
                    pass
            else:
                new_time = df[time_col]
            ts_df = pd.DataFrame({time_col: new_time, series_col: df[series_col]})
            if self.adtk_params["time_range"] is not None:
                ts_df = utilities.time_filter(
                    ts_df,
                    time_column=time_col,
                    timeframe=(
                        self.adtk_params["time_range"][0],
                        self.adtk_params["time_range"][1],
                    ),
                )

            ts_df.set_index(time_col, inplace=True)
            ts_df.dropna(inplace=True)
        else:
            raise ValueError(
                "you must give a df with column names for time and series, or lists for time and series"
            )

        self.adtk_object[select]["detect_df"] = ts_df

    def run_adtk(self, select, method, plot):

        self.adtk_object[select]["validated_series"] = data.validate_series(
            self.adtk_object[select]["detect_df"], check_categorical=True
        )

        if method == "Pipeline":
            AD = pipe.Pipeline(self.adtk_params["steps"])
        elif method == "custom":
            AD = detector.CustomizedDetectorHD(
                detect_func=self.adtk_params["detect_func"],
                detect_func_params=self.adtk_params["detect_func_params"],
                fit_func=self.adtk_params["fit_func"],
                fit_func_params=self.adtk_params["fit_func_params"],
            )
        elif method == "PcaAD":
            AD = detector.PcaAD(k=self.adtk_params["k"], c=self.adtk_params["c"])
        elif method == "SeasonalAD":
            AD = detector.SeasonalAD(
                freq=self.adtk_params["freq"],
                side=self.adtk_params["side"],
                c=self.adtk_params["c"],
                trend=self.adtk_params["trend"],
            )
        elif method == "ThresholdAD":
            AD = detector.ThresholdAD(
                low=self.adtk_params["low"], high=self.adtk_params["high"]
            )
        elif method == "QuantileAD":
            AD = detector.QuantileAD(
                low=self.adtk_params["low"], high=self.adtk_params["high"]
            )
        elif method == "InterQuartileRangeAD":
            AD = detector.InterQuartileRangeAD(c=self.adtk_params["c"])
        elif method == "PersistAD":
            AD = detector.PersistAD(
                window=self.adtk_params["window"],
                c=self.adtk_params["c"],
                side=self.adtk_params["side"],
                min_periods=self.adtk_params["min_periods"],
                agg=self.adtk_params["agg"],
            )
        elif method == "LevelShiftAD":
            AD = detector.LevelShiftAD(
                window=self.adtk_params["window"],
                c=self.adtk_params["c"],
                side=self.adtk_params["side"],
                min_periods=self.adtk_params["min_periods"],
            )
        elif method == "VolatilityShiftAD":
            AD = detector.VolatilityShiftAD(
                window=self.adtk_params["window"],
                c=self.adtk_params["c"],
                side=self.adtk_params["side"],
                min_periods=self.adtk_params["min_periods"],
                agg=self.adtk_params["agg"],
            )
        elif method == "AutoregressionAD":
            AD = detector.AutoregressionAD(
                n_steps=self.adtk_params["n_steps"],
                step_size=self.adtk_params["step_size"],
                regressor=self.adtk_params["regressor"],
                c=self.adtk_params["c"],
                side=self.adtk_params["side"],
            )
        elif method == "OutlierDetector":
            AD = detector.OutlierDetector(model=self.adtk_params["outlier_model"])
        elif method == "RegressionAD":
            AD = detector.RegressionAD(
                regressor=self.adtk_params["regressor"],
                target=self.adtk_params["target"],
                c=self.adtk_params["c"],
                side=self.adtk_params["side"],
            )
        elif method == "MinClusterDetector":
            AD = detector.MinClusterDetector(model=self.adtk_params["mincluster_model"])

        if method in ["ThresholdAD", "custom"]:
            self.adtk_object[select]["anomalies"] = AD.detect(
                self.adtk_object[select]["validated_series"]
            )
        else:
            self.adtk_object[select]["anomalies"] = AD.fit_detect(
                self.adtk_object[select]["validated_series"]
            )

        self.adtk_object[select]["detector"] = AD

        if plot:
            self.plot(
                series_data=self.adtk_object[select]["validated_series"],
                anomalies=self.adtk_object[select]["anomalies"],
                select=select,
                ts_linewidth=self.adtk_params["ts_linewidth"],
                ts_markersize=self.adtk_params["ts_markersize"],
                anomaly_markersize=self.adtk_params["anomaly_markersize"],
                anomaly_color=self.adtk_params["anomaly_color"],
                anomaly_tag=self.adtk_params["anomaly_tag"],
                curve_group=self.adtk_params["curve_group"],
                figsize=self.adtk_params["figsize"],
            )

    def plot(self, series_data=None, anomalies=None, select=False, **kwargs):

        visualization.plot(
            series_data,
            anomaly=anomalies,
            ts_linewidth=kwargs["ts_linewidth"],
            ts_markersize=kwargs["ts_markersize"],
            anomaly_markersize=kwargs["anomaly_markersize"],
            anomaly_color=kwargs["anomaly_color"],
            anomaly_tag=kwargs["anomaly_tag"],
            curve_group=kwargs["curve_group"],
            figsize=kwargs["figsize"],
        )
        if select:
            plt.title(select)
        plt.show()
