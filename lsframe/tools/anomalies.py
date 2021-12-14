"""
adtk module, for interacting with adtk
"""
from adtk import data, transformer, detector, pipe, visualization
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt


class adtk:
    """
    class for using adtk with the lsobject.frame

    Args:
    Returns:
    """

    def __init__(
        self,
        df=None,
        time=None,
        series=None,
        method="OutlierDetector",
        adtk_args={},
        plot=True,
    ):

        if isinstance(time, list) and isinstance(series, list):
            df = pd.DataFrame({"time": time, "series": series})
            time_col = "time"
            series_col = "series"
        else:
            time_col = time
            series_col = series

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
            ts_df = pd.DataFrame({"time": new_time, "series": df[series_col]})
            ts_df.set_index("time", inplace=True)
            ts_df.dropna(inplace=True)
        else:
            raise ValueError(
                "you must give a df with column names for time and series, or lists for time and series"
            )

        self.adtk_object = {}
        self.adtk_object["method"] = method
        self.adtk_object["df"] = ts_df

        if "n_clusters" in adtk_args.keys():
            n_clus = adtk_args["n_clusters"]
        else:
            n_clus = 3
        if "contamination" in adtk_args.keys():
            contam = adtk_args["contamination"]
        else:
            contam = 0.025
        if "random_state" in adtk_args.keys():
            rand_st = adtk_args["random_state"]
        else:
            rand_st = 42
        if "high" in adtk_args.keys():
            hi = adtk_args["high"]
        else:
            hi = 0.995
        if "low" in adtk_args.keys():
            lo = adtk_args["low"]
        else:
            lo = 0.005
        if "support_fraction" in adtk_args.keys():
            sup_frac = adtk_args["support_fraction"]
        else:
            sup_frac = None

        a_args = {
            "detect_func": None,
            "detect_func_params": None,
            "fit_func": None,
            "fit_func_params": None,
            "k": 1,
            "c": 3.0,
            "side": "both",
            "high": None,
            "low": None,
            "window": 3,
            "n_steps": 1,
            "step_size": 1,
            "contamination": 0.025,
            "target": "series",
            "n_clusters": 3,
            "regressor": None,
            "min_periods": None,
            "agg": "median",
            "trend": False,
            "freq": None,
            "mincluster_model": KMeans(n_clusters=n_clus, random_state=rand_st),
            "outlier_model": EllipticEnvelope(
                contamination=contam, support_fraction=sup_frac, random_state=rand_st
            ),
            "steps": [
                ("deseasonal", transformer.ClassicSeasonalDecomposition()),
                ("quantile_ad", detector.QuantileAD(high=hi, low=lo)),
            ],
        }
        a_args.update(adtk_args)
        self.adtk_object["adtk_args"] = a_args

        self._adtk(ts_df, method, a_args, plot)

    def _adtk(self, df, method, a_args, plot):

        series_data = data.validate_series(df, check_categorical=True)
        self.adtk_object["series"] = series_data.copy()

        if method == "Pipeline":
            AD = pipe.Pipeline(a_args["steps"])
        elif method == "custom":
            AD = detector.CustomizedDetectorHD(
                detect_func=a_args["detect_func"],
                detect_func_params=a_args["detect_func_params"],
                fit_func=a_args["fit_func"],
                fit_func_params=a_args["fit_func_params"],
            )
        elif method == "PcaAD":
            AD = detector.PcaAD(k=a_args["k"], c=a_args["c"])
        elif method == "SeasonalAD":
            AD = detector.SeasonalAD(
                freq=a_args["freq"],
                side=a_args["side"],
                c=a_args["c"],
                trend=a_args["trend"],
            )
        elif method == "ThresholdAD":
            AD = detector.ThresholdAD(low=a_args["low"], high=a_args["high"])
        elif method == "QuantileAD":
            AD = detector.QuantileAD(low=a_args["low"], high=a_args["high"])
        elif method == "InterQuartileRangeAD":
            AD = detector.InterQuartileRangeAD(c=a_args["c"])
        elif method == "PersistAD":
            AD = detector.PersistAD(
                window=a_args["window"],
                c=a_args["c"],
                side=a_args["side"],
                min_periods=a_args["min_periods"],
                agg=a_args["agg"],
            )
        elif method == "LevelShiftAD":
            AD = detector.LevelShiftAD(
                window=a_args["window"],
                c=a_args["c"],
                side=a_args["side"],
                min_periods=a_args["min_periods"],
            )
        elif method == "VolatilityShiftAD":
            AD = detector.VolatilityShiftAD(
                window=a_args["window"],
                c=a_args["c"],
                side=a_args["side"],
                min_periods=a_args["min_periods"],
                agg=a_args["agg"],
            )
        elif method == "AutoregressionAD":
            AD = detector.AutoregressionAD(
                n_steps=a_args["n_steps"],
                step_size=a_args["step_size"],
                regressor=a_args["regressor"],
                c=a_args["c"],
                side=a_args["side"],
            )
        elif method == "OutlierDetector":
            AD = detector.OutlierDetector(model=a_args["outlier_model"])
        elif method == "RegressionAD":
            AD = detector.RegressionAD(
                regressor=LinearRegression(),
                target=a_args["target"],
                c=a_args["c"],
                side=a_args["side"],
            )
        elif method == "MinClusterDetector":
            AD = detector.MinClusterDetector(model=a_args["mincluster_model"])

        if method in ["ThresholdAD", "custom"]:
            anomalies = AD.detect(series_data)
        else:
            anomalies = AD.fit_detect(series_data)

        self.adtk_object["detector"] = AD
        self.adtk_object["anomalies"] = anomalies

        if plot:
            self.plot()

    def plot(self, series_data=None, anomalies=None):

        if series_data is None and anomalies is None:
            series_data = self.adtk_object["series"]
            anomalies = self.adtk_object["anomalies"]

        visualization.plot(
            series_data,
            anomaly=anomalies,
            ts_linewidth=1,
            ts_markersize=3,
            anomaly_markersize=5,
            anomaly_color="red",
            anomaly_tag="marker",
            curve_group="all",
        )

        plt.show()
