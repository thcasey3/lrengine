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

    def __init__(self, time=[], variable=[], method="OutlierDetector", adtk_args={}):

        self._adtk(time=time, variable=variable, method=method, adtk_args=adtk_args)

    def _adtk(self, time, variable, method, adtk_args):

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

        a_args = {
            "detect_func": None,
            "detect_func_params": None,
            "fit_func": None,
            "fit_func_params": None,
            "k": 1,
            "c": 5.0,
            "side": "both",
            "high": None,
            "low": None,
            "window": 3,
            "n_steps": 1,
            "step_size": 1,
            "contamination": 0.025,
            "target": "variable",
            "n_clusters": 3,
            "regressor": None,
            "min_periods": None,
            "agg": "median",
            "trend": False,
            "freq": None,
            "mincluster_model": KMeans(n_clusters=n_clus, random_state=rand_st),
            "outlier_model": EllipticEnvelope(
                contamination=contam, random_state=rand_st
            ),
            "steps": [
                ("deseasonal", transformer.ClassicSeasonalDecomposition()),
                ("quantile_ad", detector.QuantileAD(high=hi, low=lo)),
            ],
        }
        a_args.update(adtk_args)

        df = pd.DataFrame({"time": time, "variable": variable})
        if not df["time"].dtype == "datetime64[ns]":
            try:
                df["time"] = pd.to_datetime(
                    df["time"], infer_datetime_format=True, errors="coerce"
                )
            except:
                pass
        df.set_index("time")

        series_data = data.validate_series(df, check_categorical=True)

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
