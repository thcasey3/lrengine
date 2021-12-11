"""
adtk module, for interacting with adtk
"""
from adtk import data, detector, visualization
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

    def __init__(self, time=[], variable=[], method="QuantileAD", adtk_args={}):

        self._adtk(time=time, variable=variable, method=method, adtk_args=adtk_args)


    def _adtk(self, time, variable, method, adtk_args):
        
        a_args = {
            "detect_func":,
        "detect_func_params":,
        "k":,
        "c":,
        "side":,
        "high":,
        "low":,
        "window":,
        "n_steps":,
        "step_size":,
        "contamination":,
        "target":,
        "n_clusters":,
        }
        a_args.update(adtk_args)

        df = pd.DataFrame({"time": time, "variable": variable})
        df.set_index("time")

        series_data = data.validate_series(df, check_categorical=True)

        if method == "custom":
            AD = detector.CustomizedDetectorHD(
                detect_func=a_args["detect_func"],
                detect_func_params=a_args["detect_func_params"],
            )
        elif method == "PcaAD":
            AD = detector.PcaAD(k=a_args["k"])
        elif method == "SeasonalAD":
            AD = detector.SeasonalAD(c=a_args["c"], side=a_args["side"])
        elif method == "ThresholdAD":
            AD = detector.ThresholdAD(high=a_args["high"], low=a_args["low"])
        elif method == "QuantileAD":
            AD = detector.QuantileAD(high=a_args["high"], low=a_args["low"])
        elif method == "InterQuartileRangeAD":
            AD = detector.InterQuartileRangeAD(c=a_args["c"])
        elif method == "PersistAD":
            AD = detector.PersistAD(c=a_args["c"], side=a_args["side"])
        elif method == "LevelShiftAD":
            AD = detector.LevelShiftAD(
                c=a_args["c"], side=a_args["side"], window=a_args["window"]
            )
        elif method == "VolatilityShiftAD":
            AD = detector.VolatilityShiftAD(
                c=a_args["c"], side=a_args["side"], window=a_args["window"]
            )
        elif method == "AutoregressionAD":
            AD = detector.AutoregressionAD(
                n_steps=a_args["n_steps"], step_size=a_args["step_size"], c=a_args["c"]
            )
        elif method == "OutlierDetector":
            AD = detector.OutlierDetector(
                EllipticEnvelope(contamination=a_args["contamination"])
            )
        elif method == "RegressionAD":
            AD = detector.RegressionAD(
                regressor=LinearRegression(), target=a_args["target"], c=a_args["c"]
            )
        elif method == "MinClusterDetector":
            AD = detector.MinClusterDetector(KMeans(n_clusters=a_args["n_clusters"]))

        if any(
            map(
                [
                    "custom",
                    "PcaAD",
                    "OutlierDetector",
                    "RegressionAD",
                    "MinClusterDetector",
                ].__contains__,
                method,
            )
        ):
            if method == "custom":
                anomalies = AD.detect(df)
            else:
                anomalies = AD.fit_detect(df)
            visualization.plot(
                df,
                anomaly=anomalies,
                ts_linewidth=1,
                ts_markersize=3,
                anomaly_color="red",
                anomaly_alpha=0.3,
                curve_group="all",
            )
        else:
            if method == "ThresholdAD":
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
            )

        plt.show()
