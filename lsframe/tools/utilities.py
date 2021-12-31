import pandas as pd
import numpy as np
from datetime import datetime
import string
from collections import Counter
from sklearn import metrics
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import (
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)

special_chars = ["-", "*", ".", ",", "'", "/", "#", "&", "@", "(", ")"] + list(
    string.digits
)
com_wrds = [
    "",
    " ",
    "the",
    "in",
    "to",
    "on",
    "for",
    "is",
    "of",
    "and",
    "at",
    "has",
    "are",
    "there",
    "by",
    "with",
    "see",
    "not",
    "have",
    "an",
    "will",
] + list(string.ascii_letters)


def adjust_datatypes(
    df,
    new_types={},
):

    for col, val in new_types.items():
        if val == "datetime64[ns]":
            df[col] = pd.to_datetime(
                df[col], infer_datetime_format=True, errors="coerce"
            ).dt.tz_localize(tz=None)
    final_types = {
        key: val for key, val in new_types.items() if val != "datetime64[ns]"
    }

    return df.astype(final_types, errors="ignore")


def group_index(
    df, by, kind="mean", resample=False, interp=False, freq=False, aggregate=False
):
    grouped_df = df.groupby(by)
    if aggregate:
        grouped_df = grouped_df.agg(aggregate)
    else:
        if resample and freq:
            grouped_df = grouped_df.resample(freq)
            if interp:
                grouped_df = grouped_df.interpolate(method=interp)

        if kind == "mean":
            grouped_df = grouped_df.mean()
        elif kind == "median":
            grouped_df = grouped_df.median()
        elif kind == "min":
            grouped_df = grouped_df.min()
        elif kind == "max":
            grouped_df = grouped_df.max()
        elif kind == "sum":
            grouped_df = grouped_df.sum()
        elif kind == "first":
            grouped_df = grouped_df.first()
        elif kind == "last":
            grouped_df = grouped_df.last()

    grouped_df = grouped_df.dropna()
    if by:
        grouped_df[by] = grouped_df.index

    return grouped_df, grouped_df.index


def resample_index(df, freq, kind, interp, time=None):

    if freq == "infer":
        raise ValueError("to resample, you must specify a freq and not use 'infer'")

    if kind == "mean":
        resampled_df = df.resample(freq).mean()
    elif kind == "median":
        resampled_df = df.resample(freq).median()
    elif kind == "min":
        resampled_df = df.resample(freq).min()
    elif kind == "max":
        resampled_df = df.resample(freq).max()
    elif kind == "sum":
        resampled_df = df.resample(freq).sum()
    elif kind == "first":
        resampled_df = df.resample(freq).first()
    elif kind == "last":
        resampled_df = df.resample(freq).last()

    if interp:
        resampled_df = resampled_df.interpolate(method=interp)

    resampled_df = resampled_df.dropna()
    if time is not None:
        resampled_df[time] = resampled_df.index

    return resampled_df, resampled_df.index


def datetime_index(df, time, freq="infer", ascending=False, drop_duplicates=False):

    if not df[time].dtype == "datetime64[ns]":
        df[time] = pd.to_datetime(df[time], infer_datetime_format=True, errors="coerce")

    df_index = pd.DatetimeIndex(df[time], freq=freq)

    if ascending:
        df_index = df_index.sort_values(ascending=ascending)
    if drop_duplicates:
        df_index = df_index.drop_duplicates()

    df.index = df_index
    return df, df.index


def time_filter(df, time_column=None, timeframe=None, index=False):
    df[time_column] = pd.to_datetime(
        df[time_column], infer_datetime_format=True, errors="coerce"
    )
    if index:
        limited_df = df[df.index > timeframe[0]]
        limited_df = limited_df[limited_df.index < timeframe[1]]
    else:
        limited_df = df[df[time_column] > timeframe[0]]
        limited_df = limited_df[limited_df[time_column] < timeframe[1]]

    return limited_df


def return_split_desc(row):
    new_desc = row.lower().split(" ")
    if not isinstance(new_desc, list):
        new_desc = [new_desc]

    newer_desc = [x for x in new_desc if not any(map(x.strip().__eq__, special_chars))]
    for x, val in enumerate(newer_desc):
        if any(map(val.strip().__contains__, special_chars)):
            for z in special_chars:
                newer_desc[x] = newer_desc[x].replace(z, "")
    return newer_desc


def extract_com_words(row):
    return [x for x in row if x not in com_wrds]


def word_to_bool(row, **kwargs):
    return 1 if kwargs["word"] in row.lower() else 0


def split_cols(df, base_column="", use_top=10):

    splitted = df[base_column].apply(return_split_desc)
    splitted = splitted.apply(extract_com_words)

    wrd_lst = [x for y in splitted for x in y]

    if use_top == "all":
        use_top = len(wrd_lst)
    count = Counter(wrd_lst).most_common(use_top)
    to_use_list = [x[0] for x in count]

    for wrd in to_use_list:
        df[wrd] = df[base_column].apply(word_to_bool, word=wrd)

    return df, to_use_list


def predictable_list(
    df,
    threshold=1,
    count_column="",
    date_column="",
    cost_column="",
):

    df1 = df[df[cost_column] > 0]
    if df1[date_column].dtype in ["int64", "float64"]:
        df2 = df1[df1[date_column] > 0]
    elif df1[date_column].dtype == "datetime64[ns]":
        df2 = df1[df1[date_column] < datetime.today()]
    else:
        try:
            df1[date_column] = pd.to_datetime(
                df1[date_column], infer_datetime_format=True, errors="coerce"
            )
            df2 = df1[df1[date_column] < datetime.today()]
        except:
            raise TypeError("cannot interpret date column")

    df2.drop_duplicates(inplace=True)
    df3 = df2[df2[count_column].map(df2[count_column].value_counts()) > threshold]
    series = df3[count_column].value_counts().to_frame().index
    return series


def outlier_filter(df, which, threshold, s_frac, col):

    out = EllipticEnvelope(
        contamination=threshold, support_fraction=s_frac, random_state=42
    )
    subjects = df[[col]].to_numpy().reshape(-1, 1)
    result = out.fit_predict(subjects)
    df["outliers"] = result
    if which == "remove":
        return df[df["outliers"] == 1].drop(labels="outliers", axis=1)
    elif which == "select":
        return df[df["outliers"] == -1].drop(labels="outliers", axis=1)


def fit_outliers(df, which, threshold, col):

    try:
        return outlier_filter(df, which, threshold, None, col)
    except ValueError:
        s_frac = 0
        while s_frac <= 1:
            try:
                s_frac += 0.05
                return outlier_filter(df, which, threshold, s_frac, col)
            except ValueError:
                continue
        else:
            return df


def drop_uniform(df):

    cols = []
    for ix in df.columns.values:
        if len(df.drop_duplicates([ix])[ix]) <= 1:
            cols.append(ix)
            df.drop(labels=[ix], axis=1, inplace=True)

    return df, cols


def try_fix_freq(df, time):

    flag = []
    for f in ["N", "U", "L", "S", "T", "H", "D", "W", "M", "Q", "Y"]:
        try:
            test_df, _ = datetime_index(df, time, freq=f)
            flag.append(f)
        except:
            continue

    if flag:
        df, _ = datetime_index(df, time, freq=flag[-1])

    return df, flag


def prepare_timeseries(
    df, time, time_range, group, aggregate, resample, freq, interpolate
):

    if df[time].dtype != "datetime64[ns]":
        df[time] = pd.to_datetime(df[time], infer_datetime_format=True).sort_values(
            ascending=True
        )

    if time_range:
        df = time_filter(df, time_column=time, timeframe=(time_range[0], time_range[1]))
    if group:
        df, _ = group_index(
            df, by=time, kind=group, interp=interpolate, aggregate=aggregate
        )
    if resample:
        new_df, _ = datetime_index(df, time, freq="infer")
        df, _ = resample_index(new_df, freq, resample, interpolate, time=time)

    try:
        df, _ = datetime_index(df, time, freq=freq)
    except:
        df, flag = try_fix_freq(df, time)
        if flag:
            print(f"your index matched '{flag}' freq, setting to {freq[-1]}")
        else:
            print(
                "Your index freq doesn't match any common frequencies, try using resample"
            )
            df, _ = datetime_index(df, time, freq="infer")

    if not df.index.freq:
        print("not able to assign freq to DatetimeIndex")

    df.drop(time, axis=1, inplace=True)
    return df, df.index


def recover_transform(
    type,
    values,
    log_lam,
    first_value,
    scale_factor,
    index,
    first_index,
    index_name,
    data_name,
):

    if type == "log":
        series2 = np.exp(values) + log_lam
    elif type == "diff":
        if first_value:
            values = np.concatenate(([first_value], values))
        if first_index:
            index = np.concatenate(([first_index], index))
        series2 = values.cumsum() * np.float(scale_factor)
    elif type == "difflog":
        if first_index:
            index = np.concatenate(([first_index], index))
        series2 = (np.exp(values) + log_lam).cumsum() * np.float(scale_factor)
        if first_value:
            series2 = np.concatenate(([first_value], series2))
    elif type == "logdiff":
        if first_value:
            values = np.concatenate(([np.log(first_value)], values))
        if first_index:
            index = np.concatenate(([first_index], index))
        series2 = np.exp(values.cumsum() * np.float(scale_factor)) + log_lam
    else:
        series2 = values

    df = pd.DataFrame({index_name: index, data_name: series2.reshape(-1)}).set_index(
        index_name, drop=True
    )
    return df


def normalizer(values, type, feature_range=(0, 1)):
    if type == "minmax":
        norm = MinMaxScaler(feature_range=feature_range)
    elif type == "maxabs":
        norm = MaxAbsScaler()
    norm = norm.fit(values)
    return norm.transform(values), norm


def denormalizer(values, norm):
    return norm.inverse_transform(values)


def encoder(X, type):
    if type == "ordinal":
        enc = OrdinalEncoder()
    elif type == "label":
        enc = LabelEncoder()
    enc = enc.fit(X)
    return enc.transform(X), enc


def decoder(X, enc):
    return enc.inverse_transform(X)


def scaler(values, type):
    if type == "robust":
        scl = RobustScaler()
    elif type == "standard":
        scl = StandardScaler()
    scl = scl.fit(values)
    return scl.transform(values), scl


def descaler(values, scl):
    return scl.inverse_transform(values)


def handle_na(df, how):

    if isinstance(how, dict):
        if list(how.items())[0][0] in [
            "fill",
            "value",
        ]:
            df.fillna(
                value=list(how.items())[0][1],
                inplace=True,
            )
        elif list(how.items())[0][0] == "method":
            df.fillna(
                method=list(how.items())[0][1],
                inplace=True,
            )
    elif how == "fill":
        for col in df.columns.values:
            if df[col].dtype == "float64":
                df[col].fillna(value=42.42, inplace=True)
            elif df[col].dtype == "int64":
                df[col].fillna(value=4242, inplace=True)
            elif df[col].dtype == "object":
                df[col].fillna(value="4242", inplace=True)
            elif df[col].dtype == "datetime64[ns]":
                df[col].fillna(
                    value=datetime(1942, 4, 2).replace(tzinfo=None),
                    inplace=True,
                )
    elif how == "drop":
        df.dropna(how="any", inplace=True)

    return df


def return_metric(obj, X_true=None, y_true=None, y_pred=None, metric="mape"):

    try:
        if metric in ["r2", "score"]:
            return obj.score(X_true, y_true)
        elif metric == "score_samples":
            return obj.score_samples(X_true)
        elif metric == "mape":
            return metrics.mean_absolute_percentage_error(y_true, y_pred)
        elif metric == "mean_ae":
            return metrics.mean_absolute_error(y_true, y_pred)
        elif metric == "median_ae":
            return metrics.median_absolute_error(y_true, y_pred)
        elif metric == "mse":
            return metrics.mean_squared_error(y_true, y_pred)
        elif metric == "jaccard":
            return metrics.jaccard_score(y_true, y_pred)
        elif metric == "adjusted_rand":
            return metrics.cluster.adjusted_rand_score(y_true, y_pred)
    except ValueError:
        return False


def scores_dict_to_df(scores_dict):

    scores_df = pd.DataFrame(
        {
            "subject": scores_dict["subject"],
            "train_score": [x["train"] for x in scores_dict["score"]],
            "test_score": [x["test"] for x in scores_dict["score"]],
        }
    )
    scores_df.astype(
        {"train_score": "float64", "test_score": "float64"},
        errors="ignore",
    )

    lst = list(scores_dict.keys())
    if len(lst) == 4:
        scores_df[lst[3]] = [x for x in scores_dict[lst[3]]]

    return scores_df
