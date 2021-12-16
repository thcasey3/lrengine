import pandas as pd
import datetime
import string
from collections import Counter

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

    df = df.astype(final_types, errors="ignore")

    return df


def group(df, col, on, kind):

    if kind == "mean":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).mean()
    elif kind == "median":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).median()
    elif kind == "min":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).min()
    elif kind == "max":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).max()
    elif kind == "sum":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).sum()
    elif kind == "first":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).first()
    elif kind == "last":
        agg_val_lst = df[[on, col]].set_index(on).groupby(on).last()

    new_df = pd.DataFrame(
        {on: agg_val_lst.index, col: agg_val_lst.values.reshape(-1)}
    ).dropna()
    return new_df, agg_val_lst.index


def resample(df, col, on, freq, interp, kind):

    if freq == "infer":
        raise ValueError("to resample, you must specify a freq and not use 'infer'")

    if kind == "mean":
        resamp = df.resample(freq).mean()
    elif kind == "median":
        resamp = df.resample(freq).median()
    elif kind == "min":
        resamp = df.resample(freq).min()
    elif kind == "max":
        resamp = df.resample(freq).max()
    elif kind == "sum":
        resamp = df.resample(freq).sum()
    elif kind == "first":
        resamp = df.resample(freq).first()
    elif kind == "last":
        resamp = df.resample(freq).last()

    if interp:
        resamp = resamp.interpolate(interp)

    new_df = pd.DataFrame({on: resamp.index, col: resamp.values.reshape(-1)}).dropna()
    return new_df, resamp.index


def datetime_index(df, col, freq="infer", ascending=False, drop_duplicates=False):

    if not df[col].dtype == "datetime64[ns]":
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)

    df_index = pd.DatetimeIndex(df[col], freq=freq)

    if ascending:
        df_index = df_index.sort_values(ascending=ascending)
    if drop_duplicates:
        df_index = df_index.drop_duplicates()

    df.index = df_index
    return df, df.index


def time_filter(df, time_column, timeframe):

    limited_df = df[
        df[time_column] > timeframe[0]
    ]  # datetime.strptime("2011-01-01", "%Y-%m-%d")]
    limited_df = limited_df[
        limited_df[time_column] < timeframe[1]
    ]  # datetime.strptime("2020-02-01", "%Y-%m-%d")]

    return limited_df


def return_split_desc(row):
    new_desc = row.lower().split(" ")
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


def decoder(X, encoder):
    return encoder.inverse_transform(X)


def predictable_list(df, threshold=2, count_column=""):

    df2 = df[df[count_column].map(df[count_column].value_counts()) > threshold]
    series = df2[count_column].value_counts().to_frame().index
    return series
