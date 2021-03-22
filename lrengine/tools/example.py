import pandas as pd


def example1(path):

    data = pd.read_csv(path)

    return list(data["date_delta"])
