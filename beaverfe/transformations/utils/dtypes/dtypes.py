import numpy as np
import pandas as pd


def bool_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["bool"]).columns.tolist()


def categorical_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["object", "category"]).columns.tolist()


def datetime_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()


def numerical_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(include=np.number).columns.tolist()


def timedelta_columns(data: pd.DataFrame) -> list:
    return data.select_dtypes(
        include=["timedelta64[ns]", "timedelta64"]
    ).columns.tolist()
