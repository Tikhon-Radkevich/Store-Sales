import pandas as pd

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series

from storesales.light_gbm.param_dataclasses import (
    TrainRollParam,
    ExtractFeaturesParam,
)


def make_roll(df: pd.DataFrame, timeshift: int = 30) -> pd.DataFrame:
    df["store_family"] = df[["store_nbr", "family"]].astype(str).agg("-".join, axis=1)
    df.drop(columns=["store_nbr", "family"], inplace=True)

    train_roll_param = TrainRollParam(
        df_or_dict=df,
        column_id="store_family",
        column_sort="date",
        max_timeshift=timeshift,
        min_timeshift=timeshift,
    )

    rolls_df = roll_time_series(**train_roll_param.__dict__)
    rolls_df.drop(columns="store_family", inplace=True)

    rolls_df.index = pd.MultiIndex.from_tuples(rolls_df["id"])
    rolls_df.set_index("date", drop=False, append=True, inplace=True)
    rolls_df.index.names = ["store_family", "date_roll_id", "date_id"]

    return rolls_df


def make_roll_features(df: pd.DataFrame):
    extract_features_param = ExtractFeaturesParam(
        timeseries_container=df,
        column_id="id",
        column_sort="date",
    )

    features_df = extract_features(**extract_features_param.__dict__)
    features_df.index.names = ["store_family", "date"]

    features_df.reset_index(level=["store_family", "date"], inplace=True)
    features_df[["store_nbr", "family"]] = features_df["store_family"].str.split("-", expand=True)
    features_df.drop(columns=["store_family"], inplace=True)

    return features_df
