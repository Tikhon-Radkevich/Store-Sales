from dataclasses import dataclass

import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series

from storesales.light_gbm.feature_engineering.rolling_window_params import ExtractFeaturesParams


@dataclass
class Roll:
    name: str
    timeshift: int
    features: list[str]
    default_fc_parameters: ComprehensiveFCParameters
    roll_df: pd.DataFrame | None = None


def make_roll(
    df: pd.DataFrame, cols_to_roll: list[str], timeshift: int = 30, n_jobs: int = 6
) -> pd.DataFrame:
    df["store_family"] = df[["store_nbr", "family"]].astype(str).agg("-".join, axis=1)
    df.drop(columns=["store_nbr", "family"], inplace=True)

    df_columns = ["store_family", "date"] + cols_to_roll
    df = df[df_columns]

    rolls_df = roll_time_series(
        df_or_dict=df,
        column_id="store_family",
        column_sort="date",
        max_timeshift=timeshift,
        min_timeshift=timeshift,
        rolling_direction=1,
        n_jobs=n_jobs,
    )
    rolls_df.drop(columns="store_family", inplace=True)

    rolls_df.index = pd.MultiIndex.from_tuples(rolls_df["id"])
    rolls_df.set_index("date", drop=False, append=True, inplace=True)
    rolls_df.index.names = ["store_family", "date_roll_id", "date_id"]

    return rolls_df


def make_roll_features(
    extract_features_param: ExtractFeaturesParams, prefix: str
) -> pd.DataFrame:
    features_df = extract_features(**extract_features_param.__dict__)
    features_df.index.names = ["store_family", "date"]
    features_df.columns = [f"{prefix}_{col}" for col in features_df.columns]

    features_df.reset_index(level=["store_family", "date"], inplace=True)
    features_df[["store_nbr", "family"]] = features_df["store_family"].str.split(
        "-", expand=True
    )
    features_df.drop(columns=["store_family"], inplace=True)
    features_df["store_nbr"] = features_df["store_nbr"].astype(int)

    return features_df


def make_featured_df_from_rolls(df, rolls) -> pd.DataFrame:
    index_columns = ["date", "family", "store_nbr"]

    featured_df_list = []
    for roll in rolls:
        roll.roll_df = make_roll(df.copy(), roll.features, timeshift=roll.timeshift)
        extract_features_params = ExtractFeaturesParams(
            timeseries_container=roll.roll_df,
            default_fc_parameters=roll.default_fc_parameters,
        )
        featured_df = make_roll_features(extract_features_params, roll.name)
        featured_df_list.append(featured_df.set_index(index_columns))

    featured_df = pd.concat(featured_df_list, axis=1, join="inner")

    cols_to_merge = ["dcoilwtico", "onpromotion"] + index_columns
    featured_df = featured_df.merge(df[cols_to_merge], on=index_columns, how="inner")

    return featured_df.reset_index()
