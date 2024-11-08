import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series

from storesales.light_gbm.param_dataclasses import ExtractFeaturesParam


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
    extract_features_param: ExtractFeaturesParam, prefix: str
) -> pd.DataFrame:
    features_df = extract_features(**extract_features_param.__dict__)
    features_df.index.names = ["store_family", "date"]
    features_df.columns = [f"{prefix}_{col}" for col in features_df.columns]

    features_df.reset_index(level=["store_family", "date"], inplace=True)
    features_df[["store_nbr", "family"]] = features_df["store_family"].str.split(
        "-", expand=True
    )
    features_df.drop(columns=["store_family"], inplace=True)

    return features_df


def get_minimal_fc_parameters() -> MinimalFCParameters:
    fc_parameters = MinimalFCParameters()
    del fc_parameters["length"]
    del fc_parameters["absolute_maximum"]
    return fc_parameters
