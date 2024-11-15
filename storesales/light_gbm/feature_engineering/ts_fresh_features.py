from dataclasses import dataclass

import pandas as pd

from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import roll_time_series

from storesales.light_gbm.param_dataclasses import ExtractFeaturesParams


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


def get_minimal_fc_parameters() -> MinimalFCParameters:
    fc_parameters = MinimalFCParameters()
    del fc_parameters["length"]
    del fc_parameters["absolute_maximum"]
    return fc_parameters


def get_custom_minimal_fc_parameters(
    number_peaks_n: list[int],
    autocorrelation_lag: list[int],
    partial_autocorrelation_lag: list[int],
) -> MinimalFCParameters:
    fc_parameters = MinimalFCParameters()
    del fc_parameters["length"]
    del fc_parameters["absolute_maximum"]

    additional_features = {
        "mean_abs_change": None,
        "mean_change": None,
        "longest_strike_above_mean": None,
        "longest_strike_below_mean": None,
        "number_peaks": [{"n": n} for n in number_peaks_n],
        "autocorrelation": [{"lag": lag} for lag in autocorrelation_lag],
        "partial_autocorrelation": [
            {"lag": lag} for lag in partial_autocorrelation_lag
        ],
        "skewness": None,
        "kurtosis": None,
        "abs_energy": None,
    }
    fc_parameters.update(additional_features)

    return fc_parameters


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
