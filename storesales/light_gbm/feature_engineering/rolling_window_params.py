from dataclasses import dataclass

import pandas as pd

from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute


@dataclass
class ExtractFeaturesParams:
    timeseries_container: pd.DataFrame
    default_fc_parameters: ComprehensiveFCParameters = MinimalFCParameters()
    column_id: str = "id"
    column_sort: str = "date"
    impute_function: callable = impute
    n_jobs: int = 6


def get_minimal_fc_parameters() -> ComprehensiveFCParameters:
    fc_parameters = MinimalFCParameters()
    del fc_parameters["length"]
    del fc_parameters["absolute_maximum"]
    return fc_parameters


def get_custom_minimal_fc_parameters(
    number_peaks_n: list[int],
    autocorrelation_lag: list[int],
    partial_autocorrelation_lag: list[int],
) -> ComprehensiveFCParameters:
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
