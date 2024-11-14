from dataclasses import dataclass

import pandas as pd

from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute


@dataclass
class TrainRollParam:
    df_or_dict: pd.DataFrame
    column_id: str = "id"
    column_sort: str = "time"
    max_timeshift: int = 30
    min_timeshift: int = 30
    rolling_direction: int = 1
    n_jobs: int = 6


@dataclass
class TargetRollParam:
    df_or_dict: pd.DataFrame
    column_id: str = "id"
    column_sort: str = "time"
    max_timeshift: int = 15
    min_timeshift: int = 15
    rolling_direction: int = -1
    n_jobs: int = 6


@dataclass
class ExtractFeaturesParams:
    timeseries_container: pd.DataFrame
    default_fc_parameters: ComprehensiveFCParameters = MinimalFCParameters()
    column_id: str = "id"
    column_sort: str = "date"
    impute_function: callable = impute
    n_jobs: int = 6


@dataclass
class InitDataLoaderParam:
    train_roll_param: TrainRollParam
    target_roll_param: TargetRollParam
    extract_features_param: ExtractFeaturesParams
