import pandas as pd
from darts import TimeSeries


def get_covariates_dicts(
    data_df: pd.DataFrame, future_cols: list[str], past_cols: list[str]
):
    future_dict = {}
    past_dict = {}

    for family, family_data in data_df.groupby("family"):
        future_covs = TimeSeries.from_group_dataframe(
            df=family_data,
            time_col="date",
            value_cols=future_cols,
            group_cols="store_nbr",
        )
        future_dict[family] = [f.with_static_covariates(None) for f in future_covs]

        past_covs = TimeSeries.from_group_dataframe(
            df=family_data,
            time_col="date",
            value_cols=past_cols,
            group_cols="store_nbr",
        )

        past_dict[family] = [p.with_static_covariates(None) for p in past_covs]

    return future_dict, past_dict
