import pandas as pd
from darts import TimeSeries


def get_covariates_dicts(
    data_df: pd.DataFrame, future_cols: list[str], past_cols: list[str]
):
    future_dict = {}
    past_dict = {}

    # family_grouped = data_df.groupby("family") todo
    for family in data_df["family"].unique():
        family_data = data_df[data_df["family"] == family]

        future_covariates = TimeSeries.from_group_dataframe(
            df=family_data,
            time_col="date",
            value_cols=future_cols,
            group_cols="store_nbr",
        )
        future_dict[family] = [
            f.with_static_covariates(None) for f in future_covariates
        ]

        past_covariates = TimeSeries.from_group_dataframe(
            df=family_data,
            time_col="date",
            value_cols=past_cols,
            group_cols="store_nbr",
            static_cols=None,
        )

        past_dict[family] = [p.with_static_covariates(None) for p in past_covariates]

    return future_dict, past_dict
