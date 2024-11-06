# todo: remove this file

# import pandas as pd
# from darts import TimeSeries
#
#
# def add_covariates_dicts(
#     dataset, data_df: pd.DataFrame, future_cols: list[str], past_cols: list[str]
# ) -> None:
#     for family, family_data in data_df.groupby("family"):
#         future_covs = TimeSeries.from_group_dataframe(
#             df=family_data,
#             time_col="date",
#             value_cols=future_cols,
#             group_cols="store_nbr",
#         )
#         future_covs = [f.with_static_covariates(None) for f in future_covs]
#         dataset[family].future_covariates = future_covs
#
#         past_covs = TimeSeries.from_group_dataframe(
#             df=family_data,
#             time_col="date",
#             value_cols=past_cols,
#             group_cols="store_nbr",
#         )
#         past_covs = [p.with_static_covariates(None) for p in past_covs]
#         dataset[family].past_covariates = past_covs
