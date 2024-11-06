# todo: remove this file

# import pandas as pd
# from darts import TimeSeries
#
# from storesales.light_gbm.dataset import FamilyDataset
# from storesales.constants import (
#     TRAIN_VALIDATION_SPLIT_DATE,
#     TRAIN_TEST_SPLIT_DATE,
#     END_TRAIN_DATE,
# )
#
#
# def make_dataset(
#     df: pd.DataFrame,
#     featured_df: pd.DataFrame,
#     static_cols: list[str],
#     future_cols: list[str],
#     past_cols: list[str],
#     horizon: int,
# ) -> dict[str, FamilyDataset]:
#     dataset = {}
#
#     target_family_groups = df.groupby("family")
#     featured_family_groups = featured_df.groupby("family")
#
#     for family in df["family"].unique():
#         # target series
#         series = TimeSeries.from_group_dataframe(
#             df=target_family_groups.get_group(family),
#             time_col="date",
#             value_cols="sales",
#             group_cols="store_nbr",
#             static_cols=static_cols,
#         )
#         _ = [s.static_covariates.astype(int) for s in series]
#         stores = [s.static_covariates.store_nbr.iloc[0] for s in series]
#
#         # future covariates
#         future_covs = TimeSeries.from_group_dataframe(
#             df=featured_family_groups.get_group(family),
#             time_col="date",
#             value_cols=future_cols,
#             group_cols="store_nbr",
#         )
#         future_covs = [f.with_static_covariates(None) for f in future_covs]
#
#         # past covariates
#         past_covs = TimeSeries.from_group_dataframe(
#             df=featured_family_groups.get_group(family),
#             time_col="date",
#             value_cols=past_cols,
#             group_cols="store_nbr",
#         )
#         past_covs = [p.with_static_covariates(None) for p in past_covs]
#
#         # Add FamilyDataset
#         family_dataset = FamilyDataset(
#             family=family,
#             series=series,
#             stores=stores,
#             future_covariates=future_covs,
#             past_covariates=past_covs,
#             train_end_date=TRAIN_VALIDATION_SPLIT_DATE,
#             # valid_end_date=TRAIN_TEST_SPLIT_DATE,
#             # test_end_date=END_TRAIN_DATE,
#         )
#         # family_dataset.set_validation_test_ranges(horizon)
#
#         dataset[family] = family_dataset
#
#     return dataset
