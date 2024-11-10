import numpy as np
import pandas as pd

from darts import TimeSeries


class FamilyDataset:
    def __init__(
        self,
        family: str,
        series: list[TimeSeries],
        stores: list[int],
        train_end_date: str,
        start_submission_date: str,
        future_covariates: list[TimeSeries] | None = None,
        past_covariates: list[TimeSeries] | None = None,
    ):
        self.family = family
        self.series = series
        self.stores = stores
        self.future_covariates = future_covariates
        self.past_covariates = past_covariates

        self.train_end_date = pd.Timestamp(train_end_date)
        self.start_submission_date = pd.Timestamp(start_submission_date)

        self.validation_range = None
        self.test_range = None

    def get_covariates(self):
        covariates = {
            "future_covariates": self.future_covariates,
            "past_covariates": self.past_covariates,
        }
        return covariates

    def get_inputs(self, series=None):
        inputs = {"series": series or self.series}
        inputs.update(self.get_covariates())
        return inputs

    def get_cut_inputs(self, split_date: pd.Timestamp):
        series = [s.drop_after(split_date) for s in self.series]
        return self.get_inputs(series=series)

    def get_train_inputs(self):
        return self.get_cut_inputs(self.train_end_date)

    def get_submission_inputs(self):
        return self.get_cut_inputs(self.start_submission_date)


def make_family_datasets(
    df: pd.DataFrame,
    featured_df: pd.DataFrame,
    static_cols: list[str],
    future_cols: list[str],
    past_cols: list[str],
    train_end_date: str,
    start_submission_date: str,
) -> dict[str, FamilyDataset]:
    family_datasets = {}

    target_family_groups = df.groupby("family")
    featured_family_groups = featured_df.groupby("family")

    for family in df["family"].unique():
        # target series
        series = TimeSeries.from_group_dataframe(
            df=target_family_groups.get_group(family),
            time_col="date",
            value_cols="sales",
            group_cols="store_nbr",
            static_cols=static_cols,
        )
        stores = [int(s.static_covariates.store_nbr.iloc[0]) for s in series]
        series = [s.astype(np.float32) for s in series]

        # future covariates
        future_covs = TimeSeries.from_group_dataframe(
            df=featured_family_groups.get_group(family),
            time_col="date",
            value_cols=future_cols,
            group_cols="store_nbr",
            drop_group_cols=["store_nbr"],
        )
        future_covs = [f.astype(np.float32) for f in future_covs]

        # past covariates
        past_covs = TimeSeries.from_group_dataframe(
            df=featured_family_groups.get_group(family),
            time_col="date",
            value_cols=past_cols,
            group_cols="store_nbr",
            drop_group_cols=["store_nbr"],
        )
        past_covs = [p.astype(np.float32) for p in past_covs]

        # Add FamilyDataset
        family_datasets[family] = FamilyDataset(
            family=family,
            series=series,
            stores=stores,
            future_covariates=future_covs,
            past_covariates=past_covs,
            train_end_date=train_end_date,
            start_submission_date=start_submission_date,
        )

    return family_datasets
