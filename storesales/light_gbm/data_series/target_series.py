import pandas as pd
from darts import TimeSeries


def get_series_and_id_dicts(df: pd.DataFrame, static_cols: list[str]):
    series_dict = {}
    series_id_dict = {}

    for family in df["family"].unique():
        series = TimeSeries.from_group_dataframe(
            df=df[df["family"] == family],
            time_col="date",
            value_cols="sales",
            group_cols="store_nbr",
            static_cols=static_cols,
        )
        series_id = [
            {"store_nbr": s.static_covariates.store_nbr.iloc[0], "family": family}
            for s in series
        ]
        series_id_dict[family] = series_id

        # series = [s.with_static_covariates(None) for s in series]

        series_dict[family] = series

    return series_dict, series_id_dict
