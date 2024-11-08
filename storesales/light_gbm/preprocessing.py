import pandas as pd


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    # todo: make more native
    multi_idx = pd.MultiIndex.from_product(
        [
            pd.date_range(df["date"].min(), df["date"].max()),
            df.store_nbr.unique(),
            df.family.unique(),
        ],
        names=["date", "store_nbr", "family"],
    )
    train = (
        df.set_index(["date", "store_nbr", "family"]).reindex(multi_idx).reset_index()
    )

    train[["sales", "onpromotion"]] = train[["sales", "onpromotion"]].fillna(0.0)
    train["id"] = train["id"].interpolate(method="linear")

    return train


def remove_leading_zeros(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True)
    group = group.loc[group["sales"].ne(0).idxmax() :]
    return group


def replace_zero_gaps(group: pd.DataFrame, n: int) -> pd.DataFrame:
    # Replace zeros with None where gap size is greater than `n`
    group = group.sort_values("date").reset_index(drop=True)

    zero_gap = group["sales"] == 0
    zero_shift_series = pd.Series(zero_gap != zero_gap.shift())
    gap_size = zero_gap.groupby(zero_shift_series.cumsum()).transform("sum")

    group.loc[zero_gap & (gap_size > n), "sales"] = None
    return group


def interpolate_missing_sales(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True)
    group["sales"] = group["sales"].interpolate(method="linear", limit_direction="both")
    return group


def preprocess(df: pd.DataFrame, zero_gap_size_to_replace=10) -> pd.DataFrame:
    daily_df = make_daily(df)

    no_leading_zeros_df = (
        daily_df.groupby(["store_nbr", "family"])
        .apply(remove_leading_zeros, include_groups=False)
        .reset_index(level=["store_nbr", "family"])
    )

    no_zero_gaps_df = (
        no_leading_zeros_df.groupby(["store_nbr", "family"])
        .apply(replace_zero_gaps, zero_gap_size_to_replace, include_groups=False)
        .reset_index(level=["store_nbr", "family"])
    )

    interpolated_sales_df = (
        no_zero_gaps_df.groupby(["store_nbr", "family"])
        .apply(interpolate_missing_sales, include_groups=False)
        .reset_index(level=["store_nbr", "family"])
    )

    # todo: some series will have all NaNs.
    interpolated_sales_df.fillna(0, inplace=True)

    return interpolated_sales_df
