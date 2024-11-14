import pandas as pd


def make_daily(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Transform data to ensure daily granularity for each combination of date, store_nbr, and family.

    Fills missing values with 0 for 'sales' and 'onpromotion' columns.

    id column is not important for training, so it can be dropped.

    - drop_id: bool - if False, interpolation will be used to fill unique missing values.
    """
    index_names = ["date", "store_nbr", "family"]

    multi_idx = pd.MultiIndex.from_product(
        iterables=[
            pd.date_range(df["date"].min(), df["date"].max()),
            df.store_nbr.unique(),
            df.family.unique(),
        ],
        names=index_names,
    )
    train = df.set_index(index_names).reindex(multi_idx).reset_index()

    train[["sales", "onpromotion"]] = train[["sales", "onpromotion"]].fillna(0.0)

    if drop_id:
        train.drop(columns="id", inplace=True)
    else:
        train["id"] = train["id"].interpolate(method="linear")

    return train


def remove_leading_zeros(group: pd.DataFrame) -> pd.DataFrame:
    """Remove leading zeros from the beginning of the series,
    returning an empty DataFrame if all values are zero."""

    group = group.sort_values("date").reset_index(drop=True)
    is_not_zero = group["sales"].ne(0)

    if is_not_zero.any():
        first_not_zero_sale_inx = is_not_zero.idxmax()
        return group.loc[first_not_zero_sale_inx:]

    return pd.DataFrame()


def replace_zero_gaps(group: pd.DataFrame, n: int) -> pd.DataFrame:
    """Replace zeros with None where the gap size is greater than `n`."""

    group = group.sort_values("date").reset_index(drop=True)

    zero_gap = group["sales"] == 0
    zero_shift_series = pd.Series(zero_gap != zero_gap.shift())
    gap_size = zero_gap.groupby(zero_shift_series.cumsum()).transform("sum")

    group.loc[zero_gap & (gap_size > n), "sales"] = None
    return group


def interpolate_missing_sales(group: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing sales values linearly."""
    group = group.sort_values("date").reset_index(drop=True)
    group["sales"] = group["sales"].interpolate(method="linear", limit_direction="both")
    return group


def preprocess(df: pd.DataFrame, zero_gap_size_to_replace=10) -> pd.DataFrame:
    """
    Transformation:

    - make daily data with 0 filled missing values;
    - remove leading zeros (sequences after transforming will have different lengths);
    - replace zero gaps with None where gap size is greater than `zero_gap_size_to_replace`;
    - interpolate missing sales values.
    """
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

    return interpolated_sales_df
