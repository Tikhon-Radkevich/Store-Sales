import pandas as pd


def get_sales(
    train_data: pd.DataFrame, store_nbr: int | None = None, family: str | None = None
) -> pd.DataFrame:
    if store_nbr is None:
        store_con = pd.Series([True] * len(train_data))
    else:
        store_con = train_data["store_nbr"] == store_nbr
    if family is None:
        family_con = pd.Series([True] * len(train_data))
    else:
        family_con = train_data["family"] == family

    df = train_data[store_con & family_con]
    return df.groupby("date")["sales"].sum().reset_index()
