import pandas as pd


class SalesPromoScaler:
    """Class to scale `sales` and `onpromotion` values."""

    def __init__(self):
        self.min_sales = dict()
        self.max_sales = dict()
        self.min_promo = dict()
        self.max_promo = dict()

    @staticmethod
    def min_max_scale(
        values: pd.Series, min_val: float, max_val: float, epsilon: float = 1e-9
    ) -> pd.Series:
        return (values - min_val) / (max_val - min_val + epsilon)

    def fit(self, df: pd.DataFrame) -> None:
        for (family, store_nbr), group in df.groupby(["family", "store_nbr"]):
            self.min_sales[(family, store_nbr)] = group["sales"].min()
            self.max_sales[(family, store_nbr)] = group["sales"].max()

            self.min_promo[(family, store_nbr)] = group["onpromotion"].min()
            self.max_promo[(family, store_nbr)] = group["onpromotion"].max()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for (family, store_nbr), group in df.groupby(["family", "store_nbr"]):
            df.loc[group.index, "sales"] = self.min_max_scale(
                values=group["sales"],
                min_val=self.min_sales[(family, store_nbr)],
                max_val=self.max_sales[(family, store_nbr)],
            )
            df.loc[group.index, "onpromotion"] = self.min_max_scale(
                values=group["onpromotion"],
                min_val=self.min_promo[(family, store_nbr)],
                max_val=self.max_promo[(family, store_nbr)],
            )
        return df

    def inverse_transform_by_key(
        self, sales: pd.Series, family: str, store_nbr: int
    ) -> pd.Series:
        numerator = sales * (
            self.max_sales[(family, store_nbr)] - self.min_sales[(family, store_nbr)]
        )
        return numerator + self.min_sales[(family, store_nbr)]
