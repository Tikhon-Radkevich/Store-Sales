import pandas as pd
from sklearn.model_selection import train_test_split
from storesales.light_gbm.tsfresh_processor import extract_features, roll_time_series

from storesales.light_gbm.param_dataclasses import InitDataLoaderParam


class DataLoader:
    def __init__(
        self,
        data_df: pd.DataFrame,
        init_dataloader_param: InitDataLoaderParam,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_df = data_df

        self.train_rolls = None
        self.train_featured = None
        self.target_grouped = None
        self._init(init_dataloader_param)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.train_featured,
            self.target_grouped,
            test_size=test_size,
            random_state=random_state,
        )

        self.validation_storage = {}

    def _init(self, init_dataloader_param: InitDataLoaderParam):
        train_rolls = roll_time_series(
            **init_dataloader_param.train_roll_param.__dict__
        )

        # Make Train Rolls Compatible with Target Rolls
        train_rolls["id"] = train_rolls["id"].apply(
            lambda x: (x[0], x[1] + pd.Timedelta("1 day"))
        )

        target_rolls = roll_time_series(
            **init_dataloader_param.target_roll_param.__dict__
        )

        comon_ids = set(train_rolls["id"].unique()).intersection(
            target_rolls["id"].unique()
        )
        self.train_rolls = train_rolls[train_rolls["id"].isin(comon_ids)]
        self.train_rolls.index.names = ["id", "time"]

        target_rolls = target_rolls[target_rolls["id"].isin(comon_ids)]

        self.target_grouped = target_rolls.groupby("id")["sales"].apply(list)
        self.target_grouped.index = pd.MultiIndex.from_tuples(
            self.target_grouped.index, names=["id", "time"]
        )

        self.train_featured = extract_features(
            self.train_rolls, **init_dataloader_param.extract_features_param.__dict__
        )

        self.train_featured.index.names = ["id", "time"]

        self.train_featured.columns = self.train_featured.columns.str.replace(
            r"[^\w\s]", "", regex=True
        )
        self.train_featured.columns = self.train_featured.columns.str.strip()
        self.train_featured.columns = self.train_featured.columns.str.replace(" ", "_")
