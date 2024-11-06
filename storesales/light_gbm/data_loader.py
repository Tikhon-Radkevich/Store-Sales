import pandas as pd
from sklearn.model_selection import train_test_split

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series

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

        self.train_rolls: pd.DataFrame
        self.train_featured: pd.DataFrame
        self.target_grouped: pd.DataFrame
        self.all_train_rolls: pd.DataFrame

        self.train_roll_dict_param = init_dataloader_param.train_roll_param.__dict__
        self.target_roll_dict_param = init_dataloader_param.target_roll_param.__dict__
        self.extract_features_dict_param = (
            init_dataloader_param.extract_features_param.__dict__
        )

        self._init()

        train_ids, valid_ids = train_test_split(
            self.train_featured.index,
            test_size=test_size,
            random_state=random_state,
        )

        self.X_train = self.train_featured.loc[train_ids]
        self.X_valid = self.train_featured.loc[valid_ids]
        self.y_train = self.target_grouped.loc[train_ids]
        self.y_valid = self.target_grouped.loc[valid_ids]

        self.validation_storage = {}

    def get_fit_target(self):
        return self.y_train.groupby(level=["id", "time"])["sales"].head(1)

    def _init(self):
        all_train_rolls = roll_time_series(**self.train_roll_dict_param)

        # Make Train Rolls Compatible with Target Rolls
        all_train_rolls["id"] = all_train_rolls["id"].apply(
            lambda x: (x[0], x[1] + pd.Timedelta("1 day"))
        )

        target_rolls = roll_time_series(**self.target_roll_dict_param)

        comon_ids = set(all_train_rolls["id"].unique()).intersection(
            target_rolls["id"].unique()
        )
        self.train_rolls = all_train_rolls[all_train_rolls["id"].isin(comon_ids)]
        self.train_featured = self.extract_tsfresh_features(self.train_rolls)

        all_train_rolls.index = pd.MultiIndex.from_tuples(all_train_rolls["id"])
        all_train_rolls.index.names = ["id", "time"]
        self.all_train_rolls = all_train_rolls

        target_rolls = target_rolls[target_rolls["id"].isin(comon_ids)]
        target_rolls.index = pd.MultiIndex.from_tuples(target_rolls["id"])
        target_rolls.index.names = ["id", "time"]
        self.target_grouped = target_rolls

        self.train_rolls.index = pd.MultiIndex.from_tuples(self.train_rolls["id"])
        self.train_rolls.index.names = ["id", "time"]

    def extract_tsfresh_features(self, rolls):
        train_featured = extract_features(rolls, **self.extract_features_dict_param)

        train_featured.index.names = ["id", "time"]

        train_featured.columns = train_featured.columns.str.replace(
            r"[^\w\s]", "", regex=True
        )
        train_featured.columns = train_featured.columns.str.strip()
        train_featured.columns = train_featured.columns.str.replace(" ", "_")

        return train_featured
