from itertools import product
from collections import defaultdict

from tqdm import tqdm
import pandas as pd
import numpy as np

import optuna

from storesales.baseline.model_wrappers import ModelBaseWrapper


class SalesPredictor:
    def __init__(
        self,
        model_wrappers: dict[str, ModelBaseWrapper],
        outer_cutoffs: list[pd.Timestamp],
        inner_cutoffs: list[int],
        family_groups: list[tuple[str]],
        optuna_optimize_kwargs: dict,
        family_group_to_stores: dict,
        n_group_store_family_choices: int = 4,
        n_single_store_family_choices: int = 2,
        initial: str | None = "760 days",
        horizon: str = "16 days",
    ):
        self.model_wrappers = model_wrappers
        self.inner_cutoffs = inner_cutoffs
        self.outer_cutoffs = outer_cutoffs
        self.family_groups = family_groups
        self.optuna_optimize_kwargs = optuna_optimize_kwargs
        self.family_group_to_stores = family_group_to_stores
        self.n_group_store_family_choices = n_group_store_family_choices
        self.n_single_store_family_choices = n_single_store_family_choices
        self.initial = initial
        self.horizon = horizon

        self.tune_storage = self._initialize_tune_storage()
        self.store_family_pairs = self._initialize_store_family_pairs()
        self.family_to_model_params_storage = {}
        self.store_family_to_model_storage = {}
        self.store_family_loss_storage = defaultdict(list)
        self.tune_loss_storage = self._initialize_tune_loss_storage()

    def evaluate(self, test_dataset): ...

    def combine_with_predictor(self, predictor: "SalesPredictor") -> None:
        # will replace trained model with new ones from `predictor`
        # self.model_wrappers.update(predictor.model_wrappers)
        self.store_family_to_model_storage.update(
            predictor.store_family_to_model_storage
        )

    def update_tune_loss_storage(
        self, family_group: tuple[str], loss: list[float], i_sample: int, i_fold: int
    ) -> None:
        self.tune_loss_storage[family_group]["sample_losses"][i_sample] += loss
        self.tune_loss_storage[family_group]["fold_losses"][i_fold] += loss

    def get_n_store_family_choices(self, family_group: tuple[str]) -> int:
        if len(family_group) == 1:
            return self.n_single_store_family_choices
        return self.n_group_store_family_choices

    def fit(self, train: pd.DataFrame, disable_tqdm: bool = False) -> None:
        if not self.family_to_model_params_storage:
            raise ValueError("Sales Predictor has not been tuned.")

        if self.store_family_to_model_storage == {}:
            for family, model_params in self.family_to_model_params_storage.items():
                for store_nbr in model_params["stores"]:
                    model = self.get_best_model(model_params["params"])
                    self.store_family_to_model_storage[(store_nbr, family)] = model

        if self.initial is not None:
            train_slice = train[
                train["ds"] >= train["ds"].max() - pd.Timedelta(self.initial)
            ]
        else:
            train_slice = train

        x_train_groups = train_slice.groupby(["store_nbr", "family"])
        for (store_nbr, family), x_group in tqdm(x_train_groups, disable=disable_tqdm):
            model = self.store_family_to_model_storage[(store_nbr, family)]
            x_group.sort_values(by="ds", inplace=True)
            model.fit(x_group)

    def predict(self, test: pd.DataFrame, disable_tqdm: bool = False) -> pd.DataFrame:
        prediction_list = []

        x_test_groups = test.groupby(["store_nbr", "family"])
        for (store_nbr, family), group in tqdm(x_test_groups, disable=disable_tqdm):
            model = self.store_family_to_model_storage[(store_nbr, family)]

            forecast = model.predict(group)
            prediction_list.append(forecast)

        predictions = pd.concat(prediction_list, ignore_index=True)
        return predictions

    def log_study(
        self,
        family_group: tuple[str],
        best_params: dict,
        loss: float,
    ) -> None:
        self.tune_storage[family_group]["loss"].append(loss)
        for key, value in best_params.items():
            self.tune_storage[family_group][key].append(value)

    def calc_and_log_mean_params(self, family_group: tuple[str]):
        best_params = self.tune_storage[family_group]
        loss = np.mean(best_params["loss"])

        values, counts = np.unique(best_params["model"], return_counts=True)
        best_model = str(values[np.argmax(counts)])

        best_mean_params = self.model_wrappers[best_model].get_mean_params(
            best_params, best_model
        )

        for family in family_group:
            self.family_to_model_params_storage[family] = {
                "params": best_mean_params,
                "stores": self.family_group_to_stores[family_group],
                "loss": loss,
            }

    def objective(self, trial: optuna.Trial, train: pd.DataFrame) -> float:
        model_name = trial.suggest_categorical(
            "model", list(self.model_wrappers.keys())
        )
        train_length = len(train)
        # self.inner_cutoffs are indices of cutoffs in train data.
        # cutoff can be negative.
        valid_cutoffs = [
            i for i in self.inner_cutoffs if -train_length <= i < train_length
        ]
        cutoffs = train["ds"].iloc[valid_cutoffs].reset_index(drop=True)
        return self.model_wrappers[model_name].objective(
            trial, train, cutoffs, self.horizon
        )

    def get_best_model(self, best_params):
        best_params = best_params.copy()
        model_name = best_params.pop("model")
        return self.model_wrappers[model_name].get_model(**best_params)

    def render_model(self, params) -> str:
        return self.model_wrappers[params["model"]].render(**params)

    def _initialize_store_family_pairs(self):
        # save all possible store-family pairs for each family group.
        # will be used to sample store-family pair for tuning.
        store_family_pairs = {}
        for family_group, stores in self.family_group_to_stores.items():
            store_family_pairs[family_group] = list(product(stores, family_group))
        return store_family_pairs

    def _initialize_tune_storage(self) -> dict:
        # store best param samples from optuna studies for each family group
        return {family_group: defaultdict(list) for family_group in self.family_groups}

    def _initialize_tune_loss_storage(self) -> dict:
        return {
            family_group: {
                "sample_losses": defaultdict(list),
                "fold_losses": defaultdict(list),
            }
            for family_group in self.family_groups
        }
