from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import numpy as np

import optuna
from sklearn.model_selection import TimeSeriesSplit

from storesales.baseline.model_wrappers import ModelBaseWrapper


class SalesPredictor:
    def __init__(
        self,
        model_wrappers: dict[str, ModelBaseWrapper],
        outer_cutoffs: list[pd.Timestamp],
        inner_cutoffs: list[int],
        family_groups: list[tuple[str]],
        outer_cv: TimeSeriesSplit,
        optuna_optimize_kwargs: dict,
        n_group_store_family_choices: int = 4,
        n_single_store_family_choices: int = 2,
        horizon: str = "16 days",
    ):
        self.model_wrappers = model_wrappers
        self.inner_cutoffs = inner_cutoffs
        self.outer_cutoffs = outer_cutoffs
        self.family_groups = family_groups
        self.outer_cv = outer_cv
        self.optuna_optimize_kwargs = optuna_optimize_kwargs
        self.n_group_store_family_choices = n_group_store_family_choices
        self.n_single_store_family_choices = n_single_store_family_choices
        self.horizon = horizon

        self.tune_storage = self._initialize_tune_storage()
        self.family_to_madel_params_storage = {}
        self.store_family_to_model_storage = {}
        self.tune_loss_storage = self._initialize_tune_loss_storage()

    def _initialize_tune_storage(self) -> dict:
        return {family_group: defaultdict(list) for family_group in self.family_groups}

    def _initialize_tune_loss_storage(self) -> dict:
        return {
            family_group: {
                "sample_losses": defaultdict(list),
                "fold_losses": defaultdict(list),
            }
            for family_group in self.family_groups
        }

    def update_tune_loss_storage(
        self, family_group: tuple[str], loss: list[float], i_sample: int, i_fold: int
    ) -> None:
        self.tune_loss_storage[family_group]["sample_losses"][i_sample] += loss
        self.tune_loss_storage[family_group]["fold_losses"][i_fold] += loss

    def get_n_store_family_choices(self, family_group: tuple[str]) -> int:
        if len(family_group) == 1:
            return self.n_single_store_family_choices
        return self.n_group_store_family_choices

    def fit(self, train: pd.DataFrame, initial: str = "760 days") -> None:
        if not self.family_to_madel_params_storage:
            raise ValueError("Sales Predictor has not been tuned.")

        store_nbrs = train["store_nbr"].unique()

        for family, model_params in self.family_to_madel_params_storage.items():
            for store_nbr in store_nbrs:
                model = self.get_best_model(model_params["params"])
                self.store_family_to_model_storage[(store_nbr, family)] = model

        train_slice = train[train["ds"] >= train["ds"].max() - pd.Timedelta(initial)]

        for (store_nbr, family), model in tqdm(
            self.store_family_to_model_storage.items()
        ):
            x_train = train_slice[
                (train_slice["store_nbr"] == store_nbr)
                & (train_slice["family"] == family)
            ].copy()
            x_train.sort_values(by="ds", inplace=True)
            model.fit(x_train)

    def predict(self, test: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
        prediction_list = []

        for (store_nbr, family), model in tqdm(
            self.store_family_to_model_storage.items()
        ):
            x_test = test[(test["store_nbr"] == store_nbr) & (test["family"] == family)]

            forecast = model.predict(x_test)[["ds", "yhat"]]

            forecast["store_nbr"] = store_nbr
            forecast["family"] = family

            x_test = x_test.merge(
                forecast, on=["store_nbr", "family", "ds"], how="left"
            )
            prediction_list.append(x_test)

        predictions = pd.concat(prediction_list, ignore_index=True)

        predictions.set_index("id", inplace=True)
        submission["sales"] = predictions["yhat"]

        return submission

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
            self.family_to_madel_params_storage[family] = {
                "params": best_mean_params,
                "loss": loss,
            }

    def objective(self, trial: optuna.Trial, train: pd.DataFrame) -> float:
        model_name = trial.suggest_categorical(
            "model", list(self.model_wrappers.keys())
        )
        cutoffs = train["ds"].iloc[self.inner_cutoffs].reset_index(drop=True)
        return self.model_wrappers[model_name].objective(
            trial, train, cutoffs, self.horizon
        )

    def get_best_model(self, best_params):
        best_params = best_params.copy()
        model_name = best_params.pop("model")
        return self.model_wrappers[model_name].get_model(**best_params)

    def render_model(self, params) -> str:
        return self.model_wrappers[params["model"]].render(**params)
