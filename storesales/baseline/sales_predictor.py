from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import numpy as np

import optuna

from storesales.baseline.model_wrappers import ModelBaseWrapper
from storesales.baseline.utils import rmsle


class SalesPredictor:
    def __init__(
        self,
        model_wrappers: dict[str, ModelBaseWrapper],
        inner_cutoffs: list[int],
        family_groups: list[set[str]],
    ):
        self.model_wrappers = model_wrappers
        self.inner_cutoffs = inner_cutoffs
        self.family_groups = family_groups

        self.tune_storage = self._initiate_tune_storage()
        self.best_storage = {}

        self.models = {}

    def _initiate_tune_storage(self):
        tune_storage = {}
        for family_group in self.family_groups:
            tune_storage[family_group] = defaultdict(list)
        return tune_storage

    def fit(self, train: pd.DataFrame) -> None:
        if not self.best_storage:
            raise ValueError("Sales Predictor has not been tuned.")

        store_nbrs = train["store_nbr"].unique()
        # if not self.models:
        for family, args in self.best_storage.items():
            for store_nbr in store_nbrs:
                model = self.get_best_model(args["params"])
                self.models[(store_nbr, family)] = model

        last_730_days = train[
            train["ds"] >= train["ds"].max() - pd.DateOffset(days=730)
        ]
        for (store_nbr, family), model in tqdm(self.models.items()):
            x_train = last_730_days[
                (last_730_days["store_nbr"] == store_nbr)
                & (last_730_days["family"] == family)
            ]
            model.fit(x_train)

    def predict(self, test: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
        prediction_list = []

        for (store_nbr, family), model in tqdm(self.models.items()):
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

    def evaluate_and_save_tune(
        self,
        family_group: set[str],
        best_params: dict,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> float:
        model = self.get_best_model(best_params)
        model.fit(train)
        forecast = model.predict(test)
        y_pred = forecast["yhat"].values
        y_true = test["y"].values
        loss = self.rmsle(y_true, y_pred)

        self.tune_storage[family_group]["loss"].append(loss)
        for key, value in best_params.items():
            self.tune_storage[family_group][key].append(value)
        return loss

    def log_best(self, family_group: set[str]):
        tune_run = self.tune_storage[family_group]
        loss = np.mean(tune_run["loss"])

        values, counts = np.unique(tune_run["model"], return_counts=True)
        best_model = values[np.argmax(counts)]

        best_params = self.model_wrappers[best_model].get_best_model(
            tune_run, best_model
        )

        for family in family_group:
            self.best_storage[family] = {"params": best_params, "loss": loss}

    def objective(self, trial: optuna.Trial, train: pd.DataFrame) -> float:
        model_name = trial.suggest_categorical(
            "model", list(self.model_wrappers.keys())
        )
        cutoffs = train["ds"].iloc[self.inner_cutoffs].reset_index(drop=True)
        return self.model_wrappers[model_name].objective(trial, train, cutoffs)

    def get_best_model(self, best_params) -> ModelBaseWrapper:
        best_params = best_params.copy()
        model_name = best_params.pop("model")
        return self.model_wrappers[model_name].get_model(**best_params)

    def render_model(self, params) -> str:
        return self.model_wrappers[params["model"]].render(**params)

    def get_best(self, best_model_name: str, run_storage) -> dict:
        best_params = run_storage[best_model_name]
        mean_params = self.model_wrappers[best_model_name].get_best_model(
            best_params, best_model_name
        )
        return mean_params

    def rmsle(self, y_true, y_pred):
        return rmsle(y_true, y_pred)
