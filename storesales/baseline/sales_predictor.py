from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import numpy as np

import optuna
from prophet import Prophet
from prophet.diagnostics import cross_validation

from storesales.baseline.stat_models import DailyMeanModel


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))


class ModelWrapper:
    pass


class DailyMeanModelWrapper(ModelWrapper):
    def __init__(self):
        super().__init__()

    @staticmethod
    def render(**kwargs) -> str:
        return f"DailyMeanModel({kwargs['window']})"

    @staticmethod
    def get_model(**kwargs) -> DailyMeanModel:
        return DailyMeanModel(**kwargs)

    def objective(
        self, trial: optuna.Trial, df: pd.DataFrame, cutoffs: pd.DataFrame
    ) -> float:
        model = self.get_model(window=trial.suggest_int("window", 5, 50))

        losses = []
        for cutoff in cutoffs:
            train_condition = df["ds"] < cutoff
            train = df[train_condition]
            future = df[~train_condition][:16]

            model.fit(train)
            forecast = model.predict(future)

            y_pred = forecast["yhat"].values
            y_true = forecast["y"].values

            losses.append(rmsle(y_true, y_pred))

        return np.mean(losses)

    @staticmethod
    def get_best_model(best_params, best_model_name) -> dict:
        mean_params = {
            "window": int(np.mean(best_params["window"])),
            "model": best_model_name,
        }
        return mean_params


class ProphetWrapper(ModelWrapper):
    def __init__(
        self,
        extra_regressors: list[str] = None,
        holidays=None,
        horizon="16 days",
        initial="730 days",
    ):
        self.extra_regressors = extra_regressors
        self.holidays = holidays
        self.horizon = horizon
        self.initial = initial

        self.daily_seasonality = False
        self.weekly_seasonality = True
        self.yearly_seasonality = True
        self.uncertainty_samples = False

        super().__init__()

    def get_model(self, **kwargs) -> Prophet:
        prophet = Prophet(
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            uncertainty_samples=self.uncertainty_samples,
            holidays=self.holidays,
            **kwargs,
        )

        if self.extra_regressors is not None:
            for regressor in self.extra_regressors:
                prophet.add_regressor(regressor)

        return prophet

    def objective(
        self, trial: optuna.Trial, train: pd.DataFrame, cutoffs: pd.DataFrame
    ) -> float:
        model = self.get_model(
            # growth=trial.suggest_categorical("growth", ["linear", "flat"]),
            # n_changepoints=trial.suggest_int("n_changepoints", 2, 50),
            changepoint_prior_scale=trial.suggest_float(
                "changepoint_prior_scale", 0.001, 0.5
            ),
            seasonality_prior_scale=trial.suggest_int("seasonality_prior_scale", 2, 20),
            # seasonality_mode=trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        ).fit(train)

        df_cv = cross_validation(
            model,
            horizon=self.horizon,
            initial=self.initial,
            cutoffs=cutoffs,
            disable_tqdm=True,
            parallel="processes",
        )
        df_cv["yhat"] = df_cv["yhat"].clip(lower=0)
        loss = (
            df_cv.groupby("cutoff")[["y", "yhat"]]
            .apply(lambda group: rmsle(group["y"], group["yhat"]))
            .mean()
        )
        return loss

    @staticmethod
    def render(**kwargs) -> str:
        return "ProphetWrapper"

    @staticmethod
    def get_best_model(best_params, best_model_name) -> dict:
        mean_params = {
            "model": best_model_name,
            "changepoint_prior_scale": np.mean(best_params["changepoint_prior_scale"]),
            "seasonality_prior_scale": int(
                np.mean(best_params["seasonality_prior_scale"])
            ),
        }
        return mean_params


class SalesPredictor:
    def __init__(
        self,
        prophet_wrapper: ProphetWrapper,
        daily_wrapper: DailyMeanModel,
        inner_cutoffs: list[int],
        family_groups: list[set[str]],
    ):
        self.model_wrappers = {
            DailyMeanModelWrapper.__name__: daily_wrapper,
            ProphetWrapper.__name__: prophet_wrapper,
        }
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

        # if not self.models:
        for family, args in self.best_storage.items():
            print(family)
            print(args)
            model = self.get_best_model(args["params"])
            print(model)
            self.models[family] = model

        # todo: get dataset for each family-store from train
        for family, model in tqdm(self.models.items()):
            model.fit(train)

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

    def get_best_model(self, best_params) -> DailyMeanModel | Prophet:
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
