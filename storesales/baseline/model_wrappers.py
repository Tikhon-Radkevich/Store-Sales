import pandas as pd
import numpy as np
import optuna

from prophet import Prophet
from prophet.diagnostics import cross_validation

from storesales.baseline.stat_models import DailyMeanModel, DayOfWeekMeanModel
from storesales.baseline.param_suggestions import (
    IntSuggestions,
    FloatSuggestions,
    CategoricalSuggestions,
)
from storesales.baseline.loss import rmsle


class ModelBaseWrapper:
    def __init__(
        self,
        estimator,
        int_suggestions: list[IntSuggestions] = None,
        float_suggestions: list[FloatSuggestions] = None,
        categorical_suggestions: list[CategoricalSuggestions] = None,
        model_base_params: dict = None,
    ):
        self.int_suggestions = int_suggestions
        self.float_suggestions = float_suggestions
        self.categorical_suggestions = categorical_suggestions
        self.model_base_params = (
            model_base_params if model_base_params is not None else {}
        )
        self.estimator = estimator

    def get_model(self, **kwargs):
        model_params = self.model_base_params.copy()
        model_params.update(kwargs)

        model = self.estimator(**model_params)

        model = self._process_model_before_fit(model)

        return model

    def objective(
        self, trial: optuna.Trial, df: pd.DataFrame, cutoffs: pd.DataFrame
    ) -> float:
        model = self.get_model(**self.suggest_params(trial))

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

    def _process_model_before_fit(self, model):
        return model

    def suggest_params(self, trial: optuna.Trial) -> dict:
        params = {}
        if self.int_suggestions:
            for suggestion in self.int_suggestions:
                params[suggestion.name] = trial.suggest_int(
                    suggestion.name, suggestion.low, suggestion.high
                )
        if self.float_suggestions:
            for suggestion in self.float_suggestions:
                params[suggestion.name] = trial.suggest_float(
                    suggestion.name, suggestion.low, suggestion.high
                )
        if self.categorical_suggestions:
            for suggestion in self.categorical_suggestions:
                params[suggestion.name] = trial.suggest_categorical(
                    suggestion.name, suggestion.choices
                )
        return params

    def get_mean_params(self, best_params: dict, best_model_name: str) -> dict:
        mean_params = {"model": best_model_name}

        if self.int_suggestions:
            for suggestion in self.int_suggestions:
                mean_params[suggestion.name] = int(
                    np.mean(best_params[suggestion.name])
                )
        if self.float_suggestions:
            for suggestion in self.float_suggestions:
                mean_params[suggestion.name] = np.mean(best_params[suggestion.name])
        if self.categorical_suggestions:
            for suggestion in self.categorical_suggestions:
                values, counts = np.unique(
                    best_params[suggestion.name], return_counts=True
                )
                mean_params[suggestion.name] = values[np.argmax(counts)]

        return mean_params

    def render(self, **kwargs) -> str:
        raise NotImplementedError


class DailyMeanModelWrapper(ModelBaseWrapper):
    def __init__(self, int_suggestions: list[IntSuggestions]):
        self.estimator = DailyMeanModel

        super().__init__(int_suggestions=int_suggestions, estimator=self.estimator)

    @staticmethod
    def render(**kwargs) -> str:
        return f"DailyMeanModel({kwargs['window']})"


class DayOfWeekMeanModelWrapper(ModelBaseWrapper):
    def __init__(self, int_suggestions: list[IntSuggestions]):
        self.estimator = DayOfWeekMeanModel

        super().__init__(int_suggestions=int_suggestions, estimator=self.estimator)

    @staticmethod
    def render(**kwargs) -> str:
        return f"DOWMeanModel(weekdays: {kwargs['weekdays_window']}, weekends: {kwargs['weekends_window']})"


class ProphetWrapper(ModelBaseWrapper):
    def __init__(
        self,
        int_suggestions: list[IntSuggestions] = None,
        float_suggestions: list[FloatSuggestions] = None,
        categorical_suggestions: list[CategoricalSuggestions] = None,
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

        self.model_params = {
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "uncertainty_samples": self.uncertainty_samples,
            "holidays": self.holidays,
        }

        super().__init__(
            estimator=Prophet,
            model_base_params=self.model_params,
            int_suggestions=int_suggestions,
            float_suggestions=float_suggestions,
            categorical_suggestions=categorical_suggestions,
        )

    def _process_model_before_fit(self, model: Prophet) -> Prophet:
        if self.extra_regressors is not None:
            for regressor in self.extra_regressors:
                model.add_regressor(regressor)
        return model

    def objective(
        self, trial: optuna.Trial, train: pd.DataFrame, cutoffs: pd.DataFrame
    ) -> float:
        model = self.get_model(**self.suggest_params(trial))
        model.fit(train[train["ds"] > train["ds"].max() - pd.Timedelta(self.initial)])

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
