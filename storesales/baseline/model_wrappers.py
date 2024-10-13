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
        self.int_suggestions = int_suggestions or []
        self.float_suggestions = float_suggestions or []
        self.categorical_suggestions = categorical_suggestions or []
        self.model_base_params = model_base_params or {}
        self.estimator = estimator

    def get_model(self, **kwargs):
        params = {**self.model_base_params, **kwargs}
        model = self.estimator(**params)
        return self._process_model_before_fit(model)

    def objective(
        self, trial: optuna.Trial, df: pd.DataFrame, cutoffs: pd.DataFrame, horizon: str
    ) -> float:
        model = self.get_model(**self.suggest_params(trial))
        losses = []
        valid_end_cutoffs = cutoffs + pd.Timedelta(horizon)

        for valid_start, valid_end in zip(cutoffs, valid_end_cutoffs):
            train_mask = df["ds"] < valid_start
            valid_mask = (df["ds"] >= valid_start) & (df["ds"] < valid_end)

            train_data = df[train_mask]
            valid_data = df[valid_mask]

            model.fit(train_data)
            forecast = model.predict(valid_data)

            y_true = valid_data["y"].values
            y_pred = forecast["yhat"].values
            losses.append(rmsle(y_true, y_pred))

        return np.mean(losses)

    def suggest_params(self, trial: optuna.Trial) -> dict:
        suggested_params = {}

        for suggestion in self.int_suggestions:
            suggested_params[suggestion.name] = trial.suggest_int(
                suggestion.name, suggestion.low, suggestion.high
            )

        for suggestion in self.float_suggestions:
            suggested_params[suggestion.name] = trial.suggest_float(
                suggestion.name, suggestion.low, suggestion.high
            )

        for suggestion in self.categorical_suggestions:
            suggested_params[suggestion.name] = trial.suggest_categorical(
                suggestion.name, suggestion.choices
            )

        return suggested_params

    def get_mean_params(self, best_params: dict, model_name: str) -> dict:
        mean_params = {"model": model_name}

        for suggestion in self.int_suggestions:
            mean_params[suggestion.name] = int(np.mean(best_params[suggestion.name]))

        for suggestion in self.float_suggestions:
            mean_params[suggestion.name] = np.mean(best_params[suggestion.name])

        for suggestion in self.categorical_suggestions:
            values, counts = np.unique(best_params[suggestion.name], return_counts=True)
            mean_params[suggestion.name] = values[np.argmax(counts)]

        return mean_params

    def _process_model_before_fit(self, model):
        return model

    def render(self, **kwargs) -> str:
        raise NotImplementedError


class DailyMeanModelWrapper(ModelBaseWrapper):
    def __init__(self, int_suggestions: list[IntSuggestions]):
        super().__init__(estimator=DailyMeanModel, int_suggestions=int_suggestions)

    @staticmethod
    def render(**kwargs) -> str:
        return f"DailyMeanModel(window={kwargs['window']})"


class DayOfWeekMeanModelWrapper(ModelBaseWrapper):
    def __init__(self, int_suggestions: list[IntSuggestions]):
        super().__init__(estimator=DayOfWeekMeanModel, int_suggestions=int_suggestions)

    @staticmethod
    def render(**kwargs) -> str:
        return f"DOWMeanModel(weekdays={kwargs['weekdays_window']}, weekends={kwargs['weekends_window']})"


class ProphetWrapper(ModelBaseWrapper):
    def __init__(
        self,
        int_suggestions: list[IntSuggestions] = None,
        float_suggestions: list[FloatSuggestions] = None,
        categorical_suggestions: list[CategoricalSuggestions] = None,
        extra_regressors: list[str] = None,
        initial="730 days",
        model_base_params: dict = None,
    ):
        self.extra_regressors = extra_regressors or []
        self.initial = initial

        super().__init__(
            estimator=Prophet,
            model_base_params=model_base_params,
            int_suggestions=int_suggestions,
            float_suggestions=float_suggestions,
            categorical_suggestions=categorical_suggestions,
        )

    def _process_model_before_fit(self, model: Prophet) -> Prophet:
        for regressor in self.extra_regressors:
            model.add_regressor(regressor)
        return model

    def objective(
        self,
        trial: optuna.Trial,
        train: pd.DataFrame,
        cutoffs: pd.DataFrame,
        horizon: str,
    ) -> float:
        model = self.get_model(**self.suggest_params(trial))
        model.fit(train)

        df_cv = cross_validation(
            model,
            horizon=horizon,
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
