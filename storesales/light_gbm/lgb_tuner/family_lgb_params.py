from dataclasses import dataclass
from lightgbm import LGBMRegressor

import optuna


@dataclass
class FamilyLightGBMModelParams:
    # extra_trees=True,
    # use_quantized_grad=True,
    # early_stopping_rounds=10,

    lags_future_covariates: list[int]
    lags_past_covariates: list[str]
    categorical_static_covariates: list[str]
    categorical_future_covariates: list[str]
    lags: int = 24
    data_sample_strategy: str = "goss"
    random_state: int = 42
    verbosity: int = -1
    n_jobs: int = 1
    force_row_wise: bool = True

    def suggest(self, trial: optuna.trial.BaseTrial) -> dict:
        suggestion = dict(
            num_leaves=trial.suggest_int("num_leaves", 16, 88),
            max_depth=trial.suggest_int("max_depth", 4, 26),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            top_rate=trial.suggest_float("top_rate", 0.1, 0.4),
            other_rate=trial.suggest_float("other_rate", 0.05, 0.2),
            max_bin=trial.suggest_int("max_bin", 48, 202),
            feature_fraction=trial.suggest_float("feature_fraction", 0.1, 0.7),
            min_gain_to_split=trial.suggest_float(
                "min_gain_to_split", 1e-3, 0.2, log=True
            ),
            max_cat_threshold=trial.suggest_int("max_cat_threshold", 8, 28),
        )
        suggestion.update(self.__dict__)
        return suggestion
