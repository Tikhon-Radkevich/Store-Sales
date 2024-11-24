from dataclasses import dataclass

import optuna


@dataclass
class FamilyLightGBMModelParams:
    lags: dict[str, int]
    lags_future_covariates: list[int]
    lags_past_covariates: list[str]
    categorical_static_covariates: list[str] | None = None
    categorical_future_covariates: list[str] | None = None
    data_sample_strategy: str = "goss"
    random_state: int = 42
    verbosity: int = -1
    n_jobs: int = 1
    force_row_wise: bool = True
    show_warnings: bool = False

    def suggest(self, trial: optuna.trial.BaseTrial) -> dict:
        suggestion = dict(
            num_leaves=trial.suggest_int("num_leaves", 8, 144),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 50, 1000),
            top_rate=trial.suggest_float("top_rate", 0.05, 0.45),
            other_rate=trial.suggest_float("other_rate", 0.05, 0.45),
            max_bin=trial.suggest_int("max_bin", 200, 300),
            feature_fraction=trial.suggest_float("feature_fraction", 0.02, 0.20),
            min_gain_to_split=trial.suggest_float(
                "min_gain_to_split", 1e-5, 1e-3, log=True
            ),
            max_cat_threshold=trial.suggest_int("max_cat_threshold", 3, 32),
        )
        suggestion.update(self.__dict__)
        return suggestion


@dataclass
class FamilyLightGBMModelBaseParams:
    lags: dict[str, int] | int
    lags_future_covariates: list[int]
    lags_past_covariates: list[str]
    categorical_static_covariates: list[str] | None = None
    categorical_future_covariates: list[str] | None = None
    data_sample_strategy: str = "bagging"
    random_state: int = 42
    verbosity: int = -1
    n_jobs: int = 1
    force_row_wise: bool = True
    show_warnings: bool = False

    def suggest(self, _: optuna.trial.BaseTrial) -> dict:
        return self.__dict__
