from dataclasses import dataclass

import optuna


@dataclass
class FamilyLightGBMModelParams:
    kwargs = dict(
        lags=24,
        lags_future_covariates=[i for i in range(-9, 1, 3)],
        lags_past_covariates=[i for i in range(-25, -15, 3)],
        verbosity=-1,
        n_jobs=1,
    )

    def suggest(self, trial: optuna.Trial) -> dict:
        suggestion = dict(
            num_leaves=trial.suggest_int("num_leaves", 16, 512),
            max_depth=trial.suggest_int("max_depth", 6, 32),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
        )
        suggestion.update(self.kwargs)
        return suggestion
