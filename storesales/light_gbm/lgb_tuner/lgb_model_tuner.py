from joblib import Parallel, delayed

import pandas as pd
import optuna
from darts.models import LightGBMModel

from storesales.light_gbm.lgb_tuner.family_lgb_params import FamilyLightGBMModelParams
from storesales.light_gbm.fit_evaluate.evaluate_models import evaluate
from storesales.light_gbm.dataset import FamilyDataset


class LightGBMModelTuner:
    def __init__(
        self,
        family_datasets: dict[str, FamilyDataset],
        families: list[str],
        param_suggestor: FamilyLightGBMModelParams,
    ):
        self.family_datasets = family_datasets
        self.families = families
        self.param_suggestor = param_suggestor

        self.studies_dict = self._initialize_studies()

    def run_tune(
        self,
        evaluate_range: pd.DatetimeIndex,
        eval_stride: int = 5,
        n_trials: int = 10,
    ) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        for family in self.families:
            self.studies_dict[family].optimize(
                lambda trial: self._objective(
                    trial, family, evaluate_range, eval_stride
                ),
                n_trials=n_trials,
                show_progress_bar=True,
            )

    def parallel_fit_best(self, n_jobs: int) -> dict[str, LightGBMModel]:
        best_models = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_best)(family) for family in self.families
        )
        return dict(best_models)

    def _fit_best(self, family: str) -> tuple[str, LightGBMModel]:
        best_trial = self.studies_dict[family].best_trial
        best_params = self.param_suggestor.suggest(best_trial)

        best_model = LightGBMModel(**best_params)
        best_model.fit(**self.family_datasets[family].get_train_inputs())
        return family, best_model

    def _objective(
        self,
        trial: optuna.Trial,
        family: str,
        evaluate_range: pd.DatetimeIndex,
        stride: int,
    ) -> float:
        light_gb_model_kwargs = self.param_suggestor.suggest(trial)

        model = LightGBMModel(**light_gb_model_kwargs)
        model.fit(**self.family_datasets[family].get_train_inputs())

        objective_loss_df = evaluate(
            dataset=self.family_datasets,
            evaluate_range=evaluate_range,
            models={family: model},
            stride=stride,
        )

        return objective_loss_df.values.mean()

    def _initialize_studies(self) -> dict[str, optuna.Study]:
        studies = {}
        for family in self.families:
            studies[family] = optuna.create_study(
                direction="minimize",
                study_name=f"{family}_study",
            )
        return studies
