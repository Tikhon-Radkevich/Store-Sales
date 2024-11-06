from joblib import Parallel, delayed

import optuna
from darts.models import LightGBMModel

from storesales.light_gbm.lgb_tuner.family_lgb_params import FamilyLightGBMModelParams
from storesales.light_gbm.fit_evaluate.evaluate_models import evaluate
from storesales.light_gbm.dataset import FamilyDataset
from storesales.constants import VALIDATION_DATE_RANGE


class LightGBMModelTuner:
    def __init__(self, dataset: dict[str, FamilyDataset], families: list[str]):
        self.dataset = dataset
        self.families = families

        self.studies_dict = self._initialize_studies()
        self._show_progress_bar_dict = (
            self._initialize_show_progress_bar()
        )  # todo: one progress bar

    def run_parallel_tune(self, eval_stride: int = 5, n_trials: int = 10) -> None:
        tuned_studies = Parallel(n_jobs=len(self.families))(
            delayed(self._parallel_optuna_study)(f, eval_stride, n_trials)
            for f in self.families
        )
        for family, study in zip(self.families, tuned_studies):
            self.studies_dict[family] = study
        return

    def fit_best(self) -> dict[str, LightGBMModel]:
        best_models = {}
        for family in self.families:
            best_params = self.studies_dict[family].best_params
            best_params.update(FamilyLightGBMModelParams().kwargs)
            best_model = LightGBMModel(**best_params)
            best_model.fit(**self.dataset[family].get_train_inputs())
            best_models[family] = best_model
        return best_models

    def _objective(self, trial: optuna.Trial, family: str, stride: int) -> float:
        light_gb_model_kwargs = FamilyLightGBMModelParams().suggest(trial)

        model = LightGBMModel(**light_gb_model_kwargs)
        model.fit(**self.dataset[family].get_train_inputs())

        objective_loss_df = evaluate(
            dataset=self.dataset,
            evaluate_range=VALIDATION_DATE_RANGE,
            models={family: model},
            stride=stride,
        )

        return objective_loss_df.values.mean()

    def _parallel_optuna_study(
        self, family: str, stride: int, n_trials: int
    ) -> optuna.Study:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = self.studies_dict[family]
        study.optimize(
            lambda trial: self._objective(trial, family, stride),
            n_trials=n_trials,
            show_progress_bar=self._show_progress_bar_dict[family],
        )
        return study

    def _initialize_studies(self) -> dict[str, optuna.Study]:
        studies = {}
        for family in self.families:
            studies[family] = optuna.create_study(
                direction="minimize",
                study_name=f"{family}_study",
            )
        return studies

    def _initialize_show_progress_bar(self) -> dict[str, bool]:
        show_progress_dict = {family: False for family in self.families}
        show_progress_dict[self.families[0]] = (
            True  # Show progress bar for one family, when run in parallel
        )
        return show_progress_dict
