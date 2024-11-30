import os
import shutil
import pickle

import pandas as pd

from storesales.light_gbm.dataset import FamilyDataset
from storesales.deployment.deployment_config import (
    deployment_date,
    lightgbm_families,
    lightgbm_suffixes,
)
from deployment.constants import LIGHTGBM_DATA_DIR_PATH, LIGHTGBM_MODELS_DIR_PATH
from storesales.constants import LIGHT_GBM_MODELS_DIR_PATH


def process_family_dataset(family_dataset_file_path: str) -> FamilyDataset:
    with open(family_dataset_file_path, "rb") as file:
        family_dataset: FamilyDataset = pickle.load(file)

    deployment_timestamp = pd.Timestamp(deployment_date)
    family_dataset.series = [
        s.drop_after(deployment_timestamp) for s in family_dataset.series
    ]
    return family_dataset


def setup_lightgbm() -> None:
    for family, suffix in zip(lightgbm_families, lightgbm_suffixes):
        family = family.replace("/", "_")

        model_dataset_dir_path = os.path.join(
            LIGHT_GBM_MODELS_DIR_PATH, f"{family}{suffix}"
        )
        family_dataset_file_path = os.path.join(
            model_dataset_dir_path, "family_dataset.pkl"
        )

        # process family dataset and save it to deployment data directory
        family_dataset = process_family_dataset(family_dataset_file_path)

        deployment_family_dataset_dir_path = str(
            os.path.join(LIGHTGBM_DATA_DIR_PATH, family)
        )
        os.makedirs(deployment_family_dataset_dir_path)
        family_dataset_file_path = os.path.join(
            deployment_family_dataset_dir_path, "family_dataset.pkl"
        )
        with open(family_dataset_file_path, "wb") as file:
            pickle.dump(family_dataset, file)

        # copy model to deployment models directory
        model_file_path = os.path.join(model_dataset_dir_path, "model.darts")
        model_deployment_dir_path = str(os.path.join(LIGHTGBM_MODELS_DIR_PATH, family))
        os.makedirs(model_deployment_dir_path)
        shutil.copy(model_file_path, model_deployment_dir_path)
