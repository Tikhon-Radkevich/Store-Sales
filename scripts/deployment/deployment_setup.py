import os
import shutil
import pickle

import pandas as pd

from storesales.light_gbm.dataset import FamilyDataset
from scripts.deployment.deployment_config import (
    deployment_date,
    families_to_deploy,
    lightgbm_families,
    lightgbm_suffixes,
)
from deployment.constants import (
    DATA_DIR_PATH,
    MODELS_DIR_PATH,
    BASELINE_DATA_DIR_PATH,
    LIGHTGBM_DATA_DIR_PATH,
    BASELINE_MODELS_DIR_PATH,
    LIGHTGBM_MODELS_DIR_PATH,
)
from storesales.constants import (
    LIGHT_GBM_MODELS_DIR_PATH,
    FAMILY_STORE_TO_MODEL_CSV_PATH,
    BASELINE_MODELS_PATH,
    EXTERNAL_TRAIN_PATH,
)


def prepare_dirs() -> None:
    if os.path.exists(DATA_DIR_PATH):
        shutil.rmtree(DATA_DIR_PATH)

    if os.path.exists(MODELS_DIR_PATH):
        shutil.rmtree(MODELS_DIR_PATH)

    os.makedirs(DATA_DIR_PATH)
    os.makedirs(BASELINE_DATA_DIR_PATH)
    os.makedirs(LIGHTGBM_DATA_DIR_PATH)

    os.makedirs(MODELS_DIR_PATH)
    os.makedirs(BASELINE_MODELS_DIR_PATH)
    os.makedirs(LIGHTGBM_MODELS_DIR_PATH)


def setup_baseline() -> None:
    for baseline_model in os.listdir(BASELINE_MODELS_PATH):
        model_path = os.path.join(BASELINE_MODELS_PATH, baseline_model)
        shutil.copy(model_path, BASELINE_MODELS_DIR_PATH)

    train_df = pd.read_csv(EXTERNAL_TRAIN_PATH, parse_dates=["date"])
    baseline_train_df = train_df.rename(columns={"date": "ds", "sales": "y"})
    baseline_train_df.drop(columns=["id", "onpromotion"], inplace=True)

    family_mask = baseline_train_df["family"].isin(families_to_deploy)
    date_mask = baseline_train_df["ds"] < deployment_date
    baseline_train_df = baseline_train_df[(family_mask & date_mask)]

    deployment_baseline_data_path = os.path.join(BASELINE_DATA_DIR_PATH, "data.csv")
    baseline_train_df.to_csv(deployment_baseline_data_path, index=False)


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


def copy_family_store_to_model_csv() -> None:
    shutil.copy(FAMILY_STORE_TO_MODEL_CSV_PATH, MODELS_DIR_PATH)


def main():
    prepare_dirs()
    setup_baseline()
    setup_lightgbm()
    copy_family_store_to_model_csv()


if __name__ == "__main__":
    main()
