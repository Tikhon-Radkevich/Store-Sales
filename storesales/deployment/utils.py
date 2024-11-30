import os
import shutil

import pandas as pd

from storesales.deployment.deployment_config import families_to_deploy
from storesales.constants import FAMILY_STORE_TO_MODEL_CSV_PATH
from deployment.constants import (
    DATA_DIR_PATH,
    MODELS_DIR_PATH,
    BASELINE_DATA_DIR_PATH,
    LIGHTGBM_DATA_DIR_PATH,
    BASELINE_MODELS_DIR_PATH,
    LIGHTGBM_MODELS_DIR_PATH,
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


def select_family_store_to_model_csv() -> None:
    data_df = pd.read_csv(FAMILY_STORE_TO_MODEL_CSV_PATH, index_col=False)
    data_df = data_df[data_df["family"].isin(families_to_deploy)]

    file_name = os.path.basename(FAMILY_STORE_TO_MODEL_CSV_PATH)
    file_path = os.path.join(MODELS_DIR_PATH, file_name)
    data_df.to_csv(file_path, index=False)
