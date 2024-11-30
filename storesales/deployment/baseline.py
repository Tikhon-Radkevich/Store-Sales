import os
import shutil

import pandas as pd

from storesales.deployment.deployment_config import (
    deployment_date,
    families_to_deploy,
)
from deployment.constants import (
    BASELINE_DATA_DIR_PATH,
    BASELINE_MODELS_DIR_PATH,
)
from storesales.constants import (
    BASELINE_MODELS_PATH,
    EXTERNAL_TRAIN_PATH,
)


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
