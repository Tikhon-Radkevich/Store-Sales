import os

from deployment.storesales.sales_predictor import SalesPredictor

from deployment.constants import (
    LIGHTGBM_FAMILIES,
    LIGHTGBM_MODELS_DIR_PATH,
    LIGHTGBM_DATA_DIR_PATH,
    BASELINE_MODELS_DIR_PATH,
    BASELINE_MODEL_FILE_NAMES,
    BASELINE_MODEL_NAMES,
    BASELINE_DATA_FILE_PATH,
    FAMILY_STORE_TO_MODEL_CSV_FILE_PATH,
)


def initialize_sales_predictor() -> SalesPredictor:
    sales_predictor = SalesPredictor(
        lightgbms_info=get_lightgbms_info(),
        baseline_model_paths=get_baseline_model_paths(),
        baseline_model_names=BASELINE_MODEL_NAMES,
        baseline_data_file_path=BASELINE_DATA_FILE_PATH,
        family_store_to_model_csv_file_path=FAMILY_STORE_TO_MODEL_CSV_FILE_PATH,
    )

    return sales_predictor


def get_baseline_model_paths() -> list:
    baseline_model_paths = []
    for file_name in BASELINE_MODEL_FILE_NAMES:
        file_path = os.path.join(BASELINE_MODELS_DIR_PATH, file_name)
        baseline_model_paths.append(file_path)
    return baseline_model_paths


def get_lightgbms_info() -> dict:
    lightgbms_info = {}

    for family in LIGHTGBM_FAMILIES:
        model_path = os.path.join(LIGHTGBM_MODELS_DIR_PATH, family, "model.darts")
        data_path = os.path.join(LIGHTGBM_DATA_DIR_PATH, family, "family_dataset.pkl")

        lightgbms_info[family] = {
            "model_file_path": model_path,
            "data_file_path": data_path,
        }

    return lightgbms_info
