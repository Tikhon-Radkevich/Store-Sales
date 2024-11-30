from storesales.deployment.utils import prepare_dirs, select_family_store_to_model_csv
from storesales.deployment.baseline import setup_baseline
from storesales.deployment.light_gbm import setup_lightgbm


def setup_deployment() -> None:
    prepare_dirs()
    setup_baseline()
    setup_lightgbm()
    select_family_store_to_model_csv()
