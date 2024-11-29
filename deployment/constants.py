import os


DEV_RUN_MODE = True

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR_PATH = os.path.join(ROOT_PATH, "data")
BASELINE_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "baseline")
LIGHTGBM_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, "light_gbm")
BASELINE_DATA_FILE_PATH = os.path.join(BASELINE_DATA_DIR_PATH, "data.csv")

MODELS_DIR_PATH = os.path.join(ROOT_PATH, "models")
BASELINE_MODELS_DIR_PATH = os.path.join(MODELS_DIR_PATH, "baseline")
LIGHTGBM_MODELS_DIR_PATH = os.path.join(MODELS_DIR_PATH, "light_gbm")

FAMILY_STORE_TO_MODEL_CSV_FILE_PATH = os.path.join(
    MODELS_DIR_PATH, "family_store_to_model.csv"
)

LIGHTGBM_FAMILIES = ["HOME AND KITCHEN I", "LIQUOR,WINE,BEER"]

BASELINE_MODEL_FILE_NAMES = [
    "daily_predictor.pkl",
    "day_of_week_predictor.pkl",
    "weighted_day_predictor.pkl",
]
BASELINE_MODEL_NAMES = ["daily", "day_of_week", "weighted_day"]
